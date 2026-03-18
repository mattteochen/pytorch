# mypy: allow-untyped-defs
import operator
from dataclasses import dataclass
from typing import Optional

import torch

from ..._dynamo.utils import counters
from ..kernel.gemm_plugin import has_gemm_plugin, make_gemm_plugin_lowering
from ..pattern_matcher import Match, PatternMatcherPass


aten = torch.ops.aten
linear = torch._C._nn.linear


GEMM_ROPE_PASS = PatternMatcherPass(pass_name="gemm_rope_pass")
GEMM_ROPE_PLUGIN = "rope_neox"


@dataclass
class RopeChainMatch:
    final_node: torch.fx.Node
    nodes: list[torch.fx.Node]
    cos: torch.fx.Node
    sin: torch.fx.Node
    head_dim: int
    rotary_dim: int


def _is_call(node: torch.fx.Node, target) -> bool:
    return node.op == "call_function" and node.target is target


def _is_call_function_target(node: torch.fx.Node, *targets) -> bool:
    return node.op == "call_function" and node.target in targets


def _is_call_method(node: torch.fx.Node, target: str) -> bool:
    return node.op == "call_method" and node.target == target


def _single_user(node: torch.fx.Node, target) -> Optional[torch.fx.Node]:
    matched = None
    for user in node.users:
        if _is_call(user, target) or _is_call_method(user, target):
            if matched is not None:
                return None
            matched = user
            continue
        if _is_call_method(user, "size") or _is_call(user, aten.sym_size.int):
            continue
        return None
    if matched is not None:
        return matched
    return None


def _node_size(node: torch.fx.Node) -> Optional[tuple[int, ...]]:
    val = node.meta.get("val", node.meta.get("example_value"))
    if val is None or not hasattr(val, "shape"):
        return None
    return tuple(int(x) for x in val.shape)


def _node_meta_tensor(node: torch.fx.Node):
    return node.meta.get("val", node.meta.get("example_value"))


def _find_slice_user(node: torch.fx.Node, dim: int, start: int, end: int) -> Optional[torch.fx.Node]:
    for user in node.users:
        if _is_call(user, aten.slice.Tensor) and tuple(user.args[1:4]) == (
            dim,
            start,
            end,
        ):
            return user
        if (
            _is_call(user, operator.getitem)
            and isinstance(user.args[1], tuple)
            and len(user.args[1]) == 2
            and user.args[1][0] is Ellipsis
        ):
            slice_arg = user.args[1][1]
            if (
                isinstance(slice_arg, slice)
                and dim == 2
                and (slice_arg.start or 0) == start
                and slice_arg.stop == end
            ):
                return user
    return None


def _find_split_pair(
    node: torch.fx.Node, split_size: int, dim: int
) -> Optional[tuple[torch.fx.Node, torch.fx.Node, torch.fx.Node]]:
    for user in node.users:
        if not _is_call(user, aten.split.Tensor):
            continue
        if user.args[1] != split_size or user.args[2] != dim:
            continue
        getitem0 = getitem1 = None
        for split_user in user.users:
            if not _is_call(split_user, operator.getitem):
                continue
            if split_user.args[1] == 0:
                getitem0 = split_user
            elif split_user.args[1] == 1:
                getitem1 = split_user
        if getitem0 is not None and getitem1 is not None:
            return user, getitem0, getitem1
    return None


def _find_binary_user(lhs: torch.fx.Node, rhs: torch.fx.Node, *targets) -> Optional[torch.fx.Node]:
    for user in lhs.users:
        if not _is_call_function_target(user, *targets):
            continue
        if user.args == (lhs, rhs):
            return user
    return None


def _match_cos_sin(node: torch.fx.Node):
    convert_node = None
    if _is_call(node, torch.ops.prims.convert_element_type.default):
        convert_node = node
        node = node.args[0]
    if not (_is_call(node, aten.unsqueeze.default) or _is_call_method(node, "unsqueeze")):
        return None
    unsqueeze_dim = node.args[1]
    if unsqueeze_dim not in (1, -2):
        return None
    slice_node = node.args[0]
    if _is_call(slice_node, aten.slice.Tensor):
        index_node = slice_node.args[0]
        if not _is_call(index_node, aten.index.Tensor):
            return None
        dim, start, end = slice_node.args[1:4]
        return index_node, dim, start, end, convert_node
    if (
        _is_call(slice_node, operator.getitem)
        and isinstance(slice_node.args[1], tuple)
        and len(slice_node.args[1]) == 2
        and slice_node.args[1][0] is Ellipsis
    ):
        index_node = slice_node.args[0]
        if not _is_call(index_node, operator.getitem):
            return None
        slice_arg = slice_node.args[1][1]
        if not isinstance(slice_arg, slice):
            return None
        return index_node, 1, slice_arg.start or 0, slice_arg.stop, convert_node
    if _is_call(slice_node, operator.getitem):
        split_node = slice_node.args[0]
        if not _is_call(split_node, aten.split.Tensor):
            return None
        index_node = split_node.args[0]
        if not _is_call(index_node, aten.index.Tensor):
            return None
        split_size, dim = split_node.args[1:3]
        if dim != -1:
            return None
        start = int(slice_node.args[1]) * int(split_size)
        end = start + int(split_size)
        return index_node, 1, start, end, convert_node
    return None


def _same_cos_sin_match(
    lhs, rhs
) -> bool:
    if lhs is None or rhs is None:
        return False
    return lhs[:4] == rhs[:4]


def _extract_view_shape(node: torch.fx.Node) -> Optional[tuple[object, ...]]:
    if _is_call(node, aten.view.default):
        return tuple(node.args[1])
    if _is_call(node, aten.reshape.default):
        return tuple(node.args[1])
    if _is_call_method(node, "view") or _is_call_method(node, "reshape"):
        return tuple(node.args[1:])
    return None


def _split_sizes_and_dim(node: torch.fx.Node) -> Optional[tuple[list[int], int]]:
    if _is_call(node, aten.split_with_sizes.default):
        split_sizes, split_dim = node.args[1], node.args[2]
        return [int(x) for x in split_sizes], int(split_dim)
    if _is_call_method(node, "split"):
        split_sizes = node.args[1]
        split_dim = -1 if len(node.args) < 3 else node.args[2]
        return [int(x) for x in split_sizes], int(split_dim)
    return None


def _match_rope_chain(split_output: torch.fx.Node) -> Optional[RopeChainMatch]:
    view = _single_user(split_output, "view")
    if view is None:
        view = _single_user(split_output, "reshape")
    if view is None:
        view = _single_user(split_output, aten.view.default)
    if view is None:
        view = _single_user(split_output, aten.reshape.default)
    if view is None:
        return None
    view_shape = _extract_view_shape(view)
    if view_shape is None:
        return None
    if len(view_shape) != 3 or int(view_shape[-1]) <= 0:
        return None
    head_dim = int(view_shape[-1])
    rope_left = _find_slice_user(view, 2, 0, head_dim // 2)
    rope_right = _find_slice_user(view, 2, head_dim // 2, head_dim)
    split_node = None
    if rope_left is None or rope_right is None:
        split_match = _find_split_pair(view, head_dim // 2, -1)
        if split_match is not None:
            split_node, rope_left, rope_right = split_match
    if rope_left is None or rope_right is None:
        return None

    left_users = list(rope_left.users)
    right_users = list(rope_right.users)
    if len(left_users) != 2 or len(right_users) != 2:
        return None

    cos = sin = None
    for node in left_users:
        if not _is_call_function_target(node, aten.mul.Tensor, operator.mul):
            return None
        matched = _match_cos_sin(node.args[1])
        if matched is None:
            return None
        _, dim, start, end, _ = matched
        if dim != 1:
            return None
        if start == 0:
            cos = node.args[1]
        else:
            sin = node.args[1]
    if cos is None or sin is None:
        return None

    mul_left_cos = _find_binary_user(rope_left, cos, aten.mul.Tensor, operator.mul)
    mul_left_sin = _find_binary_user(rope_left, sin, aten.mul.Tensor, operator.mul)
    mul_right_cos = _find_binary_user(rope_right, cos, aten.mul.Tensor, operator.mul)
    mul_right_sin = _find_binary_user(rope_right, sin, aten.mul.Tensor, operator.mul)
    if None in (mul_left_cos, mul_left_sin, mul_right_cos, mul_right_sin):
        return None

    sub = _find_binary_user(mul_left_cos, mul_right_sin, aten.sub.Tensor, operator.sub)
    add = _find_binary_user(mul_right_cos, mul_left_sin, aten.add.Tensor, operator.add)
    if sub is None or add is None:
        return None

    cat = None
    for user in sub.users:
        if _is_call(user, aten.cat.default) and user.args == ([sub, add], -1):
            cat = user
            break
        if (
            _is_call_function_target(user, torch.cat)
            and user.args[0] == (sub, add)
            and user.kwargs.get("dim", -1) == -1
        ):
            cat = user
            break
    if cat is None or len(add.users) != 1:
        return None
    final_view = _single_user(cat, "reshape")
    if final_view is None:
        final_view = _single_user(cat, aten.view.default)
    if final_view is None:
        final_view = _single_user(cat, aten.reshape.default)
    if final_view is None:
        return None

    nodes = [
        view,
        *( [split_node] if split_node is not None else [] ),
        rope_left,
        rope_right,
        mul_left_cos,
        mul_left_sin,
        mul_right_cos,
        mul_right_sin,
        sub,
        add,
        cat,
        final_view,
    ]
    return RopeChainMatch(
        final_node=final_view,
        nodes=nodes,
        cos=cos,
        sin=sin,
        head_dim=head_dim,
        rotary_dim=head_dim,
    )


def _try_fuse_gemm_rope(
    graph: torch.fx.Graph,
    addmm: torch.fx.Node,
    bias: torch.fx.Node,
    hidden_states: torch.fx.Node,
    permuted_weight: torch.fx.Node,
) -> bool:
    if not has_gemm_plugin(GEMM_ROPE_PLUGIN):
        return False

    if _is_call(permuted_weight, aten.t.default):
        weight = permuted_weight.args[0]
    elif _is_call(permuted_weight, aten.permute.default) and list(permuted_weight.args[1]) == [1, 0]:
        weight = permuted_weight.args[0]
    else:
        weight = permuted_weight

    hidden_shape = _node_size(hidden_states)
    weight_shape = _node_size(weight)
    bias_shape = _node_size(bias)
    addmm_shape = _node_size(addmm)
    if None in (hidden_shape, weight_shape, bias_shape, addmm_shape):
        return False
    if len(hidden_shape) != 2 or len(weight_shape) != 2 or len(bias_shape) != 1:
        return False
    if hidden_shape[1] != weight_shape[1]:
        return False

    split = _single_user(addmm, aten.split_with_sizes.default)
    if split is None:
        split = _single_user(addmm, "split")
    if split is None:
        return False
    split_info = _split_sizes_and_dim(split)
    if split_info is None:
        return False
    split_sizes, split_dim = split_info
    if split_dim != -1 or len(split_sizes) != 3:
        return False
    q_size, kv_size, kv_size_2 = split_sizes
    if kv_size != kv_size_2:
        return False

    getitems = {}
    for user in split.users:
        if _is_call(user, operator.getitem):
            getitems[user.args[1]] = user
    if set(getitems) != {0, 1, 2}:
        return False

    q_match = _match_rope_chain(getitems[0])
    k_match = _match_rope_chain(getitems[1])
    if q_match is None or k_match is None:
        return False
    if q_match.head_dim != k_match.head_dim:
        return False

    q_cos_match = _match_cos_sin(q_match.cos)
    q_sin_match = _match_cos_sin(q_match.sin)
    k_cos_match = _match_cos_sin(k_match.cos)
    k_sin_match = _match_cos_sin(k_match.sin)
    if None in (q_cos_match, q_sin_match, k_cos_match, k_sin_match):
        return False
    if not _same_cos_sin_match(q_cos_match, k_cos_match):
        return False
    if not _same_cos_sin_match(q_sin_match, k_sin_match):
        return False

    index_node = q_cos_match[0]
    if index_node is not q_sin_match[0]:
        return False

    cos_sin_cache, positions_list = index_node.args
    if not isinstance(positions_list, list) or len(positions_list) != 1:
        if index_node.target is not operator.getitem:
            return False
        positions = positions_list
    else:
        positions = positions_list[0]

    weight_meta = _node_meta_tensor(weight)
    if weight_meta is None or not getattr(weight_meta, "is_cuda", False):
        return False

    nodes_to_remove = [
        q_match.final_node,
        k_match.final_node,
        *q_match.nodes[:-1],
        *k_match.nodes[:-1],
        getitems[0],
        getitems[1],
        getitems[2],
        split,
        addmm,
    ]
    if weight is not permuted_weight:
        nodes_to_remove.append(permuted_weight)
    seen = set()
    deduped_nodes = []
    for node in nodes_to_remove:
        if node not in seen:
            deduped_nodes.append(node)
            seen.add(node)

    counters["inductor"]["gemm_rope"] += 1
    with graph.inserting_before(addmm):
        function = make_gemm_plugin_lowering(
            GEMM_ROPE_PLUGIN,
            q_size,
            kv_size,
            q_match.head_dim,
            q_match.rotary_dim,
        )
        replacement = graph.call_function(
            function,
            (hidden_states, weight, bias, cos_sin_cache, positions),
        )
        replacement.meta.update(addmm.meta)
        tuple_meta = (
            _node_meta_tensor(q_match.final_node),
            _node_meta_tensor(k_match.final_node),
            _node_meta_tensor(getitems[2]),
        )
        if all(x is not None for x in tuple_meta):
            if "val" in replacement.meta:
                replacement.meta["val"] = tuple_meta
            if "example_value" in replacement.meta:
                replacement.meta["example_value"] = tuple_meta
        q_new = graph.call_function(operator.getitem, (replacement, 0))
        k_new = graph.call_function(operator.getitem, (replacement, 1))
        v_new = graph.call_function(operator.getitem, (replacement, 2))
        q_new.meta.update(q_match.final_node.meta)
        k_new.meta.update(k_match.final_node.meta)
        v_new.meta.update(getitems[2].meta)
        q_match.final_node.replace_all_uses_with(q_new)
        k_match.final_node.replace_all_uses_with(k_new)
        getitems[2].replace_all_uses_with(v_new)

    erase_nodes = [node for node in graph.nodes if node in seen and not node.users]
    for node in reversed(erase_nodes):
        graph.erase_node(node)
    graph.lint()
    return True


def gemm_rope_handler(
    match: Match,
    bias: torch.fx.Node,
    hidden_states: torch.fx.Node,
    permuted_weight: torch.fx.Node,
) -> None:
    _try_fuse_gemm_rope(
        match.graph,
        match.output_node(),
        bias,
        hidden_states,
        permuted_weight,
    )


def apply_gemm_rope_pass(graph: torch.fx.Graph) -> None:
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is aten.addmm.default:
            bias, hidden_states, permuted_weight = node.args
        elif node.target is linear:
            hidden_states, permuted_weight, bias = node.args
        else:
            continue
        if all(isinstance(arg, torch.fx.Node) for arg in (bias, hidden_states, permuted_weight)):
            if _try_fuse_gemm_rope(graph, node, bias, hidden_states, permuted_weight):
                break
