# Owner(s): ["oncall: distributed"]
"""
FX pass to fuse all_reduce + residual add + RMSNorm into a single kernel.

This pass detects patterns like:
    reduced = all_reduce(x)
    waited = wait_tensor(reduced)
    added = waited + residual  # optional residual add
    normed = rms_norm(added, weight, eps)

And replaces them with:
    normed, pre_norm = torch.ops.symm_mem.fused_all_reduce_rmsnorm(
        x, weight, reduce_op, group_name, residual=residual, eps=eps
    )

The fused op returns (normed_output, pre_norm_output) where pre_norm_output
is the reduced + residual value (or None when no residual is provided).
"""

import logging
import operator
from dataclasses import dataclass, field

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
c10d = torch.ops._c10d_functional


@dataclass
class _AllReduceRMSNormMatch:
    """Represents a matched all_reduce -> [add] -> rmsnorm pattern."""

    all_reduce_node: fx.Node
    wait_node: fx.Node
    rmsnorm_node: fx.Node  # final mul(normalized, weight)

    input_node: fx.Node
    weight_node: fx.Node

    add_node: fx.Node | None
    residual_node: fx.Node | None

    reduce_op: str
    group_name: str
    eps: float

    # All intermediate nodes in the decomposed RMSNorm chain (excluding
    # wait_node / add_node / all_reduce_node which are tracked separately).
    intermediate_nodes: set[fx.Node] = field(default_factory=set)


def _is_all_reduce(node: fx.Node) -> bool:
    return node.target in (
        c10d.all_reduce.default,
        c10d.all_reduce_.default,
    )


def _is_wait_tensor(node: fx.Node) -> bool:
    return node.target == c10d.wait_tensor.default


def _is_rmsnorm_mul_weight(node: fx.Node) -> bool:
    """
    Check if *node* is the final ``mul(x * rsqrt, weight)`` of a decomposed
    RMSNorm.
    """
    if node.target != aten.mul.Tensor:
        return False
    args = node.args
    if len(args) != 2:
        return False
    arg0, arg1 = args
    if not isinstance(arg0, fx.Node) or not isinstance(arg1, fx.Node):
        return False

    def is_x_times_rsqrt(n: fx.Node) -> bool:
        if n.target != aten.mul.Tensor or len(n.args) != 2:
            return False
        return any(
            isinstance(a, fx.Node) and a.target == aten.rsqrt.default
            for a in n.args
        )

    def is_weight_like(n: fx.Node) -> bool:
        return n.op in ("placeholder", "get_attr")

    return (is_x_times_rsqrt(arg0) and is_weight_like(arg1)) or (
        is_x_times_rsqrt(arg1) and is_weight_like(arg0)
    )


def _is_add(node: fx.Node) -> bool:
    return node.target in (
        aten.add.Tensor,
        aten.add_.Tensor,
        operator.add,
    )


def _get_reduce_op(node: fx.Node) -> str:
    if len(node.args) >= 2:
        return node.args[1]
    return node.kwargs.get("reduce_op", "sum")


def _get_group_name(node: fx.Node) -> str:
    if len(node.args) >= 3:
        return node.args[2]
    return node.kwargs.get("group_name", "")


def _trace_back_to_wait_tensor(
    node: fx.Node, visited: set[fx.Node]
) -> fx.Node | None:
    """Walk backward through the decomposed RMSNorm to find wait_tensor."""
    if node in visited:
        return None
    visited.add(node)
    if _is_wait_tensor(node):
        return node
    found = None
    for arg in node.args:
        if isinstance(arg, fx.Node):
            result = _trace_back_to_wait_tensor(arg, visited)
            if result is not None:
                found = result
    return found


def _find_residual_add_pattern(
    wait_node: fx.Node, visited: set[fx.Node]
) -> tuple[fx.Node | None, fx.Node | None]:
    """
    Find a residual ``add(wait_tensor, residual)`` inside the visited set.
    Returns ``(add_node, residual_node)`` or ``(None, None)``.
    """
    for node in visited:
        if not _is_add(node) or len(node.args) < 2:
            continue
        arg0, arg1 = node.args[0], node.args[1]
        if arg0 is wait_node and isinstance(arg1, fx.Node):
            return node, arg1
        if arg1 is wait_node and isinstance(arg0, fx.Node):
            return node, arg0
    return None, None


def _match_decomposed_rmsnorm(
    final_mul_node: fx.Node,
) -> _AllReduceRMSNormMatch | None:
    """
    Starting from the final ``mul(normalised, weight)`` of a decomposed
    RMSNorm, trace backward to find an ``all_reduce -> wait_tensor`` feeding
    into it.
    """
    arg0, arg1 = final_mul_node.args[0], final_mul_node.args[1]
    if not isinstance(arg0, fx.Node) or not isinstance(arg1, fx.Node):
        return None

    if arg0.op in ("placeholder", "get_attr"):
        weight_node, x_rsqrt_mul = arg0, arg1
    elif arg1.op in ("placeholder", "get_attr"):
        weight_node, x_rsqrt_mul = arg1, arg0
    else:
        return None

    if x_rsqrt_mul.target != aten.mul.Tensor:
        return None

    visited: set[fx.Node] = set()
    wait_node = _trace_back_to_wait_tensor(x_rsqrt_mul, visited)
    if wait_node is None:
        return None

    all_reduce_input = wait_node.args[0]
    if not isinstance(all_reduce_input, fx.Node) or not _is_all_reduce(
        all_reduce_input
    ):
        return None
    all_reduce_node = all_reduce_input
    input_node = all_reduce_node.args[0]
    if not isinstance(input_node, fx.Node):
        return None

    eps = 1e-6
    for n in visited:
        if n.target in (aten.add_.Scalar, aten.add.Scalar):
            if len(n.args) >= 2 and isinstance(n.args[1], (int, float)):
                eps = n.args[1]
                break

    add_node, residual_node = _find_residual_add_pattern(wait_node, visited)

    # intermediate_nodes = everything the backward trace touched, minus the
    # nodes we track separately.
    intermediate_nodes = visited - {wait_node, add_node}
    intermediate_nodes.discard(None)  # type: ignore[arg-type]

    return _AllReduceRMSNormMatch(
        all_reduce_node=all_reduce_node,
        wait_node=wait_node,
        rmsnorm_node=final_mul_node,
        input_node=input_node,
        weight_node=weight_node,
        add_node=add_node,
        residual_node=residual_node,
        reduce_op=_get_reduce_op(all_reduce_node),
        group_name=_get_group_name(all_reduce_node),
        eps=eps,
        intermediate_nodes=intermediate_nodes,
    )


def _find_all_reduce_rmsnorm_patterns(
    graph: fx.Graph,
) -> list[_AllReduceRMSNormMatch]:
    matches = []
    for node in graph.nodes:
        if _is_rmsnorm_mul_weight(node):
            match = _match_decomposed_rmsnorm(node)
            if match is not None:
                matches.append(match)
    return matches


def _can_fuse(match: _AllReduceRMSNormMatch) -> bool:
    """
    Check whether the matched pattern can safely be replaced.

    Constraints:
    - wait_tensor users must all be within the matched pattern (intermediate
      nodes or the add node).
    - Symmetric memory must be enabled for the group.

    The add node is allowed to have users outside the pattern because the
    fused op produces the pre-norm value as a second output.
    """
    pattern_nodes = match.intermediate_nodes | {match.rmsnorm_node}
    if match.add_node is not None:
        pattern_nodes.add(match.add_node)

    wait_users = OrderedSet(match.wait_node.users.keys())
    if not wait_users.issubset(pattern_nodes):
        log.debug(
            "Cannot fuse: wait_tensor has users outside the matched pattern: %s",
            wait_users - pattern_nodes,
        )
        return False

    try:
        from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

        if not is_symm_mem_enabled_for_group(match.group_name):
            log.debug(
                "Cannot fuse: symmetric memory not enabled for group %s",
                match.group_name,
            )
            return False
    except ImportError:
        log.debug("Cannot fuse: symmetric memory module not available")
        return False

    return True


def _replace_with_fused_op(
    graph: fx.Graph,
    match: _AllReduceRMSNormMatch,
) -> tuple[fx.Node, fx.Node]:
    """
    Replace the matched pattern with the fused op and return
    ``(normed_getitem, pre_norm_getitem)`` nodes.
    """
    with graph.inserting_before(match.rmsnorm_node):
        fused_node = graph.call_function(
            torch.ops.symm_mem.fused_all_reduce_rmsnorm.default,
            args=(
                match.input_node,
                match.weight_node,
                match.reduce_op,
                match.group_name,
            ),
            kwargs={
                "residual": match.residual_node,
                "eps": match.eps,
            },
        )
        fused_node.meta.update(match.rmsnorm_node.meta)

        normed_node = graph.call_function(operator.getitem, args=(fused_node, 0))
        normed_node.meta.update(match.rmsnorm_node.meta)

        pre_norm_node = graph.call_function(operator.getitem, args=(fused_node, 1))
        if match.add_node is not None:
            pre_norm_node.meta.update(match.add_node.meta)
        else:
            pre_norm_node.meta.update(match.wait_node.meta)

    return normed_node, pre_norm_node


def _erase_old_nodes(match: _AllReduceRMSNormMatch) -> None:
    """Erase all nodes belonging to the old pattern, in safe order."""
    graph = match.rmsnorm_node.graph

    candidates = {match.rmsnorm_node, match.wait_node, match.all_reduce_node}
    candidates |= match.intermediate_nodes
    if match.add_node is not None:
        candidates.add(match.add_node)

    # Repeatedly erase nodes with no remaining users until stable.
    changed = True
    while changed:
        changed = False
        for node in list(candidates):
            if node in graph.nodes and len(node.users) == 0:
                graph.erase_node(node)
                candidates.discard(node)
                changed = True


def _absorb_trailing_converts(
    normed_node: fx.Node, input_node: fx.Node
) -> None:
    """
    The fused op outputs in the input's dtype (e.g. bf16), but the old
    ``mul(normalised, fp32_weight)`` produced fp32.  The decomposed RMSNorm
    therefore had a trailing ``convert_element_type(result, bf16)`` that is
    now redundant.  Absorb it so Inductor doesn't emit a pointwise kernel.
    """
    input_meta_val = input_node.meta.get("val")
    if input_meta_val is None:
        return
    input_dtype = input_meta_val.dtype

    for user in list(normed_node.users.keys()):
        if (
            user.target == prims.convert_element_type.default
            and len(user.args) >= 2
            and user.args[1] == input_dtype
        ):
            user.replace_all_uses_with(normed_node)
            user.graph.erase_node(user)

    # Fix metadata: the fused op's Meta impl returns empty_like(input),
    # so the actual dtype is the input dtype, not the old fp32 mul dtype.
    old_val = normed_node.meta.get("val")
    if old_val is not None and hasattr(old_val, "dtype") and old_val.dtype != input_dtype:
        normed_node.meta["val"] = old_val.to(input_dtype)


def fused_all_reduce_rmsnorm_pass(graph: fx.Graph, is_inference: bool = True) -> None:
    """
    Main pass entry point.

    Finds ``all_reduce + rmsnorm`` patterns and replaces them with fused ops.
    Only runs during inference because the fused op does not produce the
    intermediate tensors (e.g. rsqrt) that autograd saves for backward.
    """
    if not is_inference:
        log.debug("fused_all_reduce_rmsnorm_pass: skipped (training mode)")
        return
    matches = _find_all_reduce_rmsnorm_patterns(graph)
    if not matches:
        log.debug("fused_all_reduce_rmsnorm_pass: no patterns found")
        return

    log.debug("fused_all_reduce_rmsnorm_pass: found %d patterns", len(matches))

    fused_count = 0
    for match in matches:
        if not _can_fuse(match):
            continue

        normed_node, pre_norm_node = _replace_with_fused_op(graph, match)

        # Rewire: old rmsnorm users → normed_node (getitem 0)
        match.rmsnorm_node.replace_all_uses_with(normed_node)

        # Rewire: old add users outside the pattern → pre_norm_node (getitem 1)
        if match.add_node is not None:
            match.add_node.replace_all_uses_with(pre_norm_node)

        _erase_old_nodes(match)
        _absorb_trailing_converts(normed_node, match.input_node)
        fused_count += 1

    if fused_count > 0:
        log.debug("fused_all_reduce_rmsnorm_pass: fused %d patterns", fused_count)
        graph.lint()
