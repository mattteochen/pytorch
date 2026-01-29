# Owner(s): ["oncall: distributed"]
"""
FX pass to fuse all_reduce + residual add + RMSNorm into a single kernel.

This pass detects patterns like:
    reduced = all_reduce(x)
    waited = wait_tensor(reduced)
    added = waited + residual  # optional residual add
    normed = rms_norm(added, weight, eps)

And replaces them with:
    normed = torch.ops.symm_mem.fused_all_reduce_rmsnorm(x, residual, weight, reduce_op, group_name, eps)

The fused kernel performs all operations in a single memory pass, avoiding
intermediate memory round-trips.
"""

import logging
import operator
from dataclasses import dataclass
from typing import Optional

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet

from .. import config


log = logging.getLogger(__name__)
aten = torch.ops.aten
c10d = torch.ops._c10d_functional


@dataclass
class _AllReduceRMSNormMatch:
    """Represents a matched all_reduce -> [add] -> rmsnorm pattern."""

    # Core nodes
    all_reduce_node: fx.Node
    wait_node: fx.Node
    rmsnorm_node: fx.Node

    # Input nodes
    input_node: fx.Node  # Input to all_reduce
    weight_node: fx.Node  # RMSNorm weight

    # Optional residual
    add_node: Optional[fx.Node]
    residual_node: Optional[fx.Node]

    # Parameters
    reduce_op: str
    group_name: str
    normalized_shape: tuple
    eps: float


def _is_all_reduce(node: fx.Node) -> bool:
    """Check if node is an all_reduce collective."""
    return node.target in (
        c10d.all_reduce.default,
        c10d.all_reduce_.default,
    )


def _is_wait_tensor(node: fx.Node) -> bool:
    """Check if node is a wait_tensor."""
    return node.target == c10d.wait_tensor.default


def _is_rmsnorm_mul_weight(node: fx.Node) -> bool:
    """
    Check if node is the final mul in decomposed RMSNorm pattern.
    
    Decomposed RMSNorm pattern:
        pow(x, 2) → mean → add(eps) → rsqrt → mul(x, rsqrt) → mul(weight)
    
    The final mul is: mul(normalized_x, weight)
    """
    if node.target != aten.mul.Tensor:
        return False
    
    # Check if one of the args is from the rsqrt branch (mul with rsqrt result)
    # and the other is a parameter (weight)
    args = node.args
    if len(args) != 2:
        return False
    
    # One arg should be from another mul (the x * rsqrt mul)
    # The other should be weight (a placeholder or parameter)
    arg0, arg1 = args
    
    if not isinstance(arg0, fx.Node) or not isinstance(arg1, fx.Node):
        return False
    
    # Check if arg0 is mul(x, rsqrt) and arg1 is weight
    def is_x_times_rsqrt(n):
        if n.target != aten.mul.Tensor:
            return False
        mul_args = n.args
        if len(mul_args) != 2:
            return False
        # One arg should be rsqrt
        for a in mul_args:
            if isinstance(a, fx.Node) and a.target == aten.rsqrt.default:
                return True
        return False
    
    def is_weight_like(n):
        # Weight is typically a placeholder or comes from a getattr
        return n.op in ("placeholder", "get_attr")
    
    if is_x_times_rsqrt(arg0) and is_weight_like(arg1):
        return True
    if is_x_times_rsqrt(arg1) and is_weight_like(arg0):
        return True
    
    return False


def _is_add(node: fx.Node) -> bool:
    """Check if node is an add operation."""
    return node.target in (
        aten.add.Tensor,
        aten.add_.Tensor,
        operator.add,
    )


def _get_reduce_op(node: fx.Node) -> str:
    """Extract reduce_op from all_reduce node."""
    # all_reduce(input, reduce_op, group_name)
    if len(node.args) >= 2:
        return node.args[1]
    return node.kwargs.get("reduce_op", "sum")


def _get_group_name(node: fx.Node) -> str:
    """Extract group_name from all_reduce node."""
    if len(node.args) >= 3:
        return node.args[2]
    return node.kwargs.get("group_name", "")


def _trace_back_to_wait_tensor(node: fx.Node, visited: set) -> Optional[fx.Node]:
    """
    Trace back through the decomposed RMSNorm pattern to find wait_tensor.
    
    Decomposed pattern: wait_tensor → pow → mean → add → rsqrt → mul → mul(weight)
    We need to find the wait_tensor that feeds into this.
    """
    if node in visited:
        return None
    visited.add(node)
    
    if _is_wait_tensor(node):
        return node
    
    # Trace through the args
    for arg in node.args:
        if isinstance(arg, fx.Node):
            result = _trace_back_to_wait_tensor(arg, visited)
            if result is not None:
                return result
    
    return None


def _find_all_reduce_rmsnorm_patterns(graph: fx.Graph) -> list[_AllReduceRMSNormMatch]:
    """
    Find all_reduce -> [wait] -> [add] -> rmsnorm patterns in the graph.

    Matches these patterns:
    1. all_reduce -> wait -> rmsnorm (no residual)
    2. all_reduce -> wait -> add -> rmsnorm (with residual)
    
    RMSNorm is always decomposed by Inductor into:
        wait_tensor → pow → mean → add(eps) → rsqrt → mul(x, rsqrt) → mul(weight)
    
    We match the final mul(normalized, weight) and trace back to find the all_reduce.
    """
    matches = []

    for node in graph.nodes:
        # Match decomposed rmsnorm pattern
        # Look for the final mul(normalized, weight) pattern
        if _is_rmsnorm_mul_weight(node):
            match = _match_decomposed_rmsnorm(node)
            if match is not None:
                matches.append(match)

    return matches


def _find_residual_add_pattern(
    wait_node: fx.Node, visited: set
) -> tuple[Optional[fx.Node], Optional[fx.Node]]:
    """
    Find if there's a residual add between wait_tensor and the rmsnorm computation.
    
    Pattern with residual:
        wait_tensor → add(wait_tensor, residual) → pow → ...
    
    Returns (add_node, residual_node) or (None, None) if no residual.
    """
    # Look for add nodes in visited that have wait_node as an input
    for node in visited:
        if not _is_add(node):
            continue
        
        args = node.args
        if len(args) < 2:
            continue
        
        arg0, arg1 = args[0], args[1]
        
        # Check if one of the args is wait_node (directly or indirectly)
        if arg0 is wait_node:
            if isinstance(arg1, fx.Node) and arg1.op in ("placeholder", "get_attr"):
                return node, arg1
            elif isinstance(arg1, fx.Node):
                # arg1 might be the residual even if not a placeholder
                return node, arg1
        elif arg1 is wait_node:
            if isinstance(arg0, fx.Node) and arg0.op in ("placeholder", "get_attr"):
                return node, arg0
            elif isinstance(arg0, fx.Node):
                return node, arg0
    
    return None, None


def _match_decomposed_rmsnorm(final_mul_node: fx.Node) -> Optional[_AllReduceRMSNormMatch]:
    """
    Match decomposed RMSNorm pattern.
    
    Pattern without residual:
        wait_tensor → pow(x,2) → mean → add(eps) → rsqrt → mul(x,rsqrt) → mul(weight)
    
    Pattern with residual:
        wait_tensor → add(wait, residual) → pow → mean → add(eps) → rsqrt → mul(add,rsqrt) → mul(weight)
    """
    # Find weight node (one of the args to final mul)
    arg0, arg1 = final_mul_node.args[0], final_mul_node.args[1]
    
    if not isinstance(arg0, fx.Node) or not isinstance(arg1, fx.Node):
        return None
    
    # Determine which is the x*rsqrt mul and which is weight
    if arg0.op in ("placeholder", "get_attr"):
        weight_node = arg0
        x_rsqrt_mul = arg1
    elif arg1.op in ("placeholder", "get_attr"):
        weight_node = arg1
        x_rsqrt_mul = arg0
    else:
        return None
    
    # x_rsqrt_mul should be mul(x, rsqrt)
    if x_rsqrt_mul.target != aten.mul.Tensor:
        return None
    
    # Trace back to find wait_tensor
    visited: set = set()
    wait_node = _trace_back_to_wait_tensor(x_rsqrt_mul, visited)
    
    if wait_node is None:
        return None
    
    # Get all_reduce from wait_tensor
    all_reduce_input = wait_node.args[0]
    if not isinstance(all_reduce_input, fx.Node) or not _is_all_reduce(all_reduce_input):
        return None
    
    all_reduce_node = all_reduce_input
    input_node = all_reduce_node.args[0]
    
    if not isinstance(input_node, fx.Node):
        return None
    
    # Try to extract eps from the add node in the rsqrt chain
    eps = 1e-6  # default
    for n in visited:
        if n.target == aten.add_.Scalar or n.target == aten.add.Scalar:
            if len(n.args) >= 2 and isinstance(n.args[1], (int, float)):
                eps = n.args[1]
                break
    
    # Check for residual add pattern
    # In decomposed form: wait_tensor → add(wait, residual) → pow → ...
    add_node, residual_node = _find_residual_add_pattern(wait_node, visited)
    
    return _AllReduceRMSNormMatch(
        all_reduce_node=all_reduce_node,
        wait_node=wait_node,
        rmsnorm_node=final_mul_node,  # Use final mul as the "rmsnorm" node
        input_node=input_node,
        weight_node=weight_node,
        add_node=add_node,
        residual_node=residual_node,
        reduce_op=_get_reduce_op(all_reduce_node),
        group_name=_get_group_name(all_reduce_node),
        normalized_shape=None,  # Not easily extractable from decomposed form
        eps=eps,
    )


def _replace_with_fused_op(
    graph: fx.Graph,
    match: _AllReduceRMSNormMatch,
) -> fx.Node:
    """Replace matched pattern with fused operation."""
    with graph.inserting_before(match.rmsnorm_node):
        # Create the fused op call
        # For now, we'll create a placeholder that calls our custom op
        # The actual op will be: torch.ops.symm_mem.fused_all_reduce_rmsnorm
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

        # Copy metadata from rmsnorm node
        fused_node.meta.update(match.rmsnorm_node.meta)

        return fused_node


def _can_fuse(match: _AllReduceRMSNormMatch) -> bool:
    """
    Check if the matched pattern can be safely fused.

    Conditions:
    1. The wait_tensor result is only used by the add/rmsnorm (no other users)
    2. If there's an add, it's only used by rmsnorm
    3. Symmetric memory is enabled for the group
    """
    # Check wait_tensor users
    wait_users = OrderedSet(match.wait_node.users.keys())
    expected_wait_users = OrderedSet([match.add_node or match.rmsnorm_node])
    if wait_users != expected_wait_users:
        log.debug(
            "Cannot fuse: wait_tensor has unexpected users. "
            f"Expected {expected_wait_users}, got {wait_users}"
        )
        return False

    # Check add users (if present)
    if match.add_node is not None:
        add_users = OrderedSet(match.add_node.users.keys())
        if add_users != OrderedSet([match.rmsnorm_node]):
            log.debug(
                "Cannot fuse: add has unexpected users. "
                f"Expected {{rmsnorm}}, got {add_users}"
            )
            return False

    # Check if symmetric memory is enabled for the group
    try:
        from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

        if not is_symm_mem_enabled_for_group(match.group_name):
            log.debug(
                f"Cannot fuse: symmetric memory not enabled for group {match.group_name}"
            )
            return False
    except ImportError:
        log.debug("Cannot fuse: symmetric memory module not available")
        return False

    return True


def _erase_old_nodes(match: _AllReduceRMSNormMatch) -> None:
    """Erase the old nodes that were replaced by the fused op."""
    graph = match.rmsnorm_node.graph

    # Erase in reverse dependency order
    nodes_to_erase = [match.rmsnorm_node]
    if match.add_node is not None:
        nodes_to_erase.append(match.add_node)
    nodes_to_erase.extend([match.wait_node, match.all_reduce_node])

    for node in nodes_to_erase:
        if len(node.users) == 0:
            graph.erase_node(node)


def fused_all_reduce_rmsnorm_pass(graph: fx.Graph) -> None:
    """
    Main pass entry point.

    Finds all_reduce + rmsnorm patterns and replaces them with fused ops.
    """
    matches = _find_all_reduce_rmsnorm_patterns(graph)

    if not matches:
        log.debug("fused_all_reduce_rmsnorm_pass: no patterns found")
        return

    log.debug(f"fused_all_reduce_rmsnorm_pass: found {len(matches)} patterns")

    fused_count = 0
    for match in matches:
        if not _can_fuse(match):
            continue

        # Replace with fused op
        fused_node = _replace_with_fused_op(graph, match)

        # Update users of rmsnorm to use fused node
        match.rmsnorm_node.replace_all_uses_with(fused_node)

        # Erase old nodes
        _erase_old_nodes(match)

        fused_count += 1

    if fused_count > 0:
        log.debug(f"fused_all_reduce_rmsnorm_pass: fused {fused_count} patterns")
        graph.lint()
