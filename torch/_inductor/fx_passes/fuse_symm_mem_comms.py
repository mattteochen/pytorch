# Owner(s): ["oncall: distributed"]
"""
FX pass to replace symmetric-memory-eligible collectives with P2P
implementations.

**Currently supported:** ``all_reduce + wait_tensor`` → ``p2p_allreduce``.
Other collectives (all_gather, reduce_scatter, etc.) are not yet handled.

When ``torch._inductor.config._fuse_symm_mem_comms`` is enabled this
pass detects::

    reduced = all_reduce(x)
    waited = wait_tensor(reduced)

and replaces it with::

    reduced = torch.ops.symm_mem.p2p_allreduce(x, reduce_op, group_name)

The downstream RMSNorm (or any other compute) is left **untouched** so that
inductor's scheduler fuses it naturally into the same Triton kernel.  The
``p2p_allreduce`` op lowers to a ``SymmMemP2PAllReduce`` IR Pointwise whose
``inner_fn`` emits ``ops.symm_mem_p2p_reduce_load`` -- translated by the
Triton codegen into a loop of P2P loads from peer symmetric-memory buffers
with kraken device-side synchronisation (``ptx_utils.symm_mem_sync``).
"""

import logging

import torch
import torch.fx as fx


log = logging.getLogger(__name__)


def _c10d():
    return torch.ops._c10d_functional


def _is_all_reduce(node: fx.Node) -> bool:
    c10d = _c10d()
    return node.target in (
        c10d.all_reduce.default,
        c10d.all_reduce_.default,
    )


def _is_wait_tensor(node: fx.Node) -> bool:
    return node.target == _c10d().wait_tensor.default


def _get_reduce_op(node: fx.Node) -> str:
    if len(node.args) >= 2:
        return node.args[1]
    return node.kwargs.get("reduce_op", "sum")


def _get_group_name(node: fx.Node) -> str:
    if len(node.args) >= 3:
        return node.args[2]
    group_name = node.kwargs.get("group_name", "")
    if not group_name:
        log.warning(
            "all_reduce node %s has no group_name; this likely indicates a "
            "malformed graph. Defaulting to empty string.",
            node.name,
        )
    return group_name


def _find_all_reduce_wait_patterns(
    graph: fx.Graph,
) -> list[tuple[fx.Node, fx.Node]]:
    """Return ``(all_reduce_node, wait_node)`` pairs."""
    patterns: list[tuple[fx.Node, fx.Node]] = []
    for node in graph.nodes:
        if not _is_wait_tensor(node):
            continue
        ar = node.args[0]
        if isinstance(ar, fx.Node) and _is_all_reduce(ar):
            patterns.append((ar, node))
    return patterns


def _can_replace(all_reduce_node: fx.Node, wait_node: fx.Node) -> bool:
    # Guard on supported dtypes — the codegen only supports float types.
    # Lamport mode further requires 2-byte types (bf16/fp16) for the
    # sentinel protocol; that is checked at codegen time.
    val = wait_node.meta.get("val")
    if val is not None and not val.dtype.is_floating_point:
        log.debug("Cannot replace: non-floating-point dtype %s", val.dtype)
        return False

    reduce_op = _get_reduce_op(all_reduce_node)
    if reduce_op != "sum":
        log.debug("Cannot replace: unsupported reduce_op '%s' (only 'sum')", reduce_op)
        return False

    max_bytes = torch._inductor.config._fuse_symm_mem_comms_max_bytes
    if max_bytes > 0 and val is not None:
        nbytes = val.numel() * val.element_size()
        if nbytes > max_bytes:
            log.debug(
                "Cannot replace: tensor too large (%d bytes > %d threshold)",
                nbytes,
                max_bytes,
            )
            return False

    return True


def fuse_symm_mem_comms_pass(graph: fx.Graph, is_inference: bool = True) -> None:
    """
    Main pass entry point.

    Replaces ``all_reduce -> wait_tensor`` with ``symm_mem.p2p_allreduce``.
    Only runs during inference (training needs intermediates for autograd).
    """
    if not is_inference:
        log.debug("fuse_symm_mem_comms_pass: skipped (training mode)")
        return

    patterns = _find_all_reduce_wait_patterns(graph)
    if not patterns:
        log.debug("fuse_symm_mem_comms_pass: no all_reduce->wait patterns found")
        return

    log.debug(
        "fuse_symm_mem_comms_pass: found %d all_reduce->wait patterns",
        len(patterns),
    )

    replaced = 0
    for all_reduce_node, wait_node in patterns:
        if not _can_replace(all_reduce_node, wait_node):
            continue

        input_node = all_reduce_node.args[0]
        if not isinstance(input_node, fx.Node):
            continue

        reduce_op = _get_reduce_op(all_reduce_node)
        group_name = _get_group_name(all_reduce_node)

        with graph.inserting_before(wait_node):
            p2p_node = graph.call_function(
                torch.ops.symm_mem.p2p_allreduce.default,
                args=(input_node, reduce_op, group_name),
            )
            p2p_node.meta.update(wait_node.meta)

        wait_node.replace_all_uses_with(p2p_node)

        # Erase wait_node, then all_reduce_node (which now has no users).
        graph.erase_node(wait_node)
        if len(all_reduce_node.users) == 0:
            graph.erase_node(all_reduce_node)

        replaced += 1

    if replaced > 0:
        log.debug("fuse_symm_mem_comms_pass: replaced %d patterns", replaced)
        graph.lint()
