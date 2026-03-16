"""
Symmetric memory P2P allreduce codegen for TritonKernel.

Four sync modes: host_barrier (default), device_cas, device_cas_2_shot, lamport.
All functions take an explicit kernel argument instead of using self.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import sympy

import torch

from .. import config
from ..virtualized import V

if TYPE_CHECKING:
    from .triton import TritonCSEVariable, TritonKernel


_TWO_SHOT_COPY_BEGIN = "# --- symm mem two-shot copy begin ---"
_TWO_SHOT_COPY_END = "# --- symm mem two-shot copy end ---"
_TWO_SHOT_REDUCE_BEGIN = "# --- symm mem two-shot reduce begin ---"
_TWO_SHOT_REDUCE_END = "# --- symm mem two-shot reduce end ---"
_TWO_SHOT_TAIL_BEGIN = "# --- symm mem two-shot tail begin ---"


def _triton_type(dtype):
    from ..utils import triton_type

    return triton_type(dtype)


# ------------------------------------------------------------------
# State init (called from TritonKernel.__init__)
# ------------------------------------------------------------------


def init_symm_mem_state(kernel: TritonKernel):
    kernel.has_symm_mem_p2p = False
    kernel.symm_mem_world_size: int | None = None
    kernel.symm_mem_input_name: str | None = None
    kernel.symm_mem_input_dtype: torch.dtype | None = None
    kernel._symm_group_name: str = ""
    kernel._symm_mem_use_host_barriers: bool = False
    kernel._symm_mem_use_lamport: bool = False


# ------------------------------------------------------------------
# Main entry point (called from ops handler)
# ------------------------------------------------------------------


def symm_mem_p2p_reduce_load(
    kernel: TritonKernel,
    name: str,
    index: sympy.Expr,
    world_size: int,
    group_name: str = "",
) -> TritonCSEVariable:
    from .triton import IndexingOptions, TritonSymbols

    kernel.has_symm_mem_p2p = True
    kernel.symm_mem_world_size = world_size
    kernel.symm_mem_input_name = name
    kernel.symm_mem_input_dtype = V.graph.get_dtype(name)
    kernel._symm_group_name = group_name

    sync_mode = config._symm_mem_sync_mode
    if sync_mode == "lamport":
        kernel._symm_mem_use_lamport = True
        kernel._symm_mem_use_host_barriers = False
    elif sync_mode in ("device_cas", "device_cas_2_shot"):
        kernel._symm_mem_use_lamport = False
        kernel._symm_mem_use_host_barriers = False
    else:
        kernel._symm_mem_use_lamport = False
        threshold = config._symm_mem_host_barrier_threshold
        if threshold == -1:
            kernel._symm_mem_use_host_barriers = False
        elif threshold == 0:
            kernel._symm_mem_use_host_barriers = True
        else:
            xnumel = V.graph.sizevars.simplify(kernel.numels["x"])
            is_static = isinstance(xnumel, (sympy.Integer, int))
            kernel._symm_mem_use_host_barriers = (
                not is_static or int(xnumel) > threshold
            )

    kernel.args.input(name)
    kernel.must_keep_buffers.add(name)

    indexing = kernel.indexing(index, block_ptr=False)

    if not isinstance(indexing, IndexingOptions):
        raise NotImplementedError(
            "symm_mem_p2p_reduce_load only supports IndexingOptions"
        )

    cse_key = (
        f"symm_mem_p2p_reduce_load({name}, {indexing.index_str}, {world_size})"
    )
    cached = kernel.cse.try_get(cse_key)
    if cached is not None:
        cached.use_count += 1
        return cached

    load_buffer = kernel.get_load_buffer(indexing)

    shape = indexing.expand_shape or TritonSymbols.get_block_shape(indexing.index)
    shape_str = ", ".join(str(s) for s in shape) if shape else "1"

    if config._symm_mem_sync_mode == "lamport":
        _codegen_lamport_reduce_load(kernel, load_buffer, indexing, shape_str)
    elif config._symm_mem_sync_mode == "device_cas_2_shot":
        _codegen_two_shot_reduce_load(kernel, load_buffer, indexing, shape_str)
    else:
        _codegen_pull_reduce_load(kernel, load_buffer, indexing, shape_str)

    acc_var = kernel.cse.generate(
        load_buffer,
        "_symm_acc",
        dtype=torch.float32,
        shape=shape,
    )
    kernel.cse.put(cse_key, acc_var)

    if not kernel.inside_reduction or not indexing.has_rmask():
        kernel.outside_loop_vars.add(acc_var)

    return acc_var


# ------------------------------------------------------------------
# Body codegen: pull model (device_cas / host_barrier)
# ------------------------------------------------------------------


def _codegen_pull_reduce_load(kernel, load_buffer, indexing, shape_str: str):
    in_var = kernel.args.input(kernel.symm_mem_input_name)

    if (
        not config._symm_mem_skip_prologue_copy
        and not kernel._symm_mem_use_host_barriers
    ):
        load_buffer.splice(
            f"""
            for _symm_row in tl.static_range(XBLOCK):
                _symm_row_idx = _symm_x_base + _symm_row
                _symm_row_mask = _symm_row_idx < xnumel
                _symm_idx = _symm_row_idx * r0_numel + _symm_cols
                _symm_mask = _symm_col_mask & _symm_row_mask
                _symm_val = tl.load({in_var} + _symm_idx, _symm_mask, other=0.0)
                tl.store(_symm_local_buf + _symm_idx, _symm_val, _symm_mask)
            """
        )
        load_buffer.writeline(
            "_symm_mem_sync(symm_signal_pad_ptrs, None, SYMM_RANK, "
            "SYMM_WORLD_SIZE, hasPreviousMemAccess=True, "
            "hasSubsequentMemAccess=True)"
        )

    load_buffer.writeline(
        f"_symm_acc = tl.zeros([{shape_str}], dtype=tl.float32)"
    )
    for i in range(kernel.symm_mem_world_size):
        load_buffer.writeline(
            f"_symm_acc = _symm_acc + tl.load("
            f"symm_peer_buf_{i} + ({indexing.index_str}), "
            f"{indexing.mask_str}, other=0.0).to(tl.float32)"
        )


# ------------------------------------------------------------------
# Body codegen: two-shot reduce-scatter + allgather (device_cas_2_shot)
# ------------------------------------------------------------------


def _codegen_two_shot_reduce_load(kernel, load_buffer, indexing, shape_str: str):
    """Two-shot allreduce: reduce-scatter then allgather via symm_mem buffers.

    Each rank reduces its column chunk from all peers, writes the result back
    to every peer buffer, syncs, then loads the full reduced row from its own
    buffer.  Requires r0_numel % SYMM_WORLD_SIZE == 0.
    """
    in_var = kernel.args.input(kernel.symm_mem_input_name)
    use_grid_cap = config._symm_mem_grid_cap > 0

    if (
        not config._symm_mem_skip_prologue_copy
        and not kernel._symm_mem_use_host_barriers
    ):
        if use_grid_cap:
            load_buffer.writeline(_TWO_SHOT_COPY_BEGIN)
        load_buffer.splice(
            f"""
            for _symm_row in tl.static_range(XBLOCK):
                _symm_row_idx = _symm_x_base + _symm_row
                _symm_row_mask = _symm_row_idx < xnumel
                _symm_idx = _symm_row_idx * r0_numel + _symm_cols
                _symm_mask = _symm_col_mask & _symm_row_mask
                _symm_val = tl.load({in_var} + _symm_idx, _symm_mask, other=0.0)
                tl.store(_symm_local_buf + _symm_idx, _symm_val, _symm_mask)
            """
        )
        if use_grid_cap:
            load_buffer.writeline(_TWO_SHOT_COPY_END)
        load_buffer.writeline(
            "_symm_mem_sync(symm_signal_pad_ptrs, None, SYMM_RANK, "
            "SYMM_WORLD_SIZE, hasPreviousMemAccess=True, "
            "hasSubsequentMemAccess=True)"
        )

    # Reduce-scatter: each rank works on its column chunk using a
    # chunk-sized arange (R0_BLOCK // SYMM_WORLD_SIZE) rather than the
    # full R0_BLOCK, avoiding wasted vector lanes.
    if use_grid_cap:
        load_buffer.writeline(_TWO_SHOT_REDUCE_BEGIN)
    load_buffer.splice(
        """
        _2shot_chunk = r0_numel // SYMM_WORLD_SIZE
        _2shot_cols = tl.arange(0, R0_BLOCK // SYMM_WORLD_SIZE)
        _2shot_col_mask = _2shot_cols < _2shot_chunk
        _2shot_col_start = SYMM_RANK * _2shot_chunk
        """
    )

    load_buffer.writeline("for _2shot_row in tl.static_range(XBLOCK):")
    with load_buffer.indent():
        load_buffer.splice(
            """
            _2shot_row_idx = _symm_x_base + _2shot_row
            _2shot_row_mask = _2shot_row_idx < xnumel
            _2shot_idx = _2shot_row_idx * r0_numel + _2shot_col_start + _2shot_cols
            _2shot_mask = _2shot_col_mask & _2shot_row_mask
            _2shot_acc = tl.zeros([R0_BLOCK // SYMM_WORLD_SIZE], dtype=tl.float32)
            """
        )
        for i in range(kernel.symm_mem_world_size):
            load_buffer.writeline(
                f"_2shot_acc = _2shot_acc + tl.load("
                f"symm_peer_buf_{i} + _2shot_idx, "
                f"_2shot_mask, other=0.0).to(tl.float32)"
            )
        for i in range(kernel.symm_mem_world_size):
            load_buffer.writeline(
                f"tl.store(symm_peer_buf_{i} + _2shot_idx, "
                f"_2shot_acc, _2shot_mask)"
            )

    if use_grid_cap:
        load_buffer.writeline(_TWO_SHOT_REDUCE_END)
    load_buffer.writeline(
        "_symm_mem_sync(symm_signal_pad_ptrs, None, SYMM_RANK, "
        "SYMM_WORLD_SIZE, hasPreviousMemAccess=True, "
        "hasSubsequentMemAccess=True)"
    )

    if use_grid_cap:
        load_buffer.writeline(_TWO_SHOT_TAIL_BEGIN)
    load_buffer.writeline(
        f"_symm_acc = tl.load("
        f"_symm_local_buf + ({indexing.index_str}), "
        f"{indexing.mask_str}, other=0.0).to(tl.float32)"
    )


# ------------------------------------------------------------------
# Body codegen: Lamport push model
#
# The Lamport allreduce uses a push-based protocol with negative-zero
# sentinels (bf16 bit pattern 0x8000) instead of explicit barriers.
# Each rank writes its local data into every peer's symmetric memory
# buffer, and peers poll until the sentinel is overwritten with real
# data.  Triple-buffering (3 slots) allows overlap with CUDA graphs.
#
# Generated kernel structure:
#   1. Push  – load local input, strip -0.0, store into each peer's
#              buffer at [buf_offset + my_rank * chunk + row * N + col].
#   2. Clear – re-arm the oldest triple-buffer slot ((flag+2)%3) with
#              -0.0 sentinels so it is ready for reuse two calls later.
#   3. Fence + arrive – fence.sc.sys makes stores visible system-wide;
#              block_arrive counts blocks for the in-kernel flag advance.
#   4. Poll-load + accumulate – for each peer, load the data and check
#              every element for -0.0; retry until the entire vector is
#              clean.  The poll IS the data read (no separate poll step).
#              This matches FlashInfer's methodology and avoids relying
#              on NVLink cache-line delivery order.
#
# Reference: cutlass/examples/python/CuTeDSL/distributed/
#            all_reduce_one_shot_lamport.py
# The reference uses .SYS + .VOLATILE on every 128-bit load/store.
# FlashInfer (third_party/flashinfer/include/flashinfer/comm/
# trtllm_allreduce.cuh) also checks every element of each volatile
# load via has_neg_zero() — the poll is the data read.
# ------------------------------------------------------------------


def _codegen_lamport_reduce_load(kernel, load_buffer, indexing, shape_str: str):
    tl_dtype = _triton_type(kernel.symm_mem_input_dtype)
    in_var = kernel.args.input(kernel.symm_mem_input_name)

    load_buffer.writeline(
        "# ─── Lamport push: load local data, write to every peer's buffer ───"
    )
    load_buffer.writeline("for _lam_row in tl.static_range(XBLOCK):")
    with load_buffer.indent():
        load_buffer.splice(
            f"""
            _lam_row_idx = _lam_x_base + _lam_row
            _lam_row_mask = _lam_row_idx < xnumel
            _lam_row_offset = _lam_row_idx * r0_numel
            _lam_idx = _lam_row_offset + _lam_cols
            _lam_mask = _lam_col_mask & _lam_row_mask
            _lam_data = tl.load({in_var} + _lam_idx, _lam_mask, other=0.0)
            _lam_data = _lamport_remove_neg_zero(_lam_data)
            """
        )
        for i in range(kernel.symm_mem_world_size):
            load_buffer.writeline(f"if {i} != SYMM_RANK:")
            with load_buffer.indent():
                load_buffer.writeline(
                    f"tl.store(symm_peer_buf_{i} + _lam_buf_offset + "
                    f"SYMM_RANK * _lam_chunk + _lam_row_offset + _lam_cols, "
                    f"_lam_data, mask=_lam_mask)"
                )

    load_buffer.writeline(
        "# ─── Lamport clear: re-arm old triple-buffer slot with sentinels ───"
    )
    load_buffer.writeline("for _lam_row_c in tl.static_range(XBLOCK):")
    with load_buffer.indent():
        load_buffer.splice(
            """
            _lam_row_idx_c = _lam_x_base + _lam_row_c
            _lam_row_mask_c = _lam_row_idx_c < xnumel
            _lam_row_offset_c = _lam_row_idx_c * r0_numel
            _lam_mask_c = _lam_col_mask & _lam_row_mask_c
            _lamport_clear_old_slot(_lam_clear_base, _lam_row_offset_c, _lam_cols, _lam_mask_c, _lam_chunk, SYMM_RANK, SYMM_WORLD_SIZE, R0_BLOCK)
            """
        )

    load_buffer.splice(
        """
        # ─── Lamport fence + arrive: make push visible, signal readiness ───
        _lamport_fence_sys()
        _lamport_block_arrive(_lam_meta_i32)
        """
    )

    load_buffer.writeline(
        "# ─── Lamport poll-load + accumulate: load peer data, retry on sentinel ──"
    )
    load_buffer.writeline(
        f"_symm_acc = tl.load({in_var} + ({indexing.index_str}), "
        f"{indexing.mask_str}, other=0.0).to(tl.float32)"
    )
    for i in range(kernel.symm_mem_world_size):
        load_buffer.writeline(f"if {i} != SYMM_RANK:")
        with load_buffer.indent():
            load_buffer.writeline(
                f"_symm_peer = (_lam_my_buf_base + {i} * _lam_chunk)"
                f".to(tl.pointer_type({tl_dtype}))"
            )
            load_buffer.writeline(
                f"_symm_acc = _symm_acc + _lamport_poll_load("
                f"_symm_peer, ({indexing.index_str}), "
                f"{indexing.mask_str}, R0_BLOCK).to(tl.float32)"
            )


# ------------------------------------------------------------------
# Prologue / epilogue: pull model (device_cas)
# ------------------------------------------------------------------


def codegen_symm_mem_prologue(kernel, code):
    assert kernel.symm_mem_input_name is not None

    if config._symm_mem_skip_prologue_copy:
        code.splice(
            """
            # --- symm mem prologue: input already in symm mem, sync only ---
            _symm_mem_sync(symm_signal_pad_ptrs, None, SYMM_RANK, SYMM_WORLD_SIZE, hasPreviousMemAccess=True, hasSubsequentMemAccess=True)
            """
        )
        return

    code.writeline("# --- symm mem prologue: setup ---")
    for i in range(kernel.symm_mem_world_size):
        prefix = "if" if i == 0 else "elif"
        code.writeline(f"{prefix} SYMM_RANK == {i}:")
        with code.indent():
            code.writeline(f"_symm_local_buf = symm_peer_buf_{i}")

    if config._symm_mem_grid_cap <= 0:
        code.writeline("_symm_x_base = tl.program_id(0).to(tl.int64) * XBLOCK")
    code.splice(
        """
        _symm_cols = tl.arange(0, R0_BLOCK)
        _symm_col_mask = _symm_cols < r0_numel
        """
    )


def codegen_symm_mem_epilogue(kernel, code):
    code.splice(
        """
        # --- symm mem epilogue: signal reads complete ---
        _symm_mem_sync(symm_signal_pad_ptrs, None, SYMM_RANK, SYMM_WORLD_SIZE, hasPreviousMemAccess=True)
        """
    )


# ------------------------------------------------------------------
# Prologue / epilogue: Lamport push model
# ------------------------------------------------------------------


def codegen_lamport_prologue(kernel, code):
    assert kernel.symm_mem_input_name is not None
    assert kernel.symm_mem_input_dtype is not None
    elem_bytes = torch.tensor([], dtype=kernel.symm_mem_input_dtype).element_size()
    assert (
        elem_bytes == 2
    ), f"Lamport protocol requires 2-byte dtype, got {kernel.symm_mem_input_dtype}"
    r0 = kernel.numels.get("r0_")
    if r0 is not None:
        r0_hint = V.graph.sizevars.symbolic_hint(r0)
        if isinstance(r0_hint, (int, sympy.Integer)) and int(r0_hint) % 2 != 0:
            raise RuntimeError(
                f"Lamport allreduce requires even reduction dim, got {int(r0_hint)}. "
                "The sentinel protocol packs 2 bf16 elements per u32 word."
            )
    code.splice(
        f"""
        # ═══ Lamport prologue ═══════════════════════════════════════════
        # PDL stall: wait for the previous kernel's gdc_launch_dependents
        # so CUDA-graph batched replay cannot overrun the 3-slot headroom.
        tl.extra.cuda.gdc_wait()

        # Read triple-buffer flag (volatile) and derive slot offsets.
        _lam_meta_i32 = _lam_meta.to(tl.pointer_type(tl.int32))
        _lam_flag = _lamport_volatile_load_u32(_lam_meta_i32 + 1)
        _lam_chunk = xnumel * r0_numel
        _lam_slot_elems = SYMM_WORLD_SIZE * _lam_chunk
        _lam_buf_offset = (_lam_flag % 3) * _lam_slot_elems
        _lam_clear_offset = ((_lam_flag + 2) % 3) * _lam_slot_elems
        """
    )
    for i in range(kernel.symm_mem_world_size):
        prefix = "if" if i == 0 else "elif"
        code.writeline(f"{prefix} SYMM_RANK == {i}:")
        with code.indent():
            code.writeline(f"_lam_my_buf = symm_peer_buf_{i}")
    code.splice(
        f"""
        _lam_my_buf_base = _lam_my_buf + _lam_buf_offset
        _lam_clear_base = _lam_my_buf + _lam_clear_offset
        _lam_x_base = tl.program_id(0).to(tl.int64) * XBLOCK
        _lam_cols = tl.arange(0, R0_BLOCK)
        _lam_col_mask = _lam_cols < r0_numel
        # ═══ end prologue ═══════════════════════════════════════════════
        """
    )


def codegen_lamport_epilogue(kernel, code):
    code.splice(
        """
        # ═══ Lamport epilogue ═══════════════════════════════════════════
        # Block 0 waits for all blocks, then advances the triple-buffer
        # flag. All blocks must wait for the flag advance before calling
        # gdc_launch_dependents, otherwise the successor kernel reads
        # a stale flag.
        _lamport_advance_flag_block0(_lam_meta_i32, _lam_flag)
        # Non-block-0 blocks poll meta[0] until block 0 resets it to 0
        # (which happens inside _lamport_advance_flag_block0 after the
        # flag is advanced). Block 0 already reset it, so it sees 0.
        if tl.program_id(0) != 0:
            _lam_epilogue_done = tl.full([], 0, dtype=tl.int32)
            while _lam_epilogue_done == 0:
                _lam_cval = _lamport_volatile_load_u32(_lam_meta_i32)
                _lam_epilogue_done = (_lam_cval == 0).to(tl.int32)
        tl.extra.cuda.gdc_launch_dependents()
        # ═══ end epilogue ═══════════════════════════════════════════════
        """
    )


# ------------------------------------------------------------------
# Grid-stride body wrapper (for capped grids with device_cas)
# ------------------------------------------------------------------


def _rewrite_xoffset_for_grid_stride(body_text: str) -> str:
    new_body_text = re.sub(
        r"xoffset = tl\.program_id\(0\)(?:\.to\(tl\.int64\))? \* XBLOCK",
        "xoffset = _x_tile.to(tl.int64) * XBLOCK",
        body_text,
    )
    assert new_body_text != body_text, (
        "Grid-stride body rewrite failed: could not find "
        "'xoffset = tl.program_id(0) * XBLOCK' in kernel body"
    )
    return new_body_text


def _codegen_grid_stride_loop(code, body_text: str):
    code.writeline(
        "for _x_tile in range(tl.program_id(0), tl.cdiv(xnumel, XBLOCK), "
        "tl.num_programs(0)):"
    )
    with code.indent():
        code.writeline("_symm_x_base = _x_tile.to(tl.int64) * XBLOCK")
        code.splice(body_text)


def _extract_two_shot_grid_phases(
    body_text: str,
) -> tuple[str, str | None, str, str]:
    if _TWO_SHOT_COPY_BEGIN in body_text:
        common_prefix, rest = body_text.split(_TWO_SHOT_COPY_BEGIN, 1)
        copy_phase, rest = rest.split(_TWO_SHOT_COPY_END, 1)
        _, rest = rest.split(_TWO_SHOT_REDUCE_BEGIN, 1)
    else:
        common_prefix, rest = body_text.split(_TWO_SHOT_REDUCE_BEGIN, 1)
        copy_phase = None
    reduce_phase, rest = rest.split(_TWO_SHOT_REDUCE_END, 1)
    _, tail = rest.split(_TWO_SHOT_TAIL_BEGIN, 1)
    return common_prefix, copy_phase, reduce_phase, tail


def _split_two_shot_reduce_phase(reduce_phase: str) -> tuple[str, str]:
    reduce_loop = "for _2shot_row in tl.static_range(XBLOCK):"
    reduce_preheader, reduce_body = reduce_phase.split(reduce_loop, 1)
    return reduce_preheader, f"{reduce_loop}{reduce_body}"


def _split_x_tile_setup(common_prefix: str) -> tuple[str, str]:
    invariant_lines = []
    x_tile_lines = []
    found_xoffset = False

    for line in common_prefix.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("xoffset = ", "xindex = ", "xmask = ", "x0 = ")):
            x_tile_lines.append(line)
            if stripped.startswith("xoffset = "):
                found_xoffset = True
        else:
            invariant_lines.append(line)

    assert found_xoffset, (
        "Grid-stride x-tile setup split failed: could not find "
        "'xoffset = ...' in kernel body prefix"
    )
    return "".join(invariant_lines), _rewrite_xoffset_for_grid_stride(
        "".join(x_tile_lines)
    )


def codegen_grid_stride_body(kernel, code):
    _codegen_grid_stride_loop(
        code, _rewrite_xoffset_for_grid_stride(kernel.body.getvalue())
    )


def codegen_two_shot_grid_stride_body(kernel, code):
    common_prefix, copy_phase, reduce_phase, tail = _extract_two_shot_grid_phases(
        kernel.body.getvalue()
    )
    reduce_preheader, reduce_body = _split_two_shot_reduce_phase(reduce_phase)
    compute_preheader, x_tile_setup = _split_x_tile_setup(common_prefix)

    if copy_phase is not None:
        _codegen_grid_stride_loop(code, copy_phase)
        code.writeline(
            "_symm_mem_sync(symm_signal_pad_ptrs, None, SYMM_RANK, "
            "SYMM_WORLD_SIZE, hasPreviousMemAccess=True, "
            "hasSubsequentMemAccess=True)"
        )

    if reduce_preheader.strip():
        code.splice(reduce_preheader)
    _codegen_grid_stride_loop(code, reduce_body)
    code.writeline(
        "_symm_mem_sync(symm_signal_pad_ptrs, None, SYMM_RANK, "
        "SYMM_WORLD_SIZE, hasPreviousMemAccess=True, "
        "hasSubsequentMemAccess=True)"
    )
    if compute_preheader.strip():
        code.splice(compute_preheader)
    _codegen_grid_stride_loop(code, x_tile_setup + tail)


# ------------------------------------------------------------------
# Wrapper emit: setup + epilogue for each sync mode
# ------------------------------------------------------------------


def _resolve_input_wrapper_name(kernel):
    from .common import RemovedArg

    name = kernel.symm_mem_input_name
    assert name is not None
    inplaced = kernel.args.inplace_buffers.get(name)
    if inplaced is not None and not isinstance(inplaced, RemovedArg):
        return inplaced.other_names[-1]
    return name


def emit_symm_mem_setup(kernel, wrapper, call_args: list):
    in_var = _resolve_input_wrapper_name(kernel)

    wrapper.imports.writeline(
        "from torch._inductor.runtime.symm_mem_helpers import symm_mem_peer_bufs"
    )
    wrapper.writeline(
        f"_symm_peer_bufs, _symm_signal_pad_ptrs, _, _ = "
        f'symm_mem_peer_bufs({in_var}, "{kernel._symm_group_name}")'
    )

    for i in range(kernel.symm_mem_world_size):
        call_args.append(f"_symm_peer_bufs[{i}]")
    call_args.append("_symm_signal_pad_ptrs")


def emit_symm_mem_host_barrier_setup(kernel, wrapper, call_args: list):
    in_var = _resolve_input_wrapper_name(kernel)

    skip_copy = "True" if config._symm_mem_skip_prologue_copy else "False"
    wrapper.imports.writeline(
        "from torch._inductor.runtime.symm_mem_helpers import "
        "symm_mem_host_barrier_peer_bufs, symm_mem_host_barrier"
    )
    wrapper.writeline(
        f"_symm_peer_bufs, _, _ = "
        f"symm_mem_host_barrier_peer_bufs("
        f'{in_var}, "{kernel._symm_group_name}", skip_copy={skip_copy})'
    )
    wrapper.writeline(f'symm_mem_host_barrier("{kernel._symm_group_name}")')

    for i in range(kernel.symm_mem_world_size):
        call_args.append(f"_symm_peer_bufs[{i}]")


def emit_symm_mem_host_barrier_epilogue(kernel, wrapper):
    wrapper.writeline(f'symm_mem_host_barrier("{kernel._symm_group_name}")')


def emit_lamport_setup(kernel, wrapper, call_args: list):
    in_var = _resolve_input_wrapper_name(kernel)

    wrapper.imports.writeline(
        "from torch._inductor.runtime.lamport_helpers import "
        "lamport_workspace_peer_bufs"
    )
    wrapper.writeline(
        f"_lam_peer_bufs, _, _, _lam_meta = "
        f'lamport_workspace_peer_bufs({in_var}, "{kernel._symm_group_name}")'
    )

    for i in range(kernel.symm_mem_world_size):
        call_args.append(f"_lam_peer_bufs[{i}]")
    call_args.append("_lam_meta")
