# Two-Shot Allreduce Codegen: Inductor vs Kraken

Comparison of the inductor-generated `device_cas_2_shot` kernel against the
handwritten kraken `two_shot_all_reduce_bias_rms_norm` kernel.

Both implement the same algorithm—reduce-scatter then allgather via symmetric
memory—but differ in block structure, which dominates performance at scale.

## Algorithm (shared)

```
1. Copy input → local symm_mem buffer
2. SYNC (all blocks)
3. Reduce-scatter: each rank reduces its column chunk from all peers,
   writes result back to every peer buffer
4. SYNC (all blocks)
5. Load full reduced row from local buffer → fused downstream compute (RMSNorm)
6. SYNC (all blocks, epilogue)
```

Total: 3 grid-wide CAS barriers per kernel launch.

## Benchmark (NUM_TOKENS=1024, HIDDEN=2880, world_size=4)

| Variant              |  μs/iter | Blocks | Rows/block | Syncs |
|----------------------|---------:|-------:|-----------:|------:|
| compiled_2shot       |     46.8 |  1024  |     1      |   3   |
| kraken_2shot         |     34.4 |   128  |     8      |   3   |
| compiled (1-shot)    |     54.6 |  1024  |     1      |   3   |
| compiled_plain (NCCL)|     54.5 |   —    |     —      |   —   |
| baseline (eager)     |     95.0 |   —    |     —      |   —   |

The 12 μs gap between compiled_2shot and kraken_2shot comes from the 8×
difference in block count (1024 vs 128), which means 8× more CAS atomic
contention per barrier.

## Why XBLOCK=1 in the inductor kernel

The inductor persistent-reduction heuristic picks XBLOCK based on register
pressure. The RMSNorm accumulator has shape `[XBLOCK, R0_BLOCK]` (float32):

| XBLOCK | R0_BLOCK | Live regs (float32) | KB    |
|--------|----------|---------------------|-------|
| 1      | 4096     | 4096                | 16    |
| 2      | 4096     | 8192                | 32    |
| 8      | 4096     | 32768               | 128   |

XBLOCK=1 is the safe choice. Higher values risk spilling.

## How kraken avoids this

Kraken's handwritten kernel sets `rows_per_block=8` and does RMSNorm **one row
at a time** inside a `tl.static_range(rows_per_block)` loop. This keeps peak
live registers at 1×D (one row) while still batching 8 rows per block for the
sync phases:

```
num_blocks = total_rows // rows_per_block = 1024 // 8 = 128
```

128 blocks × 3 syncs = 384 CAS atomics total.
vs inductor's 1024 blocks × 3 syncs = 3072 CAS atomics total.

## Generated kernel (inductor `device_cas_2_shot`)

Config: 1024 tokens × 2880 hidden, 4 GPUs, XBLOCK=1, R0_BLOCK=4096.
Single fused kernel: allreduce + residual add + RMSNorm.

```python
@triton.jit
def triton_per_fused_wait_tensor_add_rmsnorm_0(
    in_ptr0,        # input (allreduce source)
    in_ptr1,        # residual
    in_ptr2,        # RMSNorm weight
    out_ptr0,       # pre-norm output (h = reduced + residual)
    out_ptr2,       # normed output
    xnumel, r0_numel,
    XBLOCK: tl.constexpr,
    symm_peer_buf_0, symm_peer_buf_1,
    symm_peer_buf_2, symm_peer_buf_3,
    symm_signal_pad_ptrs,
    SYMM_RANK: tl.constexpr,
    SYMM_WORLD_SIZE: tl.constexpr,
):
    xnumel = 1024
    r0_numel = 2880
    R0_BLOCK: tl.constexpr = 4096

    # ── Prologue: select local buffer ──
    if SYMM_RANK == 0:
        _symm_local_buf = symm_peer_buf_0
    elif SYMM_RANK == 1:
        _symm_local_buf = symm_peer_buf_1
    elif SYMM_RANK == 2:
        _symm_local_buf = symm_peer_buf_2
    elif SYMM_RANK == 3:
        _symm_local_buf = symm_peer_buf_3

    _symm_x_base = tl.program_id(0).to(tl.int64) * XBLOCK
    _symm_cols = tl.arange(0, R0_BLOCK)           # [4096]
    _symm_col_mask = _symm_cols < r0_numel

    # ── Standard reduction setup ──
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]   # [1, 1]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]         # [1, 4096]
    r0_mask = r0_index < r0_numel
    x0 = xindex

    # ── Phase 1: copy input → local symm_mem buffer ──
    for _symm_row in tl.static_range(XBLOCK):           # XBLOCK=1, once
        _symm_row_idx = _symm_x_base + _symm_row
        _symm_row_mask = _symm_row_idx < xnumel
        _symm_idx = _symm_row_idx * r0_numel + _symm_cols
        _symm_mask = _symm_col_mask & _symm_row_mask
        _symm_val = tl.load(in_ptr0 + _symm_idx, _symm_mask, other=0.0)
        tl.store(_symm_local_buf + _symm_idx, _symm_val, _symm_mask)
    _symm_mem_sync(...)  # BARRIER 1

    # ── Phase 2: reduce-scatter (chunk-sized vectors) ──
    _2shot_chunk = r0_numel // SYMM_WORLD_SIZE           # 2880 // 4 = 720
    _2shot_cols = tl.arange(0, R0_BLOCK // SYMM_WORLD_SIZE)  # [1024]
    _2shot_col_mask = _2shot_cols < _2shot_chunk          # mask to 720
    _2shot_col_start = SYMM_RANK * _2shot_chunk
    for _2shot_row in tl.static_range(XBLOCK):            # XBLOCK=1, once
        _2shot_row_idx = _symm_x_base + _2shot_row
        _2shot_row_mask = _2shot_row_idx < xnumel
        _2shot_idx = _2shot_row_idx * r0_numel + _2shot_col_start + _2shot_cols
        _2shot_mask = _2shot_col_mask & _2shot_row_mask
        _2shot_acc = tl.zeros([R0_BLOCK // SYMM_WORLD_SIZE], dtype=tl.float32)
        _2shot_acc += tl.load(symm_peer_buf_0 + _2shot_idx, _2shot_mask, other=0.0).to(tl.float32)
        _2shot_acc += tl.load(symm_peer_buf_1 + _2shot_idx, _2shot_mask, other=0.0).to(tl.float32)
        _2shot_acc += tl.load(symm_peer_buf_2 + _2shot_idx, _2shot_mask, other=0.0).to(tl.float32)
        _2shot_acc += tl.load(symm_peer_buf_3 + _2shot_idx, _2shot_mask, other=0.0).to(tl.float32)
        tl.store(symm_peer_buf_0 + _2shot_idx, _2shot_acc, _2shot_mask)
        tl.store(symm_peer_buf_1 + _2shot_idx, _2shot_acc, _2shot_mask)
        tl.store(symm_peer_buf_2 + _2shot_idx, _2shot_acc, _2shot_mask)
        tl.store(symm_peer_buf_3 + _2shot_idx, _2shot_acc, _2shot_mask)
    _symm_mem_sync(...)  # BARRIER 2

    # ── Phase 3: load full result + fused add-residual + RMSNorm ──
    _symm_acc = tl.load(_symm_local_buf + (r0_1 + 2880*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 2880*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr2 + (r0_1), r0_mask, other=0.0)   # RMSNorm weight
    h = _symm_acc + tmp1                          # add residual
    h_f32 = h.to(tl.float32)
    variance = tl.sum(h_f32 * h_f32, 1)[:, None] / 2880.0
    rstd = libdevice.rsqrt(variance + eps)
    normed = (h_f32 * rstd * tmp15).to(tl.bfloat16)
    tl.store(out_ptr0, h, ...)                    # pre-norm
    tl.store(out_ptr2, normed, ...)               # normed

    _symm_mem_sync(...)  # BARRIER 3 (epilogue)
```

Grid: **1024 blocks** (one per row), **1 kernel launch**, **3 barriers**.

## Kraken handwritten kernel

Config: 1024 tokens × 2880 hidden, 4 GPUs, rows_per_block=8.
Single kernel: allreduce + bias add + RMSNorm (no residual-add variant).

```python
@triton.jit
def two_shot_all_reduce_bias_rms_norm_kernel(
    symm_mem_buffer_ptrs, symm_mem_signal_pad_ptrs,
    input_ptr, bias_ptr, w_ptr, y_ptr,
    eps: tl.constexpr, D: tl.constexpr,
    size_per_rank: tl.constexpr,       # rows_per_block * D // world_size
    rows_per_block: tl.constexpr,      # 8
    rank: tl.constexpr, world_size: tl.constexpr,
    ...
):
    row_idx = tl.program_id(0) * rows_per_block        # block 0 → rows 0..7
    col_offsets = tl.arange(0, next_power_of_2(D))      # [4096]
    mask = col_offsets < D
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))

    # ── Phase 1: copy ALL rows_per_block rows → local buffer ──
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    for i in tl.static_range(rows_per_block):            # 8 iterations
        row = tl.load(input_ptr + offset + i * D + col_offsets, mask=mask)
        tl.store(buffer_ptr + offset + i * D + col_offsets, row, mask=mask)
    symm_mem_sync(...)  # BARRIER 1

    # ── Phase 2: reduce-scatter (FLAT across all 8 rows) ──
    # size_per_rank = 8 * 2880 / 4 = 5760
    # Flat chunk spanning multiple rows — NOT per-row iteration
    local_rank_offsets = (
        offset
        + size_per_rank * rank
        + tl.arange(0, next_power_of_2(size_per_rank))  # [8192]
    )
    local_rank_mask = local_rank_offsets < (offset + size_per_rank * (rank + 1))

    acc = tl.load(bias_ptr + local_rank_offsets, mask=local_rank_mask).to(tl.float32)
    for remote_rank in range(world_size):
        buf = tl.load(buffer_ptrs + remote_rank).to(tl.pointer_type(tl.bfloat16))
        acc += tl.load(buf + local_rank_offsets, mask=local_rank_mask).to(tl.float32)
    for remote_rank in range(world_size):
        buf = tl.load(buffer_ptrs + remote_rank).to(tl.pointer_type(tl.bfloat16))
        tl.store(buf + local_rank_offsets, acc, mask=local_rank_mask)
    symm_mem_sync(...)  # BARRIER 2

    # ── Phase 3: RMSNorm (one row at a time → low register pressure) ──
    buffer_ptr = tl.load(buffer_ptrs + rank).to(tl.pointer_type(tl.bfloat16))
    for i in tl.static_range(rows_per_block):            # 8 iterations
        row = tl.load(buffer_ptr + offset + i * D + col_offsets, mask=mask).to(tl.float32)
        variance = tl.sum(row * row, axis=0) / D
        rstd = tl_rsqrt(variance + eps)
        w = tl.load(w_ptr + col_offsets, mask=mask).to(tl.float32)
        tl.store(y_ptr + offset + i * D + col_offsets, row * rstd * w, mask=mask)
    symm_mem_sync(...)  # BARRIER 3
```

Grid: **128 blocks** (each handles 8 rows), **1 kernel launch**, **3 barriers**.

## Key structural differences

| Aspect                    | Inductor codegen             | Kraken handwritten          |
|---------------------------|-----------------------------|-----------------------------|
| Rows per block            | XBLOCK=1 (heuristic)       | rows_per_block=8 (manual)   |
| Grid size                 | 1024                        | 128                         |
| Reduce-scatter vectorization | Per-row, chunk-sized [1024] | Flat across rows [8192]    |
| RMSNorm live registers    | XBLOCK×R0_BLOCK = 4096      | 1×D = 2880 (loop over rows)|
| Downstream fusion         | Yes (add residual + RMSNorm)| Partial (bias + RMSNorm)   |
| Sync overhead (CAS atoms) | 1024 × 3 = 3072             | 128 × 3 = 384              |

## Path to closing the gap

The inductor kernel picks XBLOCK=1 because the persistent-reduction accumulator
`[XBLOCK, R0_BLOCK]` is materialized at full width for the RMSNorm sum. Kraken
avoids this by doing RMSNorm in a row-by-row loop, keeping peak liveness at one
row.

Possible approaches:
1. **Decouple sync granularity from XBLOCK**: process multiple rows per block
   for copy/reduce-scatter phases (batch the syncs), then do RMSNorm one row at
   a time. Requires a multi-phase kernel structure in the codegen.
2. **Grid cap with batched syncs**: extend `_symm_mem_grid_cap` so that copy and
   reduce-scatter phases iterate over all tiles before syncing (rather than
   syncing per tile in the grid-stride loop).
3. **Flat reduce-scatter**: flatten the reduce-scatter across `XBLOCK * r0_numel`
   elements (matching kraken) instead of iterating per row. This is orthogonal
   to the block count issue but improves vectorization.
