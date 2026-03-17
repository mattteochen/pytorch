# Fused GEMM + RoPE: Next Steps

## Current State

We have a working standalone Triton kernel (`fused_gemm_rope.py`) that fuses
packed QKV projection (addmm) with neox-style RoPE into a single kernel.
The kernel is validated against eager and delivers 1.7-4.1x speedup over eager
and ~2x over torch.compile for decode workloads (1-128 tokens).

The kernel works because `BLOCK_N == head_dim == 64`: each output tile covers
exactly one head, so the x1/x2 halves live in the same accumulator and RoPE
can be applied in-register via `reshape → permute → split → rotate → join →
permute → reshape`.

## Why torch.compile Can't Fuse This Today

Four blockers in the Inductor scheduler (see `ir_pre_fusion.txt`):

1. **Cross-element reads**: RoPE output element `[i]` reads GEMM output at
   both `[i]` and `[i±32]`. The `fusable_read_and_write` check requires
   1:1 index matching between producer write and consumer read.
2. **buf0 is a graph output**: V is returned as a view of the GEMM buffer,
   so it must be materialized regardless.
3. **Two output buffers**: Q and K are separate outputs; template epilogue
   only handles one at a time.
4. **Shape mismatch**: GEMM output is `[M, 5120]` flat, RoPE outputs are
   `[M, 64, 64]` and `[M, 8, 64]`.

## Path 1: FX Pattern Match → Custom Triton Template (Recommended)

This is the most natural integration path. It mirrors how `mm_plus_mm`,
`b2b_gemm`, and `scaled_mm` work in the existing codebase.

### Step 1: Define a Custom Op

Register a new composite op that represents the fused computation:

```python
# torch/_inductor/kernel/gemm_rope.py (new file)

@torch.library.custom_op("inductor::gemm_rope", mutates_args=())
def gemm_rope(
    hidden_states: Tensor,    # [M, K]
    weight: Tensor,           # [N, K]
    bias: Tensor,             # [N]
    cos_sin_cache: Tensor,    # [max_pos, rotary_dim]
    positions: Tensor,        # [M]
    q_size: int,
    kv_size: int,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[Tensor, Tensor, Tensor]:  # q, k, v
    ...
```

### Step 2: FX Pass to Recognize the Pattern

Add a lowering pattern in `torch/_inductor/fx_passes/post_grad.py` (or a new
`rope_fusion.py` under `fx_passes/`). The pattern to match is:

```
addmm(bias, hidden, weight.T)
  → split_with_sizes([q_size, kv_size, kv_size])
    → [0] → view → chunk(2) → [mul, mul, sub, mul, mul, add] → cat  (Q rope)
    → [1] → view → chunk(2) → [mul, mul, sub, mul, mul, add] → cat  (K rope)
    → [2]  (V passthrough)
```

Use `register_lowering_pattern` with `CallFunction` nesting. The `extra_check`
validates:
- `split_with_sizes` dim is -1 or the last dim
- Chunk size is 2 on the last dim (neox-style halving)
- The cos/sin tensors originate from the same `index_select` on a cache
- `head_dim` divides `q_size` and `kv_size` evenly
- `rotary_dim == head_dim` (for now)

Use `MultiOutputPattern` since the pattern has three outputs (q, k, v).

Key references in the codebase:
- `mm_plus_mm` pattern: `post_grad.py:862`
- `matmul_fuse_pattern` (multi-output): `freezing_patterns.py:221`
- `b2b_gemm` (complex multi-node): `b2b_gemm.py:550`
- `fuse_attention.py` patterns: `fuse_attention.py:26`

### Step 3: Jinja2 Triton Template

Create `torch/_inductor/kernel/templates/triton_gemm_rope.py.jinja`.

The template reuses the standard `triton_mm.py.jinja` accumulation loop
verbatim and replaces the `{{store_output(...)}}` section with custom
RoPE epilogue logic:

```jinja
{# --- identical to triton_mm.py.jinja up through the K-loop --- #}
{{def_kernel("A", "B")}}
    ...
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        ...
        acc += tl.dot(a, b, ...)

    # rematerialize rm and rn
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    {# --- custom epilogue starts here --- #}
    # bias add (reuse addmm_epilogue pattern)
    {{load_input("bias", "bias_val", ("idx_n",), mask="mask_n", indent_width=4)}}
    acc = acc + bias_val

    col_start = pid_n * BLOCK_N
    needs_rope = col_start < (q_size + kv_size)

    if needs_rope:
        # load positions[rm] and gather cos/sin from cache
        pos = tl.load(positions_ptr + rm, mask=(rm < M), other=0)
        cos_offs = pos[:, None] * {{ROTARY_DIM}} + tl.arange(0, {{HALF_ROTARY}})[None, :]
        sin_offs = pos[:, None] * {{ROTARY_DIM}} + ({{HALF_ROTARY}} + tl.arange(0, {{HALF_ROTARY}}))[None, :]
        cos = tl.load(cos_sin_ptr + cos_offs, mask=(rm[:, None] < M), other=0.0)
        sin = tl.load(cos_sin_ptr + sin_offs, mask=(rm[:, None] < M), other=0.0)

        acc_3d = tl.reshape(acc, (BLOCK_M, 2, {{HALF_ROTARY}}))
        acc_p = tl.permute(acc_3d, (0, 2, 1))
        x1, x2 = tl.split(acc_p)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        out_p = tl.permute(tl.join(o1, o2), (0, 2, 1))
        acc = tl.reshape(out_p, (BLOCK_M, BLOCK_N))

    result = acc.to({{OUT_DTYPE}})

    # store to Q, K, or V based on col_start
    ...
```

However, there is a catch with the Jinja approach: the existing `{{def_kernel}}`
and `{{load_input}}` macros are tightly coupled to the `TritonTemplateKernel`
codegen in `select_algorithm.py`. The RoPE epilogue needs **extra kernel
arguments** (cos_sin_cache pointer, positions pointer, q_size, kv_size) and
**extra output pointers** (q_ptr, k_ptr, v_ptr) that don't fit the standard
single-output template model. Two options:

**Option A: Pass extras via `suffix_args` + custom `store_output`.**
Override `store_output` in a `TritonTemplateKernel` subclass so that the
suffix input nodes (cos_sin_cache, positions) are loaded with custom
indexing rather than `make_loader()(index_symbols)`. This requires
modifying `select_algorithm.py` to support a `custom_epilogue_codegen`
hook.

**Option B: Hardcode the template (like `b2b_gemm`).**
The `b2b_gemm.py` template avoids the Jinja macro system entirely and
builds Triton code as a Python string with manual argument lists. This
is simpler and avoids fighting the macro system, at the cost of not
reusing `{{load_input}}` helpers. Given that the RoPE epilogue has
fundamentally different requirements (indirect indexing, multi-output),
this is likely the pragmatic choice for a first version.

### Step 4: Template Registration and Autotuning

```python
# torch/_inductor/kernel/gemm_rope.py

gemm_rope_template = TritonTemplate(
    name="gemm_rope",
    grid=mm_grid,
    source=load_kernel_template("triton_gemm_rope"),
)

def tuned_gemm_rope(hidden, weight, bias, cos_sin_cache, positions,
                     q_size, kv_size, head_dim, rotary_dim, *, layout):
    m, n, k, layout, hidden, weight, bias = mm_args(hidden, weight, bias, layout=layout)

    choices = []
    # extern fallback: unfused addmm + rope
    choices.append(aten_fallback_choice(...))

    # triton template choices
    kernel_inputs = MMKernelInputs(
        [bias, hidden, weight, cos_sin_cache, positions],
        scalars={"q_size": q_size, "kv_size": kv_size, ...},
    )
    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            [gemm_rope_template],
            "gemm_rope",
        )
    )
    return autotune_select_algorithm("gemm_rope", choices, ...)
```

The autotuner would sweep BLOCK_M, BLOCK_K, num_stages, num_warps while
keeping BLOCK_N fixed at head_dim.

### Step 5: Register the Lowering

```python
# In gemm_rope.py or post_grad.py
@register_lowering(torch.ops.inductor.gemm_rope)
def gemm_rope_lowering(hidden, weight, bias, cos_sin_cache, positions, ...):
    return tuned_gemm_rope(hidden, weight, bias, cos_sin_cache, positions, ...)
```

## Path 2: Extend the Existing addmm Template Epilogue (Harder)

Instead of a new template, extend the existing `store_output` mechanism to
support RoPE-style epilogues. This is more general but requires deeper changes.

### What Would Need to Change

1. **`store_output` in `select_algorithm.py`**: Support a `custom_load_fn`
   per suffix arg, instead of always using `make_loader()(index_symbols)`.
   RoPE needs `cos_sin_cache[positions[idx_m], idx_n % half_rotary]` —
   an indirect gather that depends on a different tensor.

2. **Multi-output epilogues**: Today `store_output` writes to a single
   `output_node`. For GEMM+RoPE we need to write to Q, K, V based on
   `idx_n`. This requires either:
   - A `MultiOutputTemplateBuffer` IR node, or
   - Conditional stores within a single `store_output` call

3. **In-tile shuffle in epilogue**: The `reshape → permute → split` to
   extract x1/x2 from the accumulator is not expressible via `V.ops.*`
   scalar operations (which operate on a single element). The epilogue
   function would need access to the full tile (vector ops), not just
   scalar pointwise ops.

4. **Scheduler fusion**: `can_fuse_vertical` / `fusable_read_and_write`
   would need to understand that the cross-element read pattern
   `buf0[i]` + `buf0[i+32]` is fusable when `BLOCK_N ≥ 64` (both
   elements are within the same tile).

### Why This Is Hard

The core abstraction in `store_output` is **scalar**: the epilogue_fn
receives one scalar value (`acc` at a single `(row, col)`) and returns one
scalar. RoPE needs a **vector** operation across the tile's column
dimension. This is a fundamental mismatch with the current design.

Changing this would mean either:
- Introducing a "vector epilogue" concept (the epilogue_fn receives the
  full `[BLOCK_M, BLOCK_N]` acc tensor), or
- Running the epilogue in two passes (one for each half), which doubles
  the store traffic

Neither is a small change. The scalar epilogue design is intentional — it
composes cleanly with the scheduler's dynamic epilogue fusion (where
arbitrary downstream pointwise ops are fused into the store loop).

**Recommendation**: Path 2 is worth pursuing eventually for generality, but
Path 1 (custom template + FX pass) is the right first step. It can ship
independently and serve as the reference implementation that a future
generalized epilogue system would need to match.

## Path 3: Torch Library Custom Op (Simplest Deployment)

If the goal is to deploy quickly in SGLang without modifying PyTorch core:

1. Register `gemm_rope` as a `torch.library.custom_op`
2. Provide the Triton kernel as the CUDA implementation
3. Provide a decomposition (addmm + split + rope) as the abstract impl
4. Users call `torch.ops.mylib.gemm_rope(...)` directly in their model code
5. `torch.compile` traces through the decomposition for correctness but
   uses the custom Triton kernel for execution

This requires zero changes to Inductor but requires model code changes.

## Summary

| Path | Effort | Generality | PyTorch Changes |
|------|--------|-----------|----------------|
| 1. FX pass + custom template | Medium | GEMM+RoPE specific | New files in inductor |
| 2. Extend epilogue system | High | Any tile-local epilogue | Core scheduler + codegen |
| 3. Custom op library | Low | None (manual call sites) | None |

**Recommended order**: Path 3 for immediate deployment → Path 1 for
automatic fusion in torch.compile → Path 2 for long-term generalization.

## Open Questions

- Should we support partial rotation (`rotary_dim < head_dim`)? The V
  passthrough is already handled; partial rotation would require storing
  the unrotated tail of Q/K heads.
- Should the FX pass handle both neox-style and gptj-style RoPE? The
  Triton template would need a constexpr `IS_NEOX` flag.
- For the autotuner: can we share the existing `mm_template` config
  space and just add the RoPE overhead, or do we need a separate
  config search?
- The `BLOCK_N == head_dim` constraint means this only works for
  head_dim that is a valid Triton tile size (16, 32, 64, 128). Most
  models use 64 or 128, but some use 96 or 256.
