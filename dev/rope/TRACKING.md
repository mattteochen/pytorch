# Fused GEMM + RoPE: Current Status

## Implemented

- Added an inference-only GEMM+RoPE fusion pass in [torch/_inductor/fx_passes/fuse_gemm_rope.py](/opt/pytorch/pytorch/torch/_inductor/fx_passes/fuse_gemm_rope.py).
- Wired the pass into the normal Inductor post-grad pipeline in [torch/_inductor/fx_passes/post_grad.py](/opt/pytorch/pytorch/torch/_inductor/fx_passes/post_grad.py), aligned with the existing `b2b_gemm` pattern.
- Added a small GEMM plugin registry in [torch/_inductor/kernel/gemm_plugin.py](/opt/pytorch/pytorch/torch/_inductor/kernel/gemm_plugin.py).
- Registered RoPE as plugin `rope_neox` in [torch/_inductor/kernel/gemm_rope.py](/opt/pytorch/pytorch/torch/_inductor/kernel/gemm_rope.py).
- Added a Triton-template extension surface in [torch/_inductor/select_algorithm.py](/opt/pytorch/pytorch/torch/_inductor/select_algorithm.py):
  - `extra_template_env_fn_builders`
  - `output_ptr()`
- Implemented the fused GEMM+RoPE Triton template/lowering in [torch/_inductor/kernel/gemm_rope.py](/opt/pytorch/pytorch/torch/_inductor/kernel/gemm_rope.py).
- Removed the dummy template output store path from the generated fused kernel.
- Made the fused path return V directly in the expected layout, which removed the extra V copy kernel.
- Broadened the matcher to handle the real sglang-style inference graph:
  - `linear` / `addmm`
  - `t` / `permute([1, 0])`
  - split-based RoPE halves
  - duplicated `unsqueeze(...).to(dtype)` cos/sin chains
- Added a defensive fallback: if the plugin is not registered, the pass skips fusion instead of failing compilation.
- Fixed tuple metadata on the fused replacement node so fake-tensor propagation works correctly, including `num_tokens=1`.

## Current Behavior

- The fusion is inference-only.
- The post-grad matcher rewrites the QKV GEMM + Q/K RoPE pattern to a fused plugin lowering.
- The fused lowering autotunes over the full fused GEMM+RoPE kernel, not GEMM alone.
- For the real `dev/rope/main.py` workload, the generated path now produces a single fused Triton kernel for the GEMM+RoPE portion.
- The generated inference path is numerically close to the baseline `torch.compile` path:
  - typical Q/K max abs diff is in the `3.125e-02` to `6.25e-02` range for bf16
  - V is exact

## Validation Added

- Standalone repro:
  - [dev/rope/repro_gemm_rope.py](/opt/pytorch/pytorch/dev/rope/repro_gemm_rope.py)
- Focused Inductor test:
  - [test/inductor/test_gemm_rope.py](/opt/pytorch/pytorch/test/inductor/test_gemm_rope.py)
- Benchmark harness:
  - [dev/rope/main.py](/opt/pytorch/pytorch/dev/rope/main.py)
  - supports multiple `--num-tokens`
  - compares baseline `torch.compile` vs `compile+gemm_rope`
  - supports CUDA graphs
  - optional `--nvtx`
  - prints one final summary table across token counts

## Performance Notes

- The fused path uses the standard Inductor autotune infrastructure through `autotune_select_algorithm(...)`.
- Autotuning runs once per compiled shape, then the chosen kernel is reused.
- Under CUDA-graph-style GPU timing, the fused path is competitive with the handwritten kernel and can be faster than the baseline compiled path for larger token counts.

## Remaining Work

1. Tighten correctness expectations and coverage.
   - Add more realistic-shape test coverage if needed.
   - Decide whether the current bf16 Q/K error envelope is acceptable for the target workload.

2. Improve benchmark output polish.
   - Optional: suppress or redirect compile-time autotune/debug logging during multi-shape benchmark runs.

3. Generalize the plugin model if we want more users.
   - Today the registry + template hooks are sufficient for RoPE.
   - A more general multi-output GEMM plugin API is still future work.

4. Revisit the matcher only if graph forms change again.
   - The current explicit graph walk is more flexible than a single declarative pattern and matches the observed inference graphs.

## Non-Goals For Now

- General training support
- A fully generic vector-epilogue framework
- Replacing the custom matcher with a rigid `register_graph_pattern(...)` form
