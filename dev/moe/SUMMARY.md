# MoE Native Grouped GEMM: Investigation Summary

**Model**: gpt-oss-20b (32 experts, top_k=4, hidden=2880, intermediate=2880, bf16)
**Hardware**: NVIDIA GB200 (Blackwell, SM100)
**Goal**: Close the latency gap between torch.compile + torch._grouped_mm and the hand-fused Triton fused_experts kernel from SGLang.

---

## Phase 1: Inductor Fusion Analysis

### Starting point (V0)

The native grouped_mm path used index_add_ for accumulation and no max-autotune. Result: **9 CUDA kernel launches** with ATEN/CUTLASS fallback GEMMs.

### V1: Replace index_add_ with scatter_add_

index_add_ is in Inductor's FALLBACK_ALLOW_LIST -- it becomes an opaque ExternKernel that blocks all fusion. Switching to scatter_add_ lets Inductor decompose it into native Pointwise + Scatter nodes, fusing the zeros init and bias/scale prep.

Result: **9 to 7 kernel launches** (-2: zeros init + index_add fallback).

### V2: Enable max_autotune_gemm with Triton backend

The ATEN fallback launches 2 kernels per GEMM (prepare_grouped_gemm_data + CUTLASS kernel). Enabling max_autotune_gemm=True with TRITON backend uses Inductor's persistent TMA-based Triton grouped_mm template -- single kernel per GEMM.

Result: **7 to 5 kernel launches** (-2: no more prepare steps).

Additional optimizations:
- combo_kernels=True: Packs two independent pointwise kernels into one dispatch
- searchsorted instead of histc + cumsum: Eliminated 2 eager fallback kernels
- inv_perm via scatter: inv_perm[perm] = arange(...) fuses into the sort kernel

### V2-final: Replace scatter_add_ with view + sum

After inv_perm restores original token ordering, consecutive groups of top_k rows belong to the same token. view(num_tokens, top_k, hidden).sum(dim=1) is mathematically equivalent to scatter_add_ but avoids mutation and atomics. Inductor fully unrolls the top_k=4 sum into a single pointwise kernel.

Result: **7 to 6 kernel launches** (zeros init eliminated, atomic scatter replaced by clean unrolled pointwise).

### Final kernel schedule (V2, 1 token)

- Kernel 0: Routing -- sort + gather + inv_perm
- Kernel 1: Combo -- searchsorted offsets + hidden gather
- Kernel 2: GEMM1 (Triton template, TMA)
- Kernel 3: Activation -- bias + swiglu
- Kernel 4: GEMM2 (Triton template, TMA)
- Kernel 5: Epilogue -- bias + scale + inv_perm + unrolled sum

### Performance (1 token, CUDA graph, GB200)

- Triton fused_experts (SGLang): 62.1 us (1.00x)
- Native V2 (view+sum): 66.9 us (1.08x)
- Native V1 (scatter_add): 67.7 us (1.09x)
- Native V0 (original): ~78 us (1.26x)

---

## Phase 2: Fusion Blocker Analysis

### Why epilogue fusion doesn't work for this MoE pattern

**GEMM1 epilogue blocked by shape mismatch:**
GEMM1 outputs [M, 5760] but swiglu produces [M, 2880] -- a 2:1 column reduction. Inductor's epilogue API is per-element: epilogue(acc[row, col]) returns one transformed value. The swiglu needs f(acc[row, 2*col], acc[row, 2*col+1]) -- cross-element access the API can't express.

The _split_iteration_ranges function in simd.py fails with CantSplit when the epilogue iteration space (4x2880) doesn't evenly divide the template's (4x5760).

**GEMM2 epilogue blocked by shape mismatch (view+sum):**
GEMM2 outputs [num_tokens * top_k, 2880] but the epilogue reduces to [num_tokens, 2880] via the top_k sum. Same structural issue.

**Conclusion**: Both GEMMs have shape-changing downstream ops that prevent epilogue fusion. This is fundamental to the MoE architecture.

### What was investigated but didn't help

- aggressive_fusion: No effect -- blockers are structural
- score_fusion_memory_threshold=0: No effect
- Changing weight layout (interleaved to contiguous halves): Doesn't help -- shape change is the blocker
- Split gate/up into separate GEMMs (V3): 1.44x slower -- two half-sized GEMMs can't saturate GPU
- Hand-written fused GEMM1+swiglu kernel (V4 stride-2): 1.52x slower -- non-coalesced access
- Hand-written fused GEMM1+swiglu kernel (V4 coalesced): 1.43x slower -- still can't match TMA template
- Hand-written fused routing kernel: 1.12x slower -- serializes gather on one threadblock

---

## Phase 3: Alternative GEMM Backends

### Backend availability for _grouped_mm

- ATEN: Yes, CUDA graph compatible (CUTLASS C++ grouped GEMM)
- TRITON: Yes, CUDA graph compatible (Inductor template with TMA)
- CUTLASS: No grouped_mm support (regular mm only)
- CUTEDSL: Yes, but NOT CUDA graph compatible (D2H sync in compute_total_num_clusters)
- NVGEMM: Yes, CUDA graph compatible

### CuTe DSL CUDA graph bug

The cutedsl_mm_grouped.py.jinja template computes total_num_clusters = int(compute_total_num_clusters(probs_t, ...)) which iterates over a CUDA tensor on the host -- D2H sync forbidden during graph capture. The Triton template uses a fixed grid (NUM_SMS) instead.

### ATEN backend

torch._grouped_mm with ATEN backend calls bf16bf16_grouped_mm (aten/src/ATen/native/cuda/GroupMM.cu) -- pre-compiled CUTLASS 3.x grouped GEMM with GemmUniversalMode::kGrouped. Single kernel launch, device offsets, fully CUDA-graph compatible.

### Config gotcha

TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS env var is read at torch._inductor.config import time. Setting it after import has no effect. Use torch._inductor.config.max_autotune_gemm_backends = "..." directly.

---

## Phase 4: TransformerEngine Integration

### TE general_grouped_gemm (multi-tensor path)

TE's Python API calls nvte_multi_tensor_gemm -- a host-side loop launching individual cublasLtMatmul per group. Requires CPU-side m_splits (Python list), forcing D2H sync. Fundamentally incompatible with dynamic routing under CUDA graph replay.

Performance (1 token, CUDA graph): 75.4 us (1.19x vs Triton). Uses NVIDIA's optimized nvjet_sm100_tst_* kernels but per-group launch overhead adds up.

### TE nvte_grouped_gemm (true grouped path)

TE has a C-only API (nvte_grouped_gemm) using cublasLtMatmul in grouped mode -- single launch for all groups. GPU setup kernel reads offsets from device memory. Fully CUDA-graph and dynamic-routing compatible. Marked EXPERIMENTAL, no Python binding.

### Standalone C++ extension

We wrote te_grouped_gemm_ext.cu that:
1. Links against installed libtransformer_engine.so (no TE repo changes)
2. Takes PyTorch tensors A [M,K], B [G,K,N], offs [G] (all device)
3. Constructs NVTEGroupedTensor via public C API
4. Uses row-major trick: swaps A/B so cuBLAS col-major output = row-major result
5. GPU kernel converts cumulative offs (int32) to first_dims/offsets (int64)
6. Registered as torch.ops.te_v2.grouped_gemm custom op

Correctness: Exact match (diff = 0.0) vs torch._grouped_mm including dynamic routing under CUDA graph replay.

### TE performance results

1 token, CUDA graph:
- TE multi-tensor (general_grouped_gemm): 75.4 us (1.19x)
- TE grouped (nvte_grouped_gemm): 98.9 us (1.56x)

2048 tokens, CUDA graph:
- TE grouped: 954.4 us (1.62x vs native V1 at 587.9 us)

Each GEMM via nvte_grouped_gemm needs 3 kernel launches:
1. compute_grouped_metadata (our offset conversion)
2. setup_grouped_gemm_kernel (TE's pointer array setup)
3. nvjet_sm100_tst_128x128_64x6_... (cuBLASLt grouped GEMM)

The 2 extra kernels per GEMM plus suboptimal cuBLASLt algorithm selection for MoE workloads explains the regression. The cuBLASLt grouped GEMM API appears to select a tile config (128x128) that's too large for the per-group problem sizes in MoE (a few tokens per expert).

---

## Key Findings

1. **torch.compile + torch._grouped_mm with Triton templates is the best native approach** -- 6 kernels, 1.08x vs hand-fused Triton at 1 token.

2. **The remaining 8% gap is irreducible launch overhead** -- 6 separate kernels vs 1 hand-fused kernel. Each launch costs ~1-2us on Blackwell.

3. **Replacing scatter_add_ with view + sum is the most impactful single change** -- eliminates mutation, zeros init, and atomics.

4. **cuBLASLt grouped GEMM (nvte_grouped_gemm) is slower than Inductor's Triton template** for this workload. The setup overhead and suboptimal algorithm selection make it uncompetitive.

5. **TE's multi-tensor path (individual cuBLASLt per group) was faster than the grouped API** at 1 token (75.4 vs 98.9 us) due to better per-group kernel selection, but can't support dynamic routing under CUDA graphs.

6. **The ATEN backend (CUTLASS grouped GEMM)** deserves further investigation -- single launch, device offsets, CUDA-graph native, no external dependencies.

---

## Files

- dev/moe/moe.py: Main benchmark with 5 backends
- dev/moe/te_grouped_gemm_ext.cu: C++ extension calling nvte_grouped_gemm
- dev/moe/te_grouped_gemm_v2.py: Python wrapper + custom op for the C++ extension
- dev/moe/te_grouped_gemm_repro.py: Earlier TE repro using general_grouped_gemm
- dev/moe/context.md: Chat log Phase 1 (fusion analysis)
- dev/moe/contex2.md: Chat log Phase 2-3 (fusion blockers, fused kernels, TE)
- dev/moe/SUMMARY.md: This file
