"""
TE grouped GEMM v2: single-launch cuBLASLt via nvte_grouped_gemm.

Loads a C++ extension that calls the true grouped GEMM API from
libtransformer_engine.so. Accepts device offsets, no D2H sync,
fully CUDA-graph and dynamic-routing compatible.

Usage:
    from te_grouped_gemm_v2 import te_grouped_gemm_v2
    out = te_grouped_gemm_v2(A, B, offs)  # same interface as torch._grouped_mm
"""

import os
import torch
import torch.library

_ext = None


def _load_ext():
    global _ext
    if _ext is not None:
        return _ext

    from torch.utils.cpp_extension import load

    import transformer_engine
    te_pkg_dir = os.path.dirname(os.path.abspath(transformer_engine.__file__))
    te_lib_dir = te_pkg_dir
    te_include = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        *([os.pardir] * 3),
        "third_party", "TransformerEngine", "transformer_engine",
        "common", "include",
    )
    te_include = os.path.normpath(te_include)

    ext_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "te_grouped_gemm_ext.cu")

    _ext = load(
        name="te_grouped_gemm_ext",
        sources=[ext_src],
        extra_include_paths=[te_include],
        extra_ldflags=[
            f"-L{te_lib_dir}",
            "-ltransformer_engine",
            f"-Wl,-rpath,{te_lib_dir}",
        ],
        verbose=True,
    )
    return _ext


def te_grouped_gemm_v2(
    A: torch.Tensor,
    B: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    """Single-launch cuBLASLt grouped GEMM via nvte_grouped_gemm.

    Same interface as torch._grouped_mm(A, B, offs).

    Args:
        A: [M_total, K] bf16 on CUDA (2D, rows sorted by group)
        B: [G, K, N] bf16 on CUDA (3D weights)
        offs: [G] int32 on CUDA (cumulative offsets)

    Returns:
        D: [M_total, N] bf16
    """
    ext = _load_ext()
    return ext.grouped_gemm(A, B, offs)


# Custom op for torch.compile
@torch.library.custom_op("te_v2::grouped_gemm", mutates_args=())
def te_grouped_gemm_v2_op(
    A: torch.Tensor,
    B: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    return te_grouped_gemm_v2(A, B, offs)


@te_grouped_gemm_v2_op.register_fake
def te_grouped_gemm_v2_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    offs: torch.Tensor,
) -> torch.Tensor:
    G, K, N = B.shape
    return torch.empty(A.shape[0], N, dtype=A.dtype, device=A.device)


if __name__ == "__main__":
    # Quick smoke test
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    G, M, K, N = 4, 8, 256, 512
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(G, K, N, dtype=torch.bfloat16, device="cuda") * 0.01
    offs = torch.tensor([2, 5, 7, 8], dtype=torch.int32, device="cuda")

    ref = torch._grouped_mm(A, B, offs)
    out = te_grouped_gemm_v2(A, B, offs)

    diff = (ref.float() - out.float()).abs().max().item()
    print(f"ref norm: {ref.float().norm():.4f}")
    print(f"out norm: {out.float().norm():.4f}")
    print(f"max |diff|: {diff:.6e}")
    print(f"PASS: {diff < 0.01}")

    # CUDA graph test
    print("\nCUDA graph test...")
    for _ in range(3):
        te_grouped_gemm_v2(A, B, offs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            te_grouped_gemm_v2(A, B, offs)
    torch.cuda.current_stream().wait_stream(stream)

    with torch.cuda.graph(graph):
        static_out = te_grouped_gemm_v2(A, B, offs)
    graph.replay()
    torch.cuda.synchronize()

    diff2 = (ref.float() - static_out.float()).abs().max().item()
    print(f"graph max |diff|: {diff2:.6e}")
    print(f"CUDA graph PASS: {diff2 < 0.01}")

    # Dynamic routing test: change offsets and replay
    print("\nDynamic routing test...")
    offs.copy_(torch.tensor([3, 4, 6, 8], dtype=torch.int32, device="cuda"))
    ref2 = torch._grouped_mm(A, B, offs)
    graph.replay()
    torch.cuda.synchronize()
    diff3 = (ref2.float() - static_out.float()).abs().max().item()
    print(f"dynamic routing max |diff|: {diff3:.6e}")
    print(f"Dynamic routing PASS: {diff3 < 0.01}")
