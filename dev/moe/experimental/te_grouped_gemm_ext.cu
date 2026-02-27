// Standalone C++ extension calling nvte_grouped_gemm from libtransformer_engine.so.
// Provides a single-launch cuBLASLt grouped GEMM with device offsets,
// compatible with CUDA graphs and dynamic routing.
//
// Build: loaded via torch.utils.cpp_extension from te_grouped_gemm_v2.py

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

// Small kernel: convert cumulative offsets [G] (int32) to:
//   first_dims [G] (int64): per-group row counts
//   tensor_offsets_a [G] (int64): element offsets into A (multiply by K)
//   tensor_offsets_d [G] (int64): element offsets into D (multiply by N)
__global__ void compute_grouped_metadata(
    const int32_t* __restrict__ cum_offs,   // [G] cumulative offsets
    int64_t* __restrict__ first_dims,       // [G] per-group M sizes
    int64_t* __restrict__ offsets_a,        // [G] element offsets for A
    int64_t* __restrict__ offsets_d,        // [G] element offsets for D
    int64_t* __restrict__ offsets_b,        // [G] element offsets for B
    int64_t K, int64_t N, int64_t KN,
    int G
) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= G) return;

    int32_t start = (g == 0) ? 0 : cum_offs[g - 1];
    int32_t end = cum_offs[g];
    int64_t m_g = static_cast<int64_t>(end - start);

    first_dims[g] = m_g;
    offsets_a[g] = static_cast<int64_t>(start) * K;
    offsets_d[g] = static_cast<int64_t>(start) * N;
    offsets_b[g] = static_cast<int64_t>(g) * KN;
}

struct GroupedTensorRAII {
    NVTEGroupedTensor handle;
    GroupedTensorRAII(NVTEScalingMode sm, size_t n, NVTEShape shape)
        : handle(nvte_create_grouped_tensor(sm, n, shape)) {}
    ~GroupedTensorRAII() { if (handle) nvte_destroy_grouped_tensor(handle); }
    operator NVTEGroupedTensor() const { return handle; }
    void set(NVTEGroupedTensorParam param, const NVTEBasicTensor& bt) {
        nvte_set_grouped_tensor_param(&handle, param, &bt);
    }
};

struct TensorRAII {
    NVTETensor handle;
    TensorRAII(NVTEScalingMode sm) : handle(nvte_create_tensor(sm)) {}
    ~TensorRAII() { if (handle) nvte_destroy_tensor(handle); }
    operator NVTETensor() const { return handle; }
    void set_data(void* dptr, NVTEDType dtype, NVTEShape shape) {
        NVTEBasicTensor bt{dptr, dtype, shape};
        nvte_set_tensor_param(&handle, kNVTERowwiseData, &bt);
    }
};

// Workspace sizes from TE source
static constexpr size_t kSetupAlignment = 256;
static constexpr size_t kCublasWorkspaceBytes = 32ULL * 1024 * 1024;  // 32 MiB

static size_t setup_workspace_bytes(size_t G) {
    size_t ptr_bytes = G * sizeof(void*);
    size_t int_bytes = G * sizeof(int);
    size_t total = 6 * ptr_bytes + 6 * int_bytes;
    return ((total + kSetupAlignment - 1) / kSetupAlignment) * kSetupAlignment;
}

// Persistent workspaces (allocated once, reused)
static torch::Tensor g_setup_ws;
static torch::Tensor g_cublas_ws;
static torch::Tensor g_alpha;
static torch::Tensor g_beta;
static torch::Tensor g_first_dims;
static torch::Tensor g_offsets_a;
static torch::Tensor g_offsets_d;
static torch::Tensor g_offsets_b;
static int64_t g_last_G = -1;

static void ensure_workspaces(int64_t G, torch::Device dev) {
    if (G == g_last_G) return;

    auto opts_byte = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(dev);

    g_setup_ws = torch::empty({(int64_t)setup_workspace_bytes(G)}, opts_byte);
    g_cublas_ws = torch::empty({(int64_t)kCublasWorkspaceBytes}, opts_byte);
    g_alpha = torch::ones({G}, opts_f32);
    g_beta = torch::zeros({G}, opts_f32);
    g_first_dims = torch::empty({G}, opts_i64);
    g_offsets_a = torch::empty({G}, opts_i64);
    g_offsets_d = torch::empty({G}, opts_i64);
    g_offsets_b = torch::empty({G}, opts_i64);
    g_last_G = G;
}

// Main function: grouped GEMM matching torch._grouped_mm(A, B, offs) semantics.
// A: [M_total, K] bf16, B: [G, K, N] bf16, offs: [G] int32 cumulative
// Returns: [M_total, N] bf16
torch::Tensor te_grouped_gemm_v2(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& offs
) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D [G, K, N]");
    TORCH_CHECK(offs.dim() == 1, "offs must be 1D [G]");
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be bf16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be bf16");
    TORCH_CHECK(offs.dtype() == torch::kInt32, "offs must be int32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && offs.is_contiguous());

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t G = B.size(0);
    const int64_t Kb = B.size(1);
    const int64_t N = B.size(2);
    TORCH_CHECK(K == Kb, "Contraction dim mismatch: A has K=", K, " but B has K=", Kb);

    auto D = torch::empty({M, N}, A.options());
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    ensure_workspaces(G, A.device());

    // Launch metadata kernel
    {
        int threads = 256;
        int blocks = (G + threads - 1) / threads;
        compute_grouped_metadata<<<blocks, threads, 0, stream>>>(
            offs.data_ptr<int32_t>(),
            g_first_dims.data_ptr<int64_t>(),
            g_offsets_a.data_ptr<int64_t>(),
            g_offsets_d.data_ptr<int64_t>(),
            g_offsets_b.data_ptr<int64_t>(),
            K, N, K * N, G
        );
    }

    auto make_shape = [](size_t a, size_t b) -> NVTEShape {
        NVTEShape s{};
        s.data[0] = a;
        s.data[1] = b;
        s.ndim = 2;
        return s;
    };
    auto make_shape1 = [](size_t a) -> NVTEShape {
        NVTEShape s{};
        s.data[0] = a;
        s.ndim = 1;
        return s;
    };

    NVTEShape shape_G = make_shape1(G);

    // Row-major trick: to get D_rm = A_rm @ B_rm, we call cuBLAS with
    // D_cm = B_cm @ A_cm (swapped operands, no transpose).
    // Col-major result [N, M_g] = row-major [M_g, N].

    // First cuBLAS operand = weight B [G, K, N], uniform shape
    NVTEShape b_logical = make_shape(G * K, N);
    GroupedTensorRAII gt_first(NVTE_DELAYED_TENSOR_SCALING, G, b_logical);
    {
        NVTEBasicTensor data_bt{B.data_ptr(), kNVTEBFloat16, b_logical};
        gt_first.set(kNVTEGroupedRowwiseData, data_bt);
    }

    // Second cuBLAS operand = activation A [M_total, K], varying first dim
    NVTEShape a_logical = make_shape(M, K);
    GroupedTensorRAII gt_second(NVTE_DELAYED_TENSOR_SCALING, G, a_logical);
    {
        NVTEBasicTensor data_bt{A.data_ptr(), kNVTEBFloat16, a_logical};
        gt_second.set(kNVTEGroupedRowwiseData, data_bt);
        NVTEBasicTensor fd_bt{g_first_dims.data_ptr(), kNVTEInt64, shape_G};
        gt_second.set(kNVTEGroupedFirstDims, fd_bt);
        NVTEBasicTensor off_bt{g_offsets_a.data_ptr(), kNVTEInt64, shape_G};
        gt_second.set(kNVTEGroupedTensorOffsets, off_bt);
    }

    // Output D [M_total, N] in row-major = [N, M_total] in col-major.
    // Set first_dim = N (uniform), last_dim = M_g (varying).
    NVTEShape d_logical = make_shape(N, M);
    GroupedTensorRAII gtD(NVTE_DELAYED_TENSOR_SCALING, G, d_logical);
    {
        NVTEBasicTensor data_bt{D.data_ptr(), kNVTEBFloat16, d_logical};
        gtD.set(kNVTEGroupedRowwiseData, data_bt);
        NVTEBasicTensor ld_bt{g_first_dims.data_ptr(), kNVTEInt64, shape_G};
        gtD.set(kNVTEGroupedLastDims, ld_bt);
        NVTEBasicTensor off_bt{g_offsets_d.data_ptr(), kNVTEInt64, shape_G};
        gtD.set(kNVTEGroupedTensorOffsets, off_bt);
    }

    // Alpha/beta tensors
    TensorRAII alpha_t(NVTE_DELAYED_TENSOR_SCALING);
    alpha_t.set_data(g_alpha.data_ptr(), kNVTEFloat32, shape_G);

    TensorRAII beta_t(NVTE_DELAYED_TENSOR_SCALING);
    beta_t.set_data(g_beta.data_ptr(), kNVTEFloat32, shape_G);

    // Workspace tensors
    TensorRAII ws_setup(NVTE_DELAYED_TENSOR_SCALING);
    ws_setup.set_data(g_setup_ws.data_ptr(), kNVTEByte,
                      make_shape1(g_setup_ws.numel()));

    TensorRAII ws_cublas(NVTE_DELAYED_TENSOR_SCALING);
    ws_cublas.set_data(g_cublas_ws.data_ptr(), kNVTEByte,
                       make_shape1(g_cublas_ws.numel()));

    // D_rm[M,N] = A_rm[M,K] @ B_rm[K,N]
    // Using row-major trick: pass B as first, A as second, no transpose.
    // cuBLAS computes D_cm[N,M] = B_cm[N,K] @ A_cm[K,M] → row-major [M,N]
    nvte_grouped_gemm(
        gt_first, /*transa=*/0,
        gt_second, /*transb=*/0,
        /*C=*/nullptr,
        gtD,
        alpha_t,
        beta_t,
        ws_setup,
        ws_cublas,
        /*config=*/nullptr,
        stream
    );

    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_gemm", &te_grouped_gemm_v2,
          "TE grouped GEMM via nvte_grouped_gemm (single cuBLASLt launch)",
          py::arg("A"), py::arg("B"), py::arg("offs"));
}
