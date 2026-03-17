#!/usr/bin/env python3
"""
Fused GEMM + RoPE Triton kernel.

Performs packed QKV projection (addmm) and applies neox-style RoPE to Q and K
in a single Triton kernel, avoiding the DRAM round-trip between GEMM and RoPE.

The key insight: with BLOCK_N == head_dim, each GEMM output tile covers exactly
one head. After a single full-width dot product, we reshape the accumulator to
separate the two halves, apply RoPE in-register, then store directly to Q/K/V.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_addmm_rope_kernel(
    # GEMM inputs
    bias_ptr,
    A_ptr,          # [M, K]
    B_ptr,          # [N, K] row-major
    # RoPE inputs
    cos_sin_ptr,    # [max_pos, rotary_dim] f32
    positions_ptr,  # [M] i64
    # Outputs
    q_ptr,          # [M, q_size]
    k_ptr,          # [M, kv_size]
    v_ptr,          # [M, kv_size]
    # Dimensions
    M, N, K,
    q_size, kv_size,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    rotary_dim: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_start = pid_n * BLOCK_N

    offs_a_m = rm % M
    offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)

    A_base = A_ptr + offs_a_m[:, None] * stride_am
    B_base = B_ptr + offs_b_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k_idx * BLOCK_K + offs_k
        if EVEN_K:
            a = tl.load(A_base + k_off[None, :] * stride_ak)
            b = tl.load(B_base + k_off[:, None] * stride_bk)
        else:
            k_remaining = K - k_idx * BLOCK_K
            a = tl.load(A_base + k_off[None, :] * stride_ak,
                        mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(B_base + k_off[:, None] * stride_bk,
                        mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)

    # Rematerialize
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = rm < M
    mask_n = rn < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Add bias
    bias = tl.load(bias_ptr + rn, mask=mask_n, other=0.0).to(ACC_TYPE)
    acc = acc + bias[None, :]

    # Apply RoPE for Q and K heads (col_start < q_size + kv_size)
    needs_rope = col_start < (q_size + kv_size)
    if needs_rope:
        pos = tl.load(positions_ptr + rm, mask=mask_m, other=0)

        cos_offsets = pos[:, None] * rotary_dim + tl.arange(0, HALF_ROTARY)[None, :]
        sin_offsets = pos[:, None] * rotary_dim + (HALF_ROTARY + tl.arange(0, HALF_ROTARY))[None, :]
        cos = tl.load(cos_sin_ptr + cos_offsets, mask=mask_m[:, None], other=0.0)
        sin = tl.load(cos_sin_ptr + sin_offsets, mask=mask_m[:, None], other=0.0)

        # Split acc [BLOCK_M, BLOCK_N] into two halves of the head dim.
        # Reshape to [BLOCK_M, 2, HALF_ROTARY], transpose to [BLOCK_M, HALF_ROTARY, 2],
        # then tl.split on the last dim (size 2) to get x1, x2 each [BLOCK_M, HALF_ROTARY].
        acc_3d = tl.reshape(acc, (BLOCK_M, 2, HALF_ROTARY))
        acc_t = tl.permute(acc_3d, (0, 2, 1))  # [BLOCK_M, HALF_ROTARY, 2]
        x1, x2 = tl.split(acc_t)                # each [BLOCK_M, HALF_ROTARY]

        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin

        out_joined = tl.join(o1, o2)              # [BLOCK_M, HALF_ROTARY, 2]
        out_t = tl.permute(out_joined, (0, 2, 1)) # [BLOCK_M, 2, HALF_ROTARY]
        acc = tl.reshape(out_t, (BLOCK_M, BLOCK_N))

    result = acc.to(tl.bfloat16)

    is_q = col_start < q_size
    is_k = (col_start >= q_size) & (col_start < q_size + kv_size)

    if is_q:
        off = rm[:, None] * q_size + rn[None, :]
        tl.store(q_ptr + off, result, mask=mask)
    elif is_k:
        off = rm[:, None] * kv_size + (rn[None, :] - q_size)
        tl.store(k_ptr + off, result, mask=mask)
    else:
        off = rm[:, None] * kv_size + (rn[None, :] - q_size - kv_size)
        tl.store(v_ptr + off, result, mask=mask)


def fused_addmm_rope(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, K = hidden_states.shape
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    N = q_size + 2 * kv_size
    assert weight.shape == (N, K)
    assert bias.shape == (N,)
    assert head_dim == rotary_dim, "Partial rotation not yet supported"
    half_rotary = rotary_dim // 2

    q = torch.empty(M, q_size, device=hidden_states.device, dtype=hidden_states.dtype)
    k = torch.empty(M, kv_size, device=hidden_states.device, dtype=hidden_states.dtype)
    v = torch.empty(M, kv_size, device=hidden_states.device, dtype=hidden_states.dtype)

    BLOCK_M = 16
    BLOCK_N = head_dim
    BLOCK_K = 128
    GROUP_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    fused_addmm_rope_kernel[grid](
        bias, hidden_states, weight,
        cos_sin_cache, positions,
        q, k, v,
        M, N, K,
        q_size, kv_size,
        hidden_states.stride(0), hidden_states.stride(1),
        weight.stride(0), weight.stride(1),
        rotary_dim=rotary_dim,
        HALF_ROTARY=half_rotary,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        ACC_TYPE=tl.float32,
        ALLOW_TF32=True,
        EVEN_K=(K % BLOCK_K == 0),
        num_stages=5,
        num_warps=4,
    )

    return q, k, v
