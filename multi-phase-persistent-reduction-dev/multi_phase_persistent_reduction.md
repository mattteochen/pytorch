# Multi-Phase Persistent Reduction Fusion

## Current status

This feature is still experimental.

- It is opt-in and defaults off:
  `TORCHINDUCTOR_MULTI_PHASE_PERSISTENT_REDUCTION=1`
- The persistent-reduction trigger threshold is configurable:
  `TORCHINDUCTOR_PERSISTENT_REDUCTION_INNER_THRESHOLD`
- The default threshold is `1024`
- The current RMSNorm + fp8 quant path often needs `4096` to trigger, so the
  heuristic is not settled yet

## What it does

Fuses two Triton persistent reductions into one kernel when they operate on the
same data with different reduction granularities. The canonical example is
RMSNorm (full-row variance reduction) followed by per-group fp8 quantization
(per-group amax reduction).

Before:
```text
Kernel 1: load row -> sum(x^2) -> store variance + residual      [global memory]
Kernel 2: load row + variance -> norm -> reshape -> max/group
         -> scale/clamp/cast -> store                             [global memory]
```

After:
```text
Kernel 1: load row -> sum(x^2)                                    [registers]
         -> rsqrt, norm                                           [registers]
         -> reshape -> max/group                                  [registers]
         -> broadcast grouped result back to full row             [registers]
         -> scale/clamp/cast -> store full output + compact output
```

One kernel launch, one pass over the data, and no global-memory round-trip
between the two reductions.

## Files changed

| File | What |
|------|------|
| `torch/_inductor/config.py` | Feature flag, default-off behavior, configurable persistent threshold |
| `torch/_inductor/choices.py` | Uses configurable persistent threshold instead of a hardcoded value |
| `torch/_inductor/scheduler.py` | `PersistentMultiPhaseReduction` pattern matcher and supported grouped-reduction gate |
| `torch/_inductor/codegen/simd.py` | `codegen_multi_phase_reduction`, schedule generation, compact-phase codegen |
| `torch/_inductor/codegen/triton.py` | `_grouped_persistent_reduction`, compact-result cache plumbing |
| `torch/_inductor/codegen/cuda_combined_scheduling.py` | Dispatch to Triton scheduling |
| `test/inductor/test_multi_phase_reduction.py` | Correctness, kernel count, fallback, and compact-epilogue tests |

## Architecture

```text
Scheduler fusion loop
  |
  |- SIMDScheduling.can_fuse()
  |    `- PersistentMultiPhaseReduction.can_fuse()   <- pattern match
  |
  |- BaseScheduling.fuse()
  |    `- creates FusedMultiPhaseReductions(producer, consumer)
  |
  `- Scheduler._codegen()
       `- codegen_multi_phase_reduction()
            |- _generate_multi_phase_schedule()
            |- codegen_node_schedule_with_kernel()
            |- _codegen_compact_phase_into_kernel()  <- optional compact [x, g] epilogue
            `- codegen_kernel() + call_kernel()

Inside TritonKernel.reduction():
  if self.persistent_reduction and self.multi_phase_sub_groups:
      -> _grouped_persistent_reduction()
         reshape [XBLOCK, num_groups, group_size]
         reduce over group_size
         keep compact [XBLOCK, num_groups] result
         broadcast back to [XBLOCK, R0_BLOCK] for the main body
```

There are two important values after the grouped reduction:

- A broadcasted full-domain value used by the main fused kernel body
- A compact `[XBLOCK, num_groups]` value used only by compact epilogue stores

Those now use separate cache paths so full-domain consumers do not accidentally
see the compact tensor.

## Stable parts

### Fusion scope that is currently supported

`PersistentMultiPhaseReduction.can_fuse()` is intended for:

- Two persistent reductions on the same device
- A producer-consumer relationship between the reductions
- The same total logical element count
- A grouped second reduction that subdivides the first reduction

Grouped reduction fusion is currently limited to these reduction kinds:

- `any`
- `max`
- `min`
- `prod`
- `sum`

This is the supported surface today. Unsupported grouped reduction families
should fall back instead of being fused.

### Grouped reduction codegen

`_grouped_persistent_reduction()` emits the grouped reduction in registers:

```python
reshaped = tl.reshape(value, [XBLOCK, num_groups, group_size])
reduced = triton_fn(reshaped, 2)
expanded = tl.broadcast_to(reduced[:, :, None], [XBLOCK, num_groups, group_size])
result = tl.reshape(expanded, [XBLOCK, R0_BLOCK])
```

The important property is that the fused main body continues to see a
full-domain value, while the compact grouped result is also preserved for
compact outputs.

### Covered single-kernel behavior

The currently covered and tested path is:

- Full-row reduction feeding grouped reduction
- Full-domain continuation after the grouped reduction
- Compact per-group output emitted in the same kernel

The main example is RMSNorm + per-group fp8 quantization:

- Full-row variance reduction
- Grouped `amax`
- Broadcasted grouped scale used by the full quantization path
- Compact per-group scale stored from the same kernel

### Compact epilogue implementation

The old `_replay_pointwise_on_compact()` approach is gone.

Compact epilogues are now emitted with the regular codegen path:

- Add a temporary compact `y` range to the kernel
- Run `split_and_set_ranges()`
- Run ordinary `node.codegen()`

This is materially safer than replaying a hardcoded subset of FX ops by hand.

### Fallback behavior

If the pattern does not match, or if a node is outside the supported compact
surface, the compiler should fall back to separate kernels rather than forcing
the fusion.

That fallback behavior is now explicitly covered for grouped `argmax`.

## Not stable yet

### The feature is still default-off

This is deliberate. The implementation is useful, but the heuristic and the
supported surface are not broad enough yet to enable by default.

### Heuristic sensitivity

The trigger threshold is now configurable, but the tuning is still immature.

- Default: `1024`
- The motivating RMSNorm + fp8 quant case often uses `4096` in tests

So the feature can be correct and still not trigger unless the threshold is
raised for that workload.

### Compact-phase fusion is still narrow

The compact phase is handled by `_codegen_compact_phase_into_kernel()`, but it
is not a general downstream fusion mechanism.

Today it is only intended to fold true compact-domain pointwise nodes that:

- Are non-reduction nodes
- Have `nr == 1`
- Have total logical size `num_groups * numel`
- Read only from compact buffers already produced in the fused kernel

This works for compact `[M, groups]` style epilogues. It is not broadly proven
for:

- Mixed compact-domain and full-domain consumers
- More complex reshape or broadcast chains
- Non-contiguous compact output layouts
- Multi-output downstream graphs
- Arbitrary chains of downstream nodes with external inputs

### Unsupported grouped reduction families

These are not supported for grouped fusion today and should be treated as
fallback-only:

- `argmax`
- `argmin`
- `welford_reduce`
- `welford_combine`
- `online_softmax_reduce`

If support for them is added later, it needs real grouped implementations, not
just broader scheduler matching.

### Dynamic shapes

Fusion still relies on compile-time shape reasoning. If the compiler cannot
prove the grouped relationship and persistent size constraints statically, the
pattern will not fire.

### Interaction with other features

This path is not broadly validated with:

- `cooperative_reductions`
- `multi_kernel`
- `mix_order_reduction`
- `cpp_wrapper`
- split reductions

These combinations should be treated as unproven unless specifically tested.

## How to test

Enable the feature:

```bash
export TORCHINDUCTOR_MULTI_PHASE_PERSISTENT_REDUCTION=1
```

Raise the persistent threshold if needed:

```bash
export TORCHINDUCTOR_PERSISTENT_REDUCTION_INNER_THRESHOLD=4096
```

Run focused tests:

```bash
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python test/inductor/test_multi_phase_reduction.py \
  TestMultiPhaseReduction.test_rmsnorm_fp8_quant_fusion_correctness

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python test/inductor/test_multi_phase_reduction.py \
  TestMultiPhaseReduction.test_rmsnorm_fp8_quant_kernel_count

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python test/inductor/test_multi_phase_reduction.py \
  TestMultiPhaseReduction.test_grouped_argmax_falls_back

TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python test/inductor/test_multi_phase_reduction.py \
  TestMultiPhaseReduction.test_compact_epilogue_stays_in_one_kernel
```

Dump generated output code for correctness debugging:

```bash
TORCH_COMPILE_DEBUG=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
python test/inductor/test_multi_phase_reduction.py \
  TestMultiPhaseReduction.test_rmsnorm_fp8_quant_fusion_correctness
```

Disable the feature:

```bash
export TORCHINDUCTOR_MULTI_PHASE_PERSISTENT_REDUCTION=0
```

## What would make this more general

The next step is not "add more ad hoc compact cases".

The right direction is to make the compact group dimension a more first-class
part of the normal range machinery, so compact-domain code can be scheduled and
codegen'd with the same rules as the normal main-body path.

That would help with:

- broader compact epilogue coverage
- less special-case candidate filtering
- better handling of masks, layouts, and indexing
- fewer correctness risks from compact-only plumbing

Separately, grouped support for `argmax`/`argmin`/`welford`/softmax would need
real Triton implementations and dedicated tests.
