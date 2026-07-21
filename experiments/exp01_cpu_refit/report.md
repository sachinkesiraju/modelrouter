# exp01_cpu_refit: refit the published 1.7B artifact to Qwen3-0.6B on CPU

## Setup

- Source: `RampPublic/portal-qwen3-1.7b` (public PorTAL artifact: core + task latents + 1.7B alignment).
- Target: `Qwen/Qwen3-0.6B` in bfloat16 with gradient checkpointing (fits the 8 GB CPU box; a
  float32 run without checkpointing was OOM-killed at ~7.3 GB RSS).
- Data: 14 tasks from `RampPublic/portallib-tasks`, 20 train examples/task, deterministic seed-0
  splits (`portal_dispatch.data.make_splits`), 3 epochs, batch size 2.

## Results

| Epoch | Macro accuracy (val) | Macro gold NLL |
|---|---:|---:|
| 0 (pre-refit baseline) | 48.1% | 3.30 |
| 1 | 54.3% | 2.26 |
| 2 (best) | **58.1%** | 2.09 |
| 3 | 52.9% | 2.59 |

## Gate check (G-A analogue)

The refit must beat the pre-refit baseline / random by ≥10 pp macro accuracy:
- vs pre-refit baseline: **+10.0 pp** (48.1% → 58.1%) — passes.
- vs random choice (~35–40% macro over the 14-task mix): +18–23 pp — passes clearly.

Epoch 3 regresses (small-data overfitting at 20 examples/task); the saved artifact is the
best epoch. This matches the Phase 1 observation that CPU-scale refits are a lower bound —
the GPU plan uses 300 examples/task.
