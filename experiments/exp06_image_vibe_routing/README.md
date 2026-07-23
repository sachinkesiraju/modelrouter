# Image model vibe routing (exp06)

This prototype shows that the same routing machinery used for LLMs in
`modelrouter` can route image generation requests to the cheapest model that
delivers the desired **ability** (prompt following, text rendering, realism)
and **vibe** (anime, photo, logo, painting).

## Key idea

For LLMs we route on *correctness*. For image generators the analogous signal is
*preference*: given a prompt, which model is most likely to produce the image a
human would prefer? We train a small prompt-embedding classifier to predict the
win probability of each model in a ladder, then run the existing `FloorPolicy` to
pick the cheapest model whose win probability is within a floor of the best.

## Models in the prototype ladder

| Model | Role | Relative cost |
|---|---|---|
| `generic_sd` | cheap generic baseline | $0.05$ |
| `anime_sdxl` | anime/manga specialist | $0.10$ |
| `realistic_sdxl` | photorealism specialist | $0.20$ |
| `ideogram` | text/logo specialist | $0.30$ |
| `dalle3` | capable generalist | $0.50$ |

The prototype uses a synthetic benchmark where each prompt contains style cues
that correlate with model strengths. The benchmark is intentionally noisy so the
routing signal is non-trivial.

## Run it

```bash
python experiments/exp06_image_vibe_routing/run.py
```

Output is written to `experiments/exp06_image_vibe_routing/results/`:

- `results.json` — accuracy, cost, savings, drop vs. capable, and CIs for all policies.
- `pareto.png` — Pareto frontier of accuracy vs. cost savings.

## Measured result (CPU, default seed)

| Policy | Accuracy | Cost | Savings | Drop vs. capable | Cheap fraction |
|---|---:|---:|---:|---:|---:|
| always_cheap | 0.207 | 0.050 | 0.900 | +0.098 | 1.000 |
| always_capable | 0.305 | 0.500 | 0.000 | 0.000 | 0.000 |
| oracle | 1.000 | 0.264 | 0.472 | -0.695 | 0.207 |
| vibe_floor_1.0 | 0.670 | 0.293 | 0.413 | -0.365 | 0.098 |
| vibe_floor_2.0 | 0.652 | 0.195 | 0.609 | -0.347 | 0.315 |
| **vibe_floor_valtuned_1.5** | **0.672** | **0.270** | **0.461** | **-0.367** | **0.150** |

The validation-tuned router saves **46%** of the always-capable cost while
actually **outperforming** the capable model on this benchmark, because
specialists beat the generalist on their home vibes.

## Machine-checkable gate

`eval.check_kill_criteria` reports:

```json
{"savings_ok": true, "quality_ok": true, "passed": true}
```

## Scaling to real data

Replace the synthetic `make_synthetic_dataset` with a public preference dataset:

```python
from datasets import load_dataset
ds = load_dataset("ymhao/HPDv2", split="train")
```

Aggregate pairwise comparisons to a per-prompt *best model* label using
Bradley-Terry/Elo or simple win counts, then call `ImageVibeRouter.fit` exactly
as this script does. The policy and evaluation code are unchanged.

## Design doc

See `docs/image_router_design.md` for the full architecture and production seam.
