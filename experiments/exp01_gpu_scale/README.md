# exp01_gpu_scale: the decisive GPU experiment (Modal, A100-80GB)

The P0 go/no-go run from `docs/roadmap.md`: does routing headroom survive at GPU scale
with properly-trained refits and a real model ladder?

## Setup

- Modal jobs (`modal_app.py`): refits + oracle scoring on A100-80GB, smoke on A10G;
  artifacts checkpointed to a Modal volume. Policy sweep runs locally on the bundle.
- Ladder: **cheap** = Qwen3-0.6B (refit, cost 1.0) / **mid** = Qwen3-1.7B (source artifact,
  cost 2.83) / **capable** = Qwen3-4B (refit, cost 6.67); costs ∝ parameter count.
- Refits from `RampPublic/portal-qwen3-1.7b` with 300 train/task (vs 20 in earlier CPU-scale pilots), 3 epochs,
  bf16, batch 8. Oracle: 14 tasks × up to 100 val + 100 test rows = **1230 val / 1230 test**.

## Refit results (macro val accuracy, epoch 0 = pre-refit baseline)

| Target | Baseline | Best refit | Δ |
|---|---:|---:|---:|
| Qwen3-0.6B | 48.1% | 56.7% | +8.6 pp |
| Qwen3-4B | 64.8% | 71.9% | +7.1 pp |

Both refits clearly beat their pre-refit baselines and random (~35–40%); the 4B refit shows
the alignment transfers *up* the ladder as well as down.

## Policy sweep — 3-tier (1230 test rows)

| Policy | Accuracy | Savings | Δ vs always-4B | % cheapest |
|---|---:|---:|---:|---:|
| always-cheap (0.6B) | 56.2% | 85.0% | −14.6 pp | 100% |
| always-capable (4B) | 70.8% | 0% | — | 0% |
| 3-tier oracle | **83.1%** | **59.2%** | **+12.3 pp** | — |
| score-floor 1.0 | 72.4% | 21.5% | **+1.6 pp** | 2% |
| score-floor 1.1 | 70.6% | 44.7% | +0.2 pp | 12% |
| **score-floor val-tuned (1.2)** | **68.0%** | **58.4%** | −2.8 pp | 23% |

- **Kill test (≥15% savings at ≤3 pp drop): PASSED decisively** — 58.4% savings at −2.8 pp
  (bootstrap CI: 56.9–59.6% savings). In earlier CPU-scale pilot runs the same criterion passed with 19%.
- Routing headroom *grew* with scale: the oracle beats always-capable by +12.3 pp while
  saving 59% — the tiers make complementary errors.
- **Task-latent `z` router: 100% leave-one-task-out** at predicting the best tier per task.

## Policy sweep — pairwise 1.7B vs 4B (drops the fragile 0.6B tier)

| Policy | Accuracy | Savings | Δ |
|---|---:|---:|---:|
| always-1.7B | 68.6% | 57.6% | −2.2 pp |
| score-floor 1.1 | 71.2% | 40.6% | −0.4 pp |
| **prompt-router δ=0.1 (prompt-only, live-deployable)** | **69.8%** | **47.0%** | **−1.1 pp** |

The headline production result: with a properly-trained 1.7B tier, the **prompt-only
router — the signal usable live, before any forward pass — now saves 47% within ~1 pp of
always-4B** (in the CPU-scale pilots it managed only 2–5%). Gap 1 (live routing signal) closes
at scale; the fragile 0.6B tier was a CPU-scale artifact, exactly as the roadmap
hypothesized.

## GPU-amortized cost model (re-priced sweep)

The tables above price tiers proportionally to parameter count (1.0 / 2.83 / 6.67). On a
dedicated GPU the real per-request cost is GPU-time: measured saturated serving throughput
per model on the same card, times the card's hourly price (`modelrouter.costs.GpuCostModel`).
From exp02's `load_bench` (vLLM, A10G, peak concurrent output tok/s):

| Tier | Measured tok/s (A10G) | GPU-amortized relative cost | Parameter-proportional |
|---|---:|---:|---:|
| 0.6B | 8707 | 1.00 | 1.00 |
| 1.7B | 4825 | 1.80 | 2.83 |
| 4B | 2345 | 3.71 | 6.67 |

Throughput scales sub-linearly with parameters, so the 4B tier is relatively cheaper than
parameter-proportional pricing assumes, and routing away from it saves less:

| Result | Parameter-proportional | GPU-amortized |
|---|---:|---:|
| score-floor val-tuned (1.2), 3-tier | 58.4% savings at −2.8 pp | **51.5% savings at −2.8 pp** (CI: 50.2–52.5%) |
| near-zero-loss floor 1.1, 3-tier | 44.7% at +0.2 pp | 39.6% at +0.2 pp |
| prompt-only router δ=0.1 (pairwise) | 47.0% at −1.1 pp | 42.0% at −1.1 pp |
| 3-tier oracle | 59.2% at +12.3 pp | 51.2% at +12.3 pp |
| kill test (≥15% savings at ≤3 pp drop) | passed | **still passes** |

Routing decisions are identical (the router does not see costs; only the savings
accounting changes). Assumptions: saturated GPU (peak-concurrency throughput), requests
priced by output tokens with prefill folded into measured end-to-end throughput, bare-base
throughput (vLLM's LoRA overhead is a roughly uniform multiplier across tiers and cancels
out of relative costs). Reproduce:

```bash
python experiments/exp01_gpu_scale/run_sweep.py \
  --gpu-costs experiments/exp02_vllm_bench/results/load_bench.json
```

## Cost

Smoke (A10G) + 2 refits + scoring (A100-80GB) ≈ 2.2 GPU-hours ≈ **$9** total.

The serving-substrate benchmark (vLLM adapter hot-swap; everything above uses the HF
backend) is in `experiments/exp02_vllm_bench/` — it also passed (15.4 ms swap = 2.2% of a
request).

## Reproduce

The policy sweep runs locally on the committed score bundles — no GPU needed:

```bash
python experiments/exp01_gpu_scale/run_sweep.py                 # 3-tier + pairwise tables
```

To regenerate everything from scratch on Modal (needs `modal token set ...`; ~$9):

```bash
# cheap 5-minute validity check before spending A100 time
modal run experiments/exp01_gpu_scale/modal_app.py::smoke

# refits (A100-80GB, ~40 min each)
modal run experiments/exp01_gpu_scale/modal_app.py::refit \
  --target Qwen/Qwen3-0.6B --tag refit-0.6b-gpu
modal run experiments/exp01_gpu_scale/modal_app.py::refit \
  --target Qwen/Qwen3-4B --tag refit-4b-gpu

# oracle scoring of all three tiers (A100-80GB, ~1 h)
modal run experiments/exp01_gpu_scale/modal_app.py::score --spec-json '{
  "cheap":   ["Qwen/Qwen3-0.6B", "/vol/artifacts/refit-0.6b-gpu", 1.0],
  "mid":     ["Qwen/Qwen3-1.7B", "RampPublic/portal-qwen3-1.7b", 2.83],
  "capable": ["Qwen/Qwen3-4B",   "/vol/artifacts/refit-4b-gpu",  6.67]}'

# then download /vol/results/scores_bundle_gpu.joblib into results/ and run run_sweep.py
modal volume get modelrouter results/scores_bundle_gpu.joblib \
  experiments/exp01_gpu_scale/results/scores_bundle_gpu.joblib
```

## Extending the ladder past 4B (prepared, not run)

`refit` and `score` take arbitrary targets; adding a Qwen3-8B top tier is two commands
(~1.5 A100-hours, ~$6) plus a `load_bench` run for its measured cost:

```bash
modal run experiments/exp01_gpu_scale/modal_app.py::refit \
  --target Qwen/Qwen3-8B --tag refit-8b-gpu
modal run experiments/exp01_gpu_scale/modal_app.py::score --spec-json '{
  "cheap":   ["Qwen/Qwen3-0.6B", "/vol/artifacts/refit-0.6b-gpu", 1.0],
  "mid":     ["Qwen/Qwen3-1.7B", "RampPublic/portal-qwen3-1.7b", 2.83],
  "capable": ["Qwen/Qwen3-4B",   "/vol/artifacts/refit-4b-gpu",  6.67],
  "top":     ["Qwen/Qwen3-8B",   "/vol/artifacts/refit-8b-gpu",  13.3]}' \
  --out-name scores_bundle_8b.joblib
```

No 8B results are reported here because these commands have not been run.
