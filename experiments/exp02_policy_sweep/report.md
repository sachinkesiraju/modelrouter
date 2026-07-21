# exp02_policy_sweep: routers + dispatch policies on 14 tasks (fresh CPU run)

## Setup

- Bases: cheap = Qwen3-0.6B with the exp01 refit artifact (cost 1.0); capable = Qwen3-1.7B
  with the published source artifact (cost 2.0). Max possible savings = 50%.
- Data: 14 tasks × 15 val + 15 test examples (210 val / 210 test rows), seed-0 splits,
  correct (oracle) task labels — this experiment isolates the *base-dispatch* decision.
- Both bases scored every val+test row with the task LoRA materialized at runtime
  (`build_bundle.py`); routers trained on val only, policies evaluated on test only (`run.py`).

## Results (210 test rows)

| Policy | Accuracy | Savings | Δ vs always-capable | % routed cheap |
|---|---:|---:|---:|---:|
| always-cheap | 53.8% | 50.0% | −14.3 pp | 100% |
| always-capable | 68.1% | 0.0% | — | 0% |
| oracle per-query routing | 73.8% | 40.0% | **+5.7 pp** | 80% |
| score-floor 1.0 | 67.6% | 7.6% | −0.5 pp | 15% |
| score-floor 1.1 | 67.6% | 13.1% | −0.5 pp | 26% |
| **score-floor val-tuned (1.2)** | **65.2%** | **19.0%** | **−2.9 pp** | 38% |
| score-floor 1.5 | 55.7% | 44.0% | −12.4 pp | 88% |
| prompt-router δ=0.0 | 65.7% | 4.5% | −2.4 pp | 9% |
| prompt-router δ=0.1 | 61.4% | 16.4% | −6.7 pp | 33% |
| cascade 0.5 | 68.1% | −44.3% | 0.0 pp | 6% |

- **Kill test (≥15% savings at ≤3 pp drop): PASSED** by the val-tuned score-floor policy —
  19.0% savings at a 2.9 pp drop. Bootstrap 95% CI (200 resamples): accuracy 59.0–71.4%,
  savings **16.2–22.6%**.
- The floor was selected on validation only (max savings s.t. ≤3 pp val drop), then applied
  to the held-out test set — no test-set tuning.
- **Oracle headroom is large**: +5.7 pp accuracy at 40% savings, because the two bases make
  different errors. Routing headroom exists; the router is the gap.
- **Task-latent `z` router: 78.6% leave-one-task-out accuracy** at predicting whether the
  cheap base is at least as good as the capable base for an unseen task — reproducing the
  Phase 1 number exactly and supporting zero-shot task routing.
- The prompt-only router is safe but conservative (2–5% savings within the quality bar),
  consistent with Phase 1: 210 training prompts are not enough to be aggressive from the
  prompt alone.
- Cascade shows *negative* savings here because escalation pays for both models and the
  cheap tier is only 2× cheaper; cascades need a wider cost spread (or a better keep-rate)
  to win — reported honestly, cost accounting includes the double inference.

## What this validates

1. The full dispatch loop works end-to-end from public artifacts on a CPU box: refit →
   score → route → policy → kill-criteria gate.
2. Score-router floor dispatch clears the pre-registered Go/No-Go bar at CPU scale.
3. The remaining gaps (prompt-router aggressiveness, cascade economics, task-agnostic
   robustness — see exp03) are the exact GPU-scale questions in `docs/roadmap.md`.

## Reproduce

```bash
python experiments/exp02_policy_sweep/build_bundle.py \
  --cheap-artifact experiments/exp01_cpu_refit/artifacts/refit-qwen3-0.6b
python experiments/exp02_policy_sweep/run.py
```
