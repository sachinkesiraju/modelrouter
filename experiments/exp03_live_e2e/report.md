# exp03_live_e2e: live pipeline with a task classifier (the task-classifier tax)

## Setup

- Full live path: prompt → `TaskClassifier` (MiniLM logistic, trained on 20 train
  prompts/task across all 14 tasks) → task LoRA materialization → inference.
- Tasks evaluated: `rte`, `cb`, `copa`, `wsc` × 8 test examples (32 rows).
- Two label modes: **oracle** (gold task label; the right LoRA is always materialized) vs
  **classified** (predicted label; a wrong task LoRA is materialized on mistakes).

## Results

Task classifier accuracy on test prompts: **81.2%**.

| Policy | Accuracy | Savings | Δ vs always-capable |
|---|---:|---:|---:|
| oracle / always-cheap | 65.6% | 50.0% | −18.8 pp |
| oracle / always-capable | 84.4% | 0.0% | — |
| oracle / oracle-route | 90.6% | 37.5% | +6.2 pp |
| classified / always-cheap | 65.6% | 50.0% | −15.6 pp |
| classified / always-capable | 81.2% | 0.0% | — |
| classified / oracle-route | 87.5% | 39.1% | +6.2 pp |

## What this means

1. The live loop (classify → materialize → infer) works end-to-end; adapter
   materialization is a per-task cached ~50 ms cost.
2. The task-classifier tax on this 4-task slice is ~3 pp on the capable base
   (84.4% → 81.2%) at 81% classifier accuracy. On these four tasks the cheap base
   happened to be equally accurate under both label modes (65.6%), i.e. the wrong-LoRA
   penalty did not bind here — but the slice is tiny (32 rows) and Phase 1 measured a much
   larger collapse on the full 14-task suite at 75% classifier accuracy.
3. Product implication (unchanged): ship **known-skill routes** first; in task-agnostic
   mode, abstain-to-capable on low classifier confidence so mistakes cost money, not quality.

## Reproduce

```bash
python experiments/exp03_live_e2e/run.py \
  --cheap-artifact experiments/exp01_cpu_refit/artifacts/refit-qwen3-0.6b
```
