# exp04_task_agnostic: end-to-end validation of task-agnostic mode

Measures the production task-agnostic pipeline (prompt -> `TaskClassifier` ->
per-task tier -> abstain-to-capable below a confidence threshold) against
known-task routing on the same 1230 held-out exp01 test rows and ladder costs
(cheap 1.0 / mid 2.83 / capable 6.67). Runs locally on CPU from the committed
exp01 score bundle.

## Results (1230 test rows)

- **Task classifier accuracy: 73.7%** (14-way, MiniLM embeddings + logistic
  regression trained on the 1230 val prompts).
- Per-task tier map fit on val (cheapest tier within 1 pp of the task's best):
  8 tasks -> mid, 6 tasks -> capable, 0 tasks -> cheap.

| Pipeline | Accuracy | Savings vs always-4B |
|---|---:|---:|
| always-capable (4B) | 70.8% | 0% |
| known-task routing (true label -> task's tier) | 71.1% | 31.8% |
| **task-agnostic, no abstain (thr 0.0)** | **71.3%** | **31.8%** |
| task-agnostic, abstain thr 0.3 (26% abstain) | 71.5% | 22.6% |
| task-agnostic, abstain thr 0.55 (72% abstain) | 71.5% | 6.7% |

## Findings

- **The classification tax on accuracy is zero.** At 73.7% task accuracy the
  pipeline matches known-task routing on both savings (31.8%) and accuracy
  (71.3% vs 71.1%): misclassifications land on tasks with similar tier needs,
  so the routed tier is usually still right.
- **The abstain guard is where the savings tax lives.** The default
  `abstain_below: 0.55` from `configs/routes.example.yaml` abstains on 72% of
  prompts and collapses savings from 31.8% to 6.7% for a +0.2 pp accuracy gain.
  On in-distribution traffic a much lower threshold (or none) is the right
  setting; the guard's value is for out-of-distribution prompts, which this
  benchmark cannot measure.
- Known-task tier routing itself (31.8% savings) is weaker than the per-row
  score-floor router from exp01 (58.4%): whole-task tier assignment leaves the
  within-task routing headroom on the table. Task-agnostic mode inherits that
  ceiling.

## Reproduce

```bash
python experiments/exp04_task_agnostic/run.py   # writes results/task_agnostic.json
```
