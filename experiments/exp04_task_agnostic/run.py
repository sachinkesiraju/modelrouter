"""exp04: end-to-end validation of task-agnostic mode on the exp01 score bundle.

Measures the production task-agnostic pipeline (TaskClassifier on the prompt ->
per-task tier choice -> abstain-to-capable below a confidence threshold) against
known-task routing (true task label -> per-task tier chosen on val), on the same
1230 held-out test rows as exp01. Runs locally on CPU; no GPU needed.

  python experiments/exp04_task_agnostic/run.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from modelrouter.routing import TaskClassifier

HERE = Path(__file__).parent
BUNDLE = HERE.parent / "exp01_gpu_scale" / "results" / "scores_bundle_gpu.joblib"


def _correct(rows: list[dict], base: str) -> np.ndarray:
    return np.array([int(np.argmax(r["scores"][base]) == r["gold_idx"]) for r in rows])


def _stats(chosen: list[str], test: list[dict], costs: dict[str, float]) -> dict:
    correct = np.array([int(np.argmax(t["scores"][c]) == t["gold_idx"]) for c, t in zip(chosen, test)])
    cost = sum(costs[c] for c in chosen)
    cap_cost = costs["capable"] * len(test)
    return {"accuracy": float(correct.mean()), "savings": float(1 - cost / cap_cost)}


def main() -> None:
    bundle = joblib.load(BUNDLE)
    costs: dict[str, float] = bundle["bases"]
    val, test = bundle["val"], bundle["test"]

    # Per-task best tier on val: cheapest tier within 1 pp of the task's best tier.
    tiers = sorted(costs, key=costs.get)
    task_tier: dict[str, str] = {}
    for task in sorted({r["task"] for r in val}):
        rows_t = [r for r in val if r["task"] == task]
        accs = {b: float(_correct(rows_t, b).mean()) for b in tiers}
        best = max(accs.values())
        task_tier[task] = next(b for b in tiers if accs[b] >= best - 0.01)

    # Known-task routing: true test label -> that task's tier.
    known = [task_tier[r["task"]] for r in test]
    known_stats = _stats(known, test, costs)

    # Task-agnostic pipeline: classifier prediction + abstain-to-capable.
    clf = TaskClassifier()
    clf.fit([r["prompt"] for r in val], [r["task"] for r in val])
    preds = clf.predict([r["prompt"] for r in test])
    confs = clf.confidence([r["prompt"] for r in test])
    clf_acc = float(np.mean([p == r["task"] for p, r in zip(preds, test)]))

    agnostic = {}
    for thr in (0.0, 0.3, 0.5, 0.55, 0.7, 0.9):
        chosen = [task_tier[p] if c >= thr else "capable" for p, c in zip(preds, confs)]
        agnostic[str(thr)] = {
            **_stats(chosen, test, costs),
            "abstain_frac": float(np.mean([c < thr for c in confs])),
        }

    always_capable = _stats(["capable"] * len(test), test, costs)
    result = {
        "n_test": len(test),
        "costs": costs,
        "task_classifier_accuracy": clf_acc,
        "task_tier_map": task_tier,
        "always_capable": always_capable,
        "known_task_routing": known_stats,
        "task_agnostic_by_abstain_threshold": agnostic,
    }
    (HERE / "results").mkdir(exist_ok=True)
    with open(HERE / "results" / "task_agnostic.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
