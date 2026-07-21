"""exp03: live end-to-end — task classifier -> route -> materialize -> infer.

Measures the task-classifier tax: the same dispatch policy evaluated with
oracle task labels vs classifier-predicted task labels (the wrong task LoRA is
materialized on classifier mistakes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from portallib import ChoiceDataset, ChoiceExample

from portal_dispatch.data import make_splits
from portal_dispatch.routing import TaskClassifier
from portal_dispatch.runtime import HFBackend
from portal_dispatch.eval import policy_stats
from portal_dispatch.tracing import TraceJournal

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cheap-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cheap-artifact", required=True)
    parser.add_argument("--capable-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--capable-artifact", default="RampPublic/portal-qwen3-1.7b")
    parser.add_argument("--tasks", default="rte,cb,copa,wsc")
    parser.add_argument("--test-per-task", type=int, default=8)
    args = parser.parse_args()

    tasks = args.tasks.split(",")
    suite = ChoiceDataset.from_hub("RampPublic/portallib-tasks")
    splits = make_splits(suite, val_per_task=15, test_per_task=args.test_per_task)

    # Task classifier trained on train prompts of all 14 tasks.
    clf = TaskClassifier()
    train_prompts = [r.prompt for t in splits.tasks for r in splits.train[t]]
    train_tasks = [t for t in splits.tasks for _ in splits.train[t]]
    clf.fit(train_prompts, train_tasks)

    test_rows = [(t, r) for t in tasks for r in splits.test[t]]
    prompts = [r.prompt for _, r in test_rows]
    gold_tasks = [t for t, _ in test_rows]
    pred_tasks = clf.predict(prompts)
    clf_acc = float(np.mean([p == g for p, g in zip(pred_tasks, gold_tasks)]))
    print(f"task classifier accuracy on test: {clf_acc:.3f}", flush=True)

    journal = TraceJournal(HERE / "results" / "traces.jsonl")
    (HERE / "results").mkdir(exist_ok=True)

    results: dict[str, dict] = {}
    for name, (model_id, artifact_id) in {
        "cheap": (args.cheap_model, args.cheap_artifact),
        "capable": (args.capable_model, args.capable_artifact),
    }.items():
        backend = HFBackend(model_id=model_id, artifact_id=artifact_id)
        print("loading", model_id, flush=True)
        backend.load()
        for mode, task_labels in (("oracle", gold_tasks), ("classified", pred_tasks)):
            correct = []
            for (gold_task, row), label in zip(test_rows, task_labels):
                example = ChoiceExample(task=label, prompt=row.prompt, choices=row.choices,
                                        gold_idx=row.gold_idx)
                scores = backend.score_rows([example], label)[0]
                correct.append(int(int(np.argmax(scores)) == row.gold_idx))
            results.setdefault(mode, {})[name] = np.array(correct)
            print(f"{name} {mode}: acc={np.mean(correct):.3f}", flush=True)
        backend.close()
        del backend

    # Canonical baselines under each label mode quantify the task-classifier tax.
    out: dict[str, object] = {"task_classifier_accuracy": clf_acc, "n_test": len(test_rows)}
    costs = {"cheap": 1.0, "capable": 2.0}
    for mode in ("oracle", "classified"):
        correct = results[mode]
        n = len(test_rows)
        stats = [
            policy_stats(f"{mode}_always_cheap", np.array(["cheap"] * n), correct, costs, "capable"),
            policy_stats(f"{mode}_always_capable", np.array(["capable"] * n), correct, costs, "capable"),
        ]
        oracle_route = np.array(
            ["cheap" if correct["cheap"][i] >= correct["capable"][i] else "capable" for i in range(n)]
        )
        stats.append(policy_stats(f"{mode}_oracle_route", oracle_route, correct, costs, "capable"))
        out[mode] = [s.to_dict() for s in stats]
        for s in stats:
            print(f"{s.name:28s} acc={s.accuracy:.3f} savings={s.savings:.3f} "
                  f"drop={s.drop_vs_capable:+.3f}", flush=True)
        journal.emit(mode=mode, stats=[s.to_dict() for s in stats])

    (HERE / "results" / "results.json").write_text(json.dumps(out, indent=2))
    print("wrote", HERE / "results" / "results.json")


if __name__ == "__main__":
    main()
