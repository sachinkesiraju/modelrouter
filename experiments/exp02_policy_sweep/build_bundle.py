"""exp02 step 1: score val+test rows on both bases with the correct task LoRA.

Produces the score bundle consumed by the policy sweep.  Loads each base once
and scores all rows for all tasks (materializing each task LoRA once).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
from portallib import ChoiceDataset

from portal_dispatch.data import make_splits
from portal_dispatch.runtime import HFBackend

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cheap-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cheap-artifact", required=True, help="Path/id of the refit 0.6B artifact.")
    parser.add_argument("--capable-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--capable-artifact", default="RampPublic/portal-qwen3-1.7b")
    parser.add_argument("--val-per-task", type=int, default=15)
    parser.add_argument("--test-per-task", type=int, default=15)
    parser.add_argument("--tasks", default=None, help="Comma-separated subset of tasks.")
    parser.add_argument("--out", default=str(HERE / "results" / "scores_bundle.joblib"))
    args = parser.parse_args()

    suite = ChoiceDataset.from_hub("RampPublic/portallib-tasks")
    splits = make_splits(suite, val_per_task=args.val_per_task, test_per_task=args.test_per_task)
    tasks = args.tasks.split(",") if args.tasks else list(splits.tasks)

    specs = {
        "cheap": (args.cheap_model, args.cheap_artifact, 1.0),
        "capable": (args.capable_model, args.capable_artifact, 2.0),
    }
    records: dict[str, list[dict]] = {"val": [], "test": []}
    for split_name, split in (("val", splits.val), ("test", splits.test)):
        for task in tasks:
            for row in split[task]:
                records[split_name].append(
                    {"task": task, "prompt": row.prompt, "choices": list(row.choices),
                     "gold_idx": row.gold_idx, "scores": {}}
                )

    task_latents: dict[str, list[float]] = {}
    for name, (model_id, artifact_id, _cost) in specs.items():
        backend = HFBackend(model_id=model_id, artifact_id=artifact_id)
        print(f"loading {name}: {model_id}", flush=True)
        backend.load()
        if name == "capable":
            for i, t in enumerate(backend.portal.config.tasks):
                task_latents[t] = backend.portal.task_latents[i].tolist()
        for split_name in ("val", "test"):
            for task in tasks:
                rows = [r for r in records[split_name] if r["task"] == task]
                examples = [
                    type(splits.val[task][0])(task=task, prompt=r["prompt"],
                                              choices=tuple(r["choices"]), gold_idx=r["gold_idx"])
                    for r in rows
                ]
                start = time.perf_counter()
                scores = backend.score_rows(examples, task)
                for r, s in zip(rows, scores):
                    r["scores"][name] = s
                print(f"{name} {split_name} {task}: {len(rows)} rows in "
                      f"{time.perf_counter()-start:.1f}s", flush=True)
        backend.close()
        del backend

    bundle = {
        "bases": {"cheap": 1.0, "capable": 2.0},
        "cheap": "cheap",
        "capable": "capable",
        "models": {n: s[0] for n, s in specs.items()},
        "val": records["val"],
        "test": records["test"],
        "task_latents": {t: task_latents[t] for t in tasks if t in task_latents},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
