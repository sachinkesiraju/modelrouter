"""exp01: refit the published Qwen3-1.7B PorTAL artifact to Qwen3-0.6B on CPU.

Kill criterion (gate G-A analogue at CPU scale): the refit must beat random
choice by >=10 pp macro accuracy after the final epoch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portallib import ChoiceDataset

from modelrouter.data import make_splits
from modelrouter.refit import refit_artifact

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="RampPublic/portal-qwen3-1.7b")
    parser.add_argument("--target", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train-per-task", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", default=str(HERE / "artifacts" / "refit-qwen3-0.6b"))
    args = parser.parse_args()

    suite = ChoiceDataset.from_hub("RampPublic/portallib-tasks")
    splits = make_splits(suite, train_per_task=args.train_per_task)
    dataset = ChoiceDataset(
        train=[r for rows in splits.train.values() for r in rows],
        validation=[r for rows in splits.val.values() for r in rows],
    )

    history = []

    def on_epoch(metrics) -> None:
        record = {k: getattr(metrics, k) for k in dir(metrics) if not k.startswith("_") and
                  isinstance(getattr(metrics, k), (int, float))}
        history.append(record)
        print("epoch:", record, flush=True)

    result = refit_artifact(
        args.source,
        args.target,
        dataset,
        epochs=args.epochs,
        output_dir=args.out,
        on_epoch=on_epoch,
    )
    out = {
        "source": args.source,
        "target": args.target,
        "best_epoch": result.best_epoch,
        "history": history,
        "diagnostics": getattr(result, "diagnostics", None) and str(result.diagnostics),
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "refit.json").write_text(json.dumps(out, indent=2, default=str))
    print("saved artifact to", args.out)


if __name__ == "__main__":
    main()
