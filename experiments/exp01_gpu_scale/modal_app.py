"""exp01: GPU-scale validation on Modal.

Stages (each a separate billed job; artifacts checkpointed to a Modal volume):

  modal run experiments/exp01_gpu_scale/modal_app.py::smoke
  modal run experiments/exp01_gpu_scale/modal_app.py::refit --target Qwen/Qwen3-0.6B --tag refit-0.6b
  modal run experiments/exp01_gpu_scale/modal_app.py::refit --target Qwen/Qwen3-4B --tag refit-4b
  modal run experiments/exp01_gpu_scale/modal_app.py::score --spec-json '...'
  modal volume get modelrouter results/<name> ...
"""

from __future__ import annotations

import json

import modal

app = modal.App("modelrouter-gpu")

volume = modal.Volume.from_name("modelrouter", create_if_missing=True)
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.14.1",
        "portallib[training]==0.1.2",
        "scikit-learn>=1.4",
        "joblib>=1.3",
        "numpy>=1.26",
        "pyyaml>=6.0",
    )
    .env({"HF_HOME": f"{VOL}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("modelrouter")
)


def _load_splits(train_per_task: int, val_per_task: int, test_per_task: int):
    from portallib import ChoiceDataset

    from modelrouter.data import make_splits

    suite = ChoiceDataset.from_hub("RampPublic/portallib-tasks")
    return make_splits(suite, train_per_task=train_per_task, val_per_task=val_per_task,
                       test_per_task=test_per_task)


def _run_refit(source: str, target: str, tag: str, *, train_per_task: int, epochs: int,
               batch_size: int, dtype: str) -> dict:
    import torch
    from portallib import ChoiceDataset

    from modelrouter.refit import refit_artifact

    assert torch.cuda.is_available(), "no CUDA device in container"
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    splits = _load_splits(train_per_task, 15, 15)
    dataset = ChoiceDataset(
        train=[r for rows in splits.train.values() for r in rows],
        validation=[r for rows in splits.val.values() for r in rows],
    )
    history: list[dict] = []

    def on_epoch(metrics) -> None:
        record = {k: getattr(metrics, k) for k in dir(metrics) if not k.startswith("_")
                  and isinstance(getattr(metrics, k), (int, float))}
        history.append(record)
        print("epoch:", record, flush=True)
        volume.commit()

    out_dir = f"{VOL}/artifacts/{tag}"
    result = refit_artifact(source, target, dataset, epochs=epochs, batch_size=batch_size,
                            dtype=dtype, output_dir=out_dir, on_epoch=on_epoch)
    summary = {"source": source, "target": target, "tag": tag, "epochs": epochs,
               "train_per_task": train_per_task, "best_epoch": result.best_epoch,
               "history": history, "artifact": out_dir}
    with open(f"{VOL}/results/{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    volume.commit()
    return summary


@app.function(image=image, gpu="A10G", volumes={VOL: volume}, timeout=1800)
def smoke() -> dict:
    """Cheap end-to-end check of the exact refit code path on GPU: 2 tasks, 1 epoch."""
    import os

    os.makedirs(f"{VOL}/results", exist_ok=True)
    from portallib import ChoiceDataset

    from modelrouter.refit import refit_artifact

    import torch

    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    # The refitter iterates the source artifact's full task list, so every task
    # must be present — keep per-task counts tiny instead of subsetting tasks.
    splits = _load_splits(4, 4, 4)
    dataset = ChoiceDataset(
        train=[r for rows in splits.train.values() for r in rows],
        validation=[r for rows in splits.val.values() for r in rows],
    )
    history: list[dict] = []

    def on_epoch(metrics) -> None:
        record = {k: getattr(metrics, k) for k in dir(metrics) if not k.startswith("_")
                  and isinstance(getattr(metrics, k), (int, float))}
        history.append(record)
        print("epoch:", record, flush=True)

    refit_artifact("RampPublic/portal-qwen3-1.7b", "Qwen/Qwen3-0.6B", dataset, epochs=1,
                   batch_size=4, dtype="bfloat16", output_dir=f"{VOL}/artifacts/smoke",
                   on_epoch=on_epoch)
    volume.commit()
    return {"history": history}


@app.function(image=image, gpu="A100-80GB", volumes={VOL: volume}, timeout=4 * 3600)
def refit(source: str = "RampPublic/portal-qwen3-1.7b", target: str = "Qwen/Qwen3-0.6B",
          tag: str = "refit", train_per_task: int = 300, epochs: int = 3,
          batch_size: int = 8, dtype: str = "bfloat16") -> dict:
    import os

    os.makedirs(f"{VOL}/results", exist_ok=True)
    return _run_refit(source, target, tag, train_per_task=train_per_task, epochs=epochs,
                      batch_size=batch_size, dtype=dtype)


@app.function(image=image, gpu="A100-80GB", volumes={VOL: volume}, timeout=4 * 3600)
def score(spec_json: str, val_per_task: int = 100, test_per_task: int = 100,
          out_name: str = "scores_bundle_gpu.joblib", batch_size: int = 16) -> str:
    """Score all bases on val+test rows.  spec_json: {name: [model_id, artifact, cost]}."""
    import os

    import joblib
    import torch

    from modelrouter.scoring import build_score_bundle

    assert torch.cuda.is_available()
    os.makedirs(f"{VOL}/results", exist_ok=True)
    specs = {k: tuple(v) for k, v in json.loads(spec_json).items()}
    splits = _load_splits(1, val_per_task, test_per_task)
    bundle = build_score_bundle(specs, splits, list(splits.tasks), batch_size=batch_size)
    out = f"{VOL}/results/{out_name}"
    joblib.dump(bundle, out)
    volume.commit()
    print("wrote", out)
    return out
