"""Continuous evaluation loop: retrain routers from production traces.

Traces are the training data (Ramp-style): each request logs prompt, chosen
model, and — once a grader or user feedback labels it — an ``outcome`` (0/1).
``retrain_from_traces`` rebuilds per-model correctness labels and refits the
prompt router, saving a versioned artifact so weights can be canaried and
rolled back.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .routing import PromptEmbeddingRouter
from .tracing import TraceJournal


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def label_traces(journal: TraceJournal, grader: Callable[[dict[str, Any]], int | None]) -> list[dict[str, Any]]:
    """Attach outcome labels to unlabeled traces using a grader function.

    The grader receives one trace record and returns 1 (correct), 0 (incorrect),
    or None (cannot judge).  Already-labeled traces pass through unchanged.
    """
    labeled = []
    for record in journal.read():
        if "outcome" not in record:
            outcome = grader(record)
            if outcome is None:
                continue
            record = {**record, "outcome": int(outcome)}
        labeled.append(record)
    return labeled


def retrain_from_traces(
    traces: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    embed: Callable[[list[str]], np.ndarray],
    min_examples_per_model: int = 20,
) -> dict[str, Any]:
    """Fit a fresh PromptEmbeddingRouter on labeled traces and save a versioned artifact.

    Returns metadata including per-model example counts and the artifact path.
    """
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in traces:
        served = record.get("served")
        if served and "outcome" in record and record.get("prompt"):
            by_model.setdefault(served, []).append(record)
    usable = {m: rows for m, rows in by_model.items() if len(rows) >= min_examples_per_model}
    if not usable:
        raise ValueError(
            f"not enough labeled traces (need >= {min_examples_per_model} per model, "
            f"got {({m: len(r) for m, r in by_model.items()})})"
        )

    router = PromptEmbeddingRouter()
    counts: dict[str, int] = {}
    for model_name, rows in usable.items():
        embeddings = embed([r["prompt"] for r in rows])
        labels = {model_name: np.array([int(r["outcome"]) for r in rows])}
        partial = PromptEmbeddingRouter()
        partial.fit(embeddings, labels)
        router.models[model_name] = partial.models[model_name]
        counts[model_name] = len(rows)

    version = time.strftime("%Y%m%d-%H%M%S")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / f"prompt_router_{version}.joblib"
    router.save(str(artifact))
    metadata = {
        "version": version,
        "git_sha": _git_sha(),
        "examples_per_model": counts,
        "artifact": str(artifact),
    }
    (out / f"prompt_router_{version}.json").write_text(json.dumps(metadata, indent=2))
    latest = out / "latest.json"
    latest.write_text(json.dumps(metadata, indent=2))
    return metadata
