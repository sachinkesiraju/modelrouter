"""Build oracle score bundles: score val/test rows on every registered base."""

from __future__ import annotations

import time
from typing import Any

from portallib import ChoiceExample

from .data import SuiteSplits
from .runtime import HFBackend


def build_score_bundle(
    specs: dict[str, tuple[str, str, float]],
    splits: SuiteSplits,
    tasks: list[str],
    *,
    capable: str = "capable",
    cheap: str = "cheap",
    dtype: str = "bfloat16",
    batch_size: int = 8,
    log=print,
) -> dict[str, Any]:
    """Score every val/test row on every base with the correct task LoRA.

    ``specs`` maps base name -> (hf model id, portal artifact path/id, relative cost).
    Returns the bundle dict consumed by ``sweep.run_policy_sweep``.
    """
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
        backend = HFBackend(model_id=model_id, artifact_id=artifact_id, dtype=dtype,
                            batch_size=batch_size)
        log(f"loading {name}: {model_id}")
        backend.load()
        if name == capable:
            for i, t in enumerate(backend.portal.config.tasks):
                task_latents[t] = backend.portal.task_latents[i].tolist()
        for split_name in ("val", "test"):
            for task in tasks:
                rows = [r for r in records[split_name] if r["task"] == task]
                examples = [ChoiceExample(task=task, prompt=r["prompt"],
                                          choices=tuple(r["choices"]), gold_idx=r["gold_idx"])
                            for r in rows]
                start = time.perf_counter()
                scores = backend.score_rows(examples, task)
                for r, s in zip(rows, scores):
                    r["scores"][name] = s
                log(f"{name} {split_name} {task}: {len(rows)} rows in "
                    f"{time.perf_counter() - start:.1f}s")
        backend.close()
        del backend

    return {
        "bases": {n: s[2] for n, s in specs.items()},
        "cheap": cheap,
        "capable": capable,
        "models": {n: s[0] for n, s in specs.items()},
        "val": records["val"],
        "test": records["test"],
        "task_latents": {t: task_latents[t] for t in tasks if t in task_latents},
    }
