"""Policy sweep over a precomputed score bundle.

Bundle format (joblib dict):
  bases: {name: cost}
  capable / cheap: base names
  val / test: list of {task, prompt, gold_idx, scores: {base: [per-choice float]}}
  task_latents: optional {task: vector} for the latent router
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np

from .dispatch import BaseSpec, CascadePolicy, FloorPolicy
from .eval import PolicyStats, bootstrap_ci, check_kill_criteria, policy_stats
from .routing import LatentRouter, PromptEmbeddingRouter, ScoreRouter, score_features


def _correct(rows: list[dict[str, Any]], base: str) -> np.ndarray:
    return np.array([int(np.argmax(r["scores"][base]) == r["gold_idx"]) for r in rows])


def _features(rows: list[dict[str, Any]], base: str) -> np.ndarray:
    return np.stack([score_features(r["scores"][base]) for r in rows])


def _embed(prompts: list[str], encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    return np.asarray(SentenceTransformer(encoder_name).encode(prompts, show_progress_bar=False))


def run_policy_sweep(bundle_path: str, out_path: str, *, with_prompt_router: bool = True) -> dict[str, Any]:
    bundle = joblib.load(bundle_path)
    cheap, capable = bundle["cheap"], bundle["capable"]
    costs: dict[str, float] = bundle["bases"]
    bases = [BaseSpec(name=n, cost=c) for n, c in costs.items()]
    val, test = bundle["val"], bundle["test"]
    n = len(test)

    correct = {b: _correct(test, b) for b in costs}
    results: list[PolicyStats] = []

    # Fixed baselines.
    results.append(policy_stats("always_cheap", np.array([cheap] * n), correct, costs, capable))
    results.append(policy_stats("always_capable", np.array([capable] * n), correct, costs, capable))
    oracle = np.array([cheap if correct[cheap][i] >= correct[capable][i] else capable for i in range(n)])
    results.append(policy_stats("oracle", oracle, correct, costs, capable))

    # Score router + floor sweep.
    router = ScoreRouter()
    router.fit(
        {b: _features(val, b) for b in costs},
        {b: _correct(val, b) for b in costs},
    )
    probs_test = router.predict_proba({b: _features(test, b) for b in costs})
    floor_rows = {}
    for floor in (1.0, 1.1, 1.2, 1.5, 2.0):
        policy = FloorPolicy(floor=floor)
        chosen = np.array(
            [policy.decide({b: probs_test[b][i] for b in costs}, bases).chosen for i in range(n)]
        )
        stats = policy_stats(f"score_floor_{floor}", chosen, correct, costs, capable)
        results.append(stats)
        floor_rows[floor] = (chosen, stats)

    # Cascade: cheap-base score margin as confidence; escalation adds cheap cost to escalated rows.
    margins = np.array([score_features(r["scores"][cheap])[2] for r in test])
    conf = (margins - margins.min()) / (margins.ptp() + 1e-9)
    for thr in (0.3, 0.5):
        policy = CascadePolicy(threshold=thr)
        chosen = np.array([policy.decide(float(conf[i]), bases).chosen for i in range(n)])
        extra = np.array([costs[cheap] if c == capable else 0.0 for c in chosen])
        results.append(policy_stats(f"cascade_{thr}", chosen, correct, costs, capable, extra_cost=extra))

    # Prompt-embedding router + delta sweep.
    prompt_router_diag = {}
    if with_prompt_router:
        emb_val = _embed([r["prompt"] for r in val])
        emb_test = _embed([r["prompt"] for r in test])
        prouter = PromptEmbeddingRouter()
        prouter.fit(emb_val, {b: _correct(val, b) for b in costs})
        p = prouter.predict_proba(emb_test)
        for delta in (-0.1, -0.05, 0.0, 0.1):
            chosen = np.array([cheap if p[cheap][i] + delta >= p[capable][i] else capable for i in range(n)])
            results.append(policy_stats(f"prompt_delta_{delta}", chosen, correct, costs, capable))
        prompt_router_diag = {
            "router": prouter,
            "test_probs": {b: p[b].tolist() for b in costs},
        }

    # Task-latent router: LOO accuracy at predicting cheap-is-good-enough per task.
    latent_loo = None
    if bundle.get("task_latents"):
        tasks = sorted(bundle["task_latents"].keys())
        latents = np.stack([np.asarray(bundle["task_latents"][t]) for t in tasks])
        labels = []
        for t in tasks:
            rows_t = [r for r in test if r["task"] == t]
            labels.append(int(np.mean(_correct(rows_t, cheap)) >= np.mean(_correct(rows_t, capable))))
        latent_loo = LatentRouter.loo_accuracy(latents, np.array(labels))

    # Pick the best floor by validation (max savings subject to <=3 pp drop on val).
    correct_val = {b: _correct(val, b) for b in costs}
    probs_val = router.predict_proba({b: _features(val, b) for b in costs})
    best_floor, best_savings = 1.0, -1.0
    for floor in (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5):
        policy = FloorPolicy(floor=floor)
        chosen_v = np.array(
            [policy.decide({b: probs_val[b][i] for b in costs}, bases).chosen for i in range(len(val))]
        )
        s = policy_stats(f"val_floor_{floor}", chosen_v, correct_val, costs, capable)
        if s.drop_vs_capable <= 0.03 and s.savings > best_savings:
            best_floor, best_savings = floor, s.savings
    tuned_policy = FloorPolicy(floor=best_floor)
    tuned_chosen = np.array(
        [tuned_policy.decide({b: probs_test[b][i] for b in costs}, bases).chosen for i in range(n)]
    )
    tuned_stats = policy_stats(f"score_floor_valtuned_{best_floor}", tuned_chosen, correct, costs, capable)
    results.append(tuned_stats)
    ci = bootstrap_ci(tuned_chosen, correct, costs, capable)
    kill = check_kill_criteria(tuned_stats)

    report = {
        "n_test": n,
        "policies": [s.to_dict() for s in results],
        "val_tuned_floor": best_floor,
        "val_tuned_ci": ci,
        "kill_criteria": kill,
        "latent_router_loo_accuracy": latent_loo,
    }
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    report["_stats"] = results
    report["_prompt_router"] = prompt_router_diag
    return report
