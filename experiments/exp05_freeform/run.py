"""Routing analysis over free-form GSM8K generations with measured $ costs.

Reads results/gsm8k_scores.json (from score_gsm8k.py) and evaluates always-X,
oracle, and prompt-embedding-router policies using each model's measured
average cost per request. Analysis only; reruns free from the committed scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from modelrouter.dispatch import BaseSpec, FloorPolicy
from modelrouter.eval import bootstrap_ci, policy_stats
from modelrouter.routing import PromptEmbeddingRouter
from modelrouter.sweep import _embed

HERE = Path(__file__).parent


def main() -> None:
    blob = json.loads((HERE / "results" / "gsm8k_scores.json").read_text())
    models = blob["models"]
    val, test = blob["splits"]["val"], blob["splits"]["test"]

    correct = {s: {m: np.array([r["correct"] for r in blob["splits"][s][m]])
                   for m in models} for s in ("val", "test")}
    # Measured average $ per request per model (val+test pooled).
    costs = {m: float(np.mean([r["cost_usd"] for s in ("val", "test")
                               for r in blob["splits"][s][m]])) for m in models}
    capable = max(costs, key=costs.get)
    n = len(test["prompts"])
    print("avg $/request:", {m: round(c, 6) for m, c in costs.items()})

    stats = []

    def add(name, chosen):
        s = policy_stats(name, np.array(chosen), correct["test"], costs, capable)
        stats.append(s)
        print(f"{name:24s} acc={s.accuracy:.3f} savings={s.savings:.3f} "
              f"drop={s.drop_vs_capable:+.3f}")
        return s

    for m in models:
        add(f"always_{m.split('/')[-1]}", [m] * n)

    ordered = sorted(models, key=costs.get)
    add("oracle", [next((m for m in ordered if correct["test"][m][i]), capable)
                   for i in range(n)])

    # Train the router on the first 70% of val; tune the floor on the rest;
    # evaluate the frozen (router, floor) on test.
    emb_val = _embed(val["prompts"])
    emb_test = _embed(test["prompts"])
    perm = np.random.default_rng(0).permutation(len(val["prompts"]))
    fit_idx, tune_idx = perm[: int(0.7 * len(perm))], perm[int(0.7 * len(perm)):]
    router = PromptEmbeddingRouter()
    router.fit(emb_val[fit_idx], {m: correct["val"][m][fit_idx] for m in models})
    proba_tune = router.predict_proba(emb_val[tune_idx])
    proba = router.predict_proba(emb_test)
    tune_correct = {m: correct["val"][m][tune_idx] for m in models}

    specs = [BaseSpec(name=m, cost=costs[m]) for m in models]
    floors = (1.0, 1.01, 1.02, 1.03, 1.05, 1.1, 1.2)

    def route(pb, floor, count):
        policy = FloorPolicy(floor=floor)
        return [policy.decide({m: pb[m][i] for m in models}, specs).chosen
                for i in range(count)]

    tuned_floor, tuned_savings = 1.0, -1.0
    for floor in floors:
        s = policy_stats(f"tune_{floor}", np.array(route(proba_tune, floor, len(tune_correct[capable]))),
                         tune_correct, costs, capable)
        if s.drop_vs_capable <= 0.03 and s.savings > tuned_savings:
            tuned_floor, tuned_savings = floor, s.savings

    best = None
    for floor in floors:
        chosen = route(proba, floor, n)
        s = add(f"prompt_floor_{floor}", chosen)
        if floor == tuned_floor:
            best = (chosen, s)
    print("val-tuned floor:", tuned_floor)

    report = {"costs_usd_per_request": costs, "capable": capable,
              "val_tuned_floor": tuned_floor,
              "stats": [s.to_dict() for s in stats]}
    if best is not None:
        report["val_tuned"] = best[1].to_dict()
        report["val_tuned_ci"] = bootstrap_ci(np.array(best[0]), correct["test"], costs, capable)
        print("val-tuned on test:", report["val_tuned"], report["val_tuned_ci"])
    with open(HERE / "results" / "results.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
