"""End-to-end image model vibe routing prototype.

Generates a synthetic preference benchmark, trains an ``ImageVibeRouter``,
and runs a policy sweep. The synthetic benchmark intentionally includes
style-keywords, ambiguous prompts, and noise so the routing signal is
non-trivial while still small enough to run on a CPU.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from modelrouter.dispatch import BaseSpec, FloorPolicy
from modelrouter.eval import PolicyStats, bootstrap_ci, check_kill_criteria, plot_pareto, policy_stats
from modelrouter.image_routing import ImageVibeRouter


# Model ladder: cost is per-image in relative units.
MODELS = ["generic_sd", "anime_sdxl", "realistic_sdxl", "ideogram", "dalle3"]
COSTS = {
    "generic_sd": 0.05,
    "anime_sdxl": 0.10,
    "realistic_sdxl": 0.20,
    "ideogram": 0.30,
    "dalle3": 0.50,
}
CAPABLE = "dalle3"
CHEAP = "generic_sd"

VIBE_HINTS = {
    "generic_sd": ["", "simple", "minimal", "basic"],
    "anime_sdxl": ["anime style", "manga", "cel shaded", "studio ghibli style"],
    "realistic_sdxl": ["photorealistic", "8k photo", "realistic portrait", "dslr"],
    "ideogram": ["logo", "typography", "brand mark", "text poster"],
    "dalle3": ["highly detailed", "complex scene", "editorial"],
}

BASE_PROMPTS = [
    "a cat sitting on a windowsill",
    "a futuristic city at sunset",
    "a portrait of a woman",
    "a logo for a coffee shop",
    "a dragon flying through clouds",
    "an astronaut riding a bicycle",
    "a bowl of fruit on a wooden table",
    "a robot in a garden",
]


def _gold_model(prompt: str, rng: random.Random) -> str:
    """Determine the preferred model for a prompt from style keywords.

    Specialists win their niche most of the time; the capable model is the
    general fallback for ambiguous or complex prompts. This mirrors a real
    ladder where the most expensive model is broadly strong but cheap specialists
    win on their home turf.
    """
    lower = prompt.lower()
    for model, hints in VIBE_HINTS.items():
        if model in ("generic_sd", CAPABLE):
            continue
        if any(h and h in lower for h in hints):
            # Specialist usually wins; capable steals a few hard cases.
            if rng.random() < 0.15:
                return CAPABLE
            return model
    # Ambiguous/simple prompts: capable often wins, generic_sd wins minimal ones.
    if "minimal" in lower or "simple" in lower:
        if rng.random() < 0.8:
            return "generic_sd"
    if rng.random() < 0.6:
        return CAPABLE
    return "generic_sd"


def make_synthetic_dataset(n: int = 2000, seed: int = 0) -> list[dict]:
    """Generate (prompt, gold model, vibe) examples with noise."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        base = rng.choice(BASE_PROMPTS)
        # 70% of prompts carry a strong style hint; 30% are ambiguous.
        if rng.random() < 0.7:
            model = rng.choices(MODELS, weights=[10, 20, 20, 20, 10])[0]
            hint = rng.choice(VIBE_HINTS[model])
            prompt = f"{hint}, {base}" if hint else base
        else:
            prompt = base
        # Inject synthetic noise by occasionally swapping the gold label.
        gold = _gold_model(prompt, rng)
        if rng.random() < 0.05:
            gold = rng.choice(MODELS)
        rows.append({"prompt": prompt, "gold": gold, "id": i})
    return rows


def run(seed: int = 0, out_dir: str = "experiments/exp06_image_vibe_routing/results") -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = make_synthetic_dataset(n=2000, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    train = rows[: int(0.6 * n)]
    val = rows[int(0.6 * n) : int(0.8 * n)]
    test = rows[int(0.8 * n) :]

    # Train router.
    prompts_train = [r["prompt"] for r in train]
    gold_train = [r["gold"] for r in train]
    labels_train = {m: np.array([1 if g == m else 0 for g in gold_train]) for m in MODELS}
    router = ImageVibeRouter()
    router.fit(prompts_train, labels_train)

    # Evaluate on test.
    prompts_test = [r["prompt"] for r in test]
    gold_test = [r["gold"] for r in test]
    probs_test = router.predict_proba(prompts_test)
    bases = [BaseSpec(name=m, cost=COSTS[m]) for m in MODELS]

    correct = {m: np.array([1 if g == m else 0 for g in gold_test]) for m in MODELS}

    results: list[PolicyStats] = []

    # Baselines.
    n_test = len(test)
    results.append(policy_stats("always_cheap", np.array([CHEAP] * n_test), correct, COSTS, CAPABLE))
    results.append(policy_stats("always_capable", np.array([CAPABLE] * n_test), correct, COSTS, CAPABLE))
    oracle = np.array([min((COSTS[m], m) for m in MODELS if correct[m][i])[1] for i in range(n_test)])
    results.append(policy_stats("oracle", oracle, correct, COSTS, CAPABLE))

    # Floor policy sweep over router probabilities.
    for floor in (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0):
        policy = FloorPolicy(floor=floor)
        chosen = np.array(
            [policy.decide({m: probs_test[m][i] for m in MODELS}, bases).chosen for i in range(n_test)]
        )
        results.append(policy_stats(f"vibe_floor_{floor}", chosen, correct, COSTS, CAPABLE))

    # Validation-tuned floor (select best floor on val that keeps drop <= 3 pp).
    prompts_val = [r["prompt"] for r in val]
    probs_val = router.predict_proba(prompts_val)
    gold_val = [r["gold"] for r in val]
    correct_val = {m: np.array([1 if g == m else 0 for g in gold_val]) for m in MODELS}
    best_floor, best_savings = 1.0, -1.0
    for floor in (1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5):
        policy = FloorPolicy(floor=floor)
        chosen_v = np.array(
            [policy.decide({m: probs_val[m][i] for m in MODELS}, bases).chosen for i in range(len(val))]
        )
        s = policy_stats(f"val_floor_{floor}", chosen_v, correct_val, COSTS, CAPABLE)
        if s.drop_vs_capable <= 0.03 and s.savings > best_savings:
            best_floor, best_savings = floor, s.savings

    tuned_policy = FloorPolicy(floor=best_floor)
    tuned_chosen = np.array(
        [tuned_policy.decide({m: probs_test[m][i] for m in MODELS}, bases).chosen for i in range(n_test)]
    )
    tuned_stats = policy_stats(f"vibe_floor_valtuned_{best_floor}", tuned_chosen, correct, COSTS, CAPABLE)
    results.append(tuned_stats)
    ci = bootstrap_ci(tuned_chosen, correct, COSTS, CAPABLE, iters=200, seed=seed)
    kill = check_kill_criteria(tuned_stats)

    # Save Pareto plot.
    plot_pareto(results, str(out_path / "pareto.png"))

    report = {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": n_test,
        "models": MODELS,
        "costs": COSTS,
        "capable": CAPABLE,
        "cheap": CHEAP,
        "policies": [s.to_dict() for s in results],
        "val_tuned_floor": best_floor,
        "val_tuned_ci": ci,
        "kill_criteria": kill,
    }
    with open(out_path / "results.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"val_tuned_floor={best_floor}")
    for s in results:
        print(
            f"{s.name:32s} acc={s.accuracy:.3f} cost={s.cost:.3f} "
            f"savings={s.savings:.3f} drop={s.drop_vs_capable:+.3f} cheap_frac={s.routed_cheap_frac:.3f}"
        )
    print("val_tuned_ci:", ci)
    print("kill_criteria:", kill)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="experiments/exp06_image_vibe_routing/results")
    args = parser.parse_args()
    run(seed=args.seed, out_dir=args.out_dir)
