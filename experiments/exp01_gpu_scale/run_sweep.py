"""exp01 step 2: policy sweep over the GPU score bundle (run locally).

Default costs are the parameter-proportional ones baked into the committed
bundle. Pass ``--gpu-costs`` (a combined load_bench.json from exp02's
``load_bench_ladder``) to re-price the ladder with the GPU-amortized cost
model (``modelrouter.costs.GpuCostModel``) instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from modelrouter.costs import GpuCostModel
from modelrouter.sweep import run_policy_sweep

HERE = Path(__file__).parent

TIER_MODEL_IDS = {
    "cheap": "Qwen/Qwen3-0.6B",
    "mid": "Qwen/Qwen3-1.7B",
    "capable": "Qwen/Qwen3-4B",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(HERE / "results" / "scores_bundle_gpu.joblib"))
    parser.add_argument("--gpu-costs", default=None,
                        help="combined load_bench.json; re-price tiers with the GPU-amortized model")
    parser.add_argument("--gpu-hourly-usd", type=float, default=1.10,
                        help="hourly GPU price used by --gpu-costs (default: Modal A10G)")
    args = parser.parse_args()

    (HERE / "results").mkdir(exist_ok=True)
    suffix = ""
    bundle_path = args.bundle

    if args.gpu_costs:
        model = GpuCostModel.from_load_bench(args.gpu_costs, args.gpu_hourly_usd)
        rel = model.relative_costs()
        bundle = joblib.load(args.bundle)
        bundle["bases"] = {tier: rel[TIER_MODEL_IDS[tier]] for tier in bundle["bases"]}
        print("GPU-amortized relative costs:", bundle["bases"])
        print("measured tok/s:", model.throughput_tok_s, f"({model.gpu}, ${args.gpu_hourly_usd}/h)")
        bundle_path = str(HERE / "results" / "scores_bundle_gpucost.joblib")
        joblib.dump(bundle, bundle_path)
        suffix = "_gpucost"

    # Full 3-tier sweep (cheap=0.6B refit, mid=1.7B source, capable=4B refit).
    report = run_policy_sweep(bundle_path, str(HERE / "results" / f"sweep_3tier{suffix}.json"))
    print("=== 3-tier (0.6B / 1.7B / 4B) ===")
    for s in report["_stats"]:
        print(f"{s.name:28s} acc={s.accuracy:.3f} savings={s.savings:.3f} drop={s.drop_vs_capable:+.3f} "
              f"cheap%={s.routed_cheap_frac:.2f}")
    print("val-tuned floor:", report["val_tuned_floor"], "CI:", report["val_tuned_ci"])
    print("kill criteria:", report["kill_criteria"])
    print("latent LOO:", report["latent_router_loo_accuracy"])

    # Pairwise mid-vs-capable variant (drops the fragile 0.6B tier).
    bundle = joblib.load(bundle_path)
    pair = dict(bundle)
    pair["bases"] = {k: v for k, v in bundle["bases"].items() if k in ("mid", "capable")}
    pair["cheap"] = "mid"
    for split in ("val", "test"):
        pair[split] = [dict(r, scores={k: v for k, v in r["scores"].items() if k != "cheap"})
                       for r in bundle[split]]
    # task latents label mid-vs-capable now
    pair_path = HERE / "results" / f"scores_bundle_pair{suffix}.joblib"
    joblib.dump(pair, pair_path)
    report2 = run_policy_sweep(str(pair_path), str(HERE / "results" / f"sweep_pair{suffix}.json"))
    print("\n=== pairwise (1.7B vs 4B) ===")
    for s in report2["_stats"]:
        print(f"{s.name:28s} acc={s.accuracy:.3f} savings={s.savings:.3f} drop={s.drop_vs_capable:+.3f} "
              f"cheap%={s.routed_cheap_frac:.2f}")
    print("val-tuned floor:", report2["val_tuned_floor"], "CI:", report2["val_tuned_ci"])
    print("kill criteria:", report2["kill_criteria"])
    print("latent LOO:", report2["latent_router_loo_accuracy"])

    # 3-tier oracle (any base) for headroom context.
    costs = bundle["bases"]
    test = bundle["test"]
    ordered = sorted(costs, key=costs.get)
    chosen, cost, acc = [], 0.0, 0
    for r in test:
        pick = next((b for b in ordered if int(np.argmax(r["scores"][b]) == r["gold_idx"])), ordered[-1])
        chosen.append(pick)
        cost += costs[pick]
        acc += int(np.argmax(r["scores"][pick]) == r["gold_idx"])
    n = len(test)
    cap_cost = costs["capable"] * n
    print(f"\n3-tier oracle: acc={acc / n:.3f} savings={1 - cost / cap_cost:.3f}")
    with open(HERE / "results" / f"oracle_3tier{suffix}.json", "w") as fh:
        json.dump({"accuracy": acc / n, "savings": 1 - cost / cap_cost}, fh)


if __name__ == "__main__":
    main()
