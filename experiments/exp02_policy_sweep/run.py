"""exp02 step 2: train routers on val, sweep dispatch policies on test."""

from __future__ import annotations

import argparse
from pathlib import Path

from portal_dispatch.eval import plot_pareto
from portal_dispatch.sweep import run_policy_sweep

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(HERE / "results" / "scores_bundle.joblib"))
    parser.add_argument("--out", default=str(HERE / "results" / "results.json"))
    args = parser.parse_args()

    report = run_policy_sweep(args.bundle, args.out)
    for stats in report["_stats"]:
        print(f"{stats.name:32s} acc={stats.accuracy:.3f} savings={stats.savings:.3f} "
              f"drop={stats.drop_vs_capable:+.3f} cheap%={stats.routed_cheap_frac:.2f}")
    print("val-tuned floor:", report["val_tuned_floor"], "CI:", report["val_tuned_ci"])
    print("kill criteria:", report["kill_criteria"])
    print("latent router LOO accuracy:", report["latent_router_loo_accuracy"])
    plot_pareto(report["_stats"], str(HERE / "results" / "pareto.png"))


if __name__ == "__main__":
    main()
