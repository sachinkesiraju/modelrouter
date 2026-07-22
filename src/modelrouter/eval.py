"""Evaluation: policy stats, bootstrap CIs, kill criteria, Pareto plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyStats:
    name: str
    accuracy: float
    cost: float
    savings: float
    drop_vs_capable: float
    routed_cheap_frac: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "cost": self.cost,
            "savings": self.savings,
            "drop_vs_capable": self.drop_vs_capable,
            "routed_cheap_frac": self.routed_cheap_frac,
        }


def policy_stats(
    name: str,
    chosen: np.ndarray,
    correct: dict[str, np.ndarray],
    costs: dict[str, float],
    capable: str,
    extra_cost: np.ndarray | None = None,
) -> PolicyStats:
    """Compute accuracy/cost/savings of a per-example base assignment.

    ``chosen`` is an array of base names per example; ``correct[base]`` is the
    0/1 correctness of that base per example.  ``extra_cost`` adds per-example
    cost (e.g. cascade double inference).
    """
    acc = float(np.mean([correct[b][i] for i, b in enumerate(chosen)]))
    cost = float(np.mean([costs[b] for b in chosen])) + (float(np.mean(extra_cost)) if extra_cost is not None else 0.0)
    capable_cost = costs[capable]
    capable_acc = float(np.mean(correct[capable]))
    cheap_name = min(costs, key=lambda k: costs[k])
    return PolicyStats(
        name=name,
        accuracy=acc,
        cost=cost,
        savings=1.0 - cost / capable_cost,
        drop_vs_capable=capable_acc - acc,
        routed_cheap_frac=float(np.mean([b == cheap_name for b in chosen])),
    )


def bootstrap_ci(
    chosen: np.ndarray,
    correct: dict[str, np.ndarray],
    costs: dict[str, float],
    capable: str,
    iters: int = 200,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(chosen)
    accs, savs = [], []
    capable_cost = costs[capable]
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        accs.append(float(np.mean([correct[chosen[i]][i] for i in idx])))
        savs.append(1.0 - float(np.mean([costs[chosen[i]] for i in idx])) / capable_cost)
    return {
        "accuracy": (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))),
        "savings": (float(np.percentile(savs, 2.5)), float(np.percentile(savs, 97.5))),
    }


def check_kill_criteria(
    stats: PolicyStats, *, min_savings: float = 0.15, max_quality_drop: float = 0.03
) -> dict[str, bool]:
    """Machine-checkable Go/No-Go gate: >=15% savings at <=3 pp drop by default."""
    return {
        "savings_ok": stats.savings >= min_savings,
        "quality_ok": stats.drop_vs_capable <= max_quality_drop,
        "passed": stats.savings >= min_savings and stats.drop_vs_capable <= max_quality_drop,
    }


def plot_pareto(stats: list[PolicyStats], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for s in stats:
        ax.scatter(s.savings * 100, s.accuracy * 100, label=s.name)
        ax.annotate(s.name, (s.savings * 100, s.accuracy * 100), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("cost savings (%)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("modelrouter policy frontier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
