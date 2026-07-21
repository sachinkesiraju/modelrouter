"""Dispatch policies: choose a base per query under cost/quality constraints."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaseSpec:
    """A registered candidate base model."""

    name: str
    cost: float
    model_id: str = ""


@dataclass(frozen=True)
class RoutingDecision:
    chosen: str
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class FloorPolicy:
    """Pre-route to the cheapest base whose predicted-correctness, boosted by
    ``floor``, matches the best candidate.

    ``floor >= 1`` biases toward cheaper bases: a cheap base is chosen when
    ``p_cheap * floor >= max_b p_b``.  ``floor = 1`` is pure argmax; larger
    floors trade quality for savings.
    """

    floor: float = 1.2

    def decide(self, probs: dict[str, float], bases: list[BaseSpec]) -> RoutingDecision:
        ordered = sorted(bases, key=lambda b: b.cost)
        best = max(probs[b.name] for b in ordered)
        for base in ordered:
            if probs[base.name] * self.floor >= best:
                return RoutingDecision(chosen=base.name, scores=dict(probs), reason=f"floor={self.floor}")
        return RoutingDecision(chosen=ordered[-1].name, scores=dict(probs), reason="fallback_capable")


@dataclass(frozen=True)
class CascadePolicy:
    """Run the cheap base first; escalate when its confidence is below threshold.

    ``confidence`` is supplied by the caller (e.g. normalized score margin of the
    cheap base's answer, or a learned escalation rule's probability).
    """

    threshold: float = 0.5

    def decide(self, cheap_confidence: float, bases: list[BaseSpec]) -> RoutingDecision:
        ordered = sorted(bases, key=lambda b: b.cost)
        if cheap_confidence >= self.threshold:
            return RoutingDecision(
                chosen=ordered[0].name,
                scores={"cheap_confidence": cheap_confidence},
                reason=f"cascade_keep@{self.threshold}",
            )
        return RoutingDecision(
            chosen=ordered[-1].name,
            scores={"cheap_confidence": cheap_confidence},
            reason=f"cascade_escalate@{self.threshold}",
        )
