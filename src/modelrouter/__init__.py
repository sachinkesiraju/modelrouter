"""modelrouter: cost-aware dispatch across bases with runtime-materialized PorTAL adapters."""

from .dispatch import BaseSpec, CascadePolicy, FloorPolicy, RoutingDecision
from .routing import LatentRouter, PromptEmbeddingRouter, ScoreRouter, TaskClassifier

__all__ = [
    "BaseSpec",
    "CascadePolicy",
    "FloorPolicy",
    "RoutingDecision",
    "LatentRouter",
    "PromptEmbeddingRouter",
    "ScoreRouter",
    "TaskClassifier",
]

__version__ = "0.1.0"
