"""modelrouter: cost-aware dispatch across bases with runtime-materialized PorTAL adapters."""

from .dispatch import BaseSpec, CascadePolicy, FloorPolicy, RoutingDecision
from .image_routing import ImageVibeClassifier, ImageVibeRouter
from .routing import LatentRouter, PromptEmbeddingRouter, ScoreRouter, TaskClassifier

__all__ = [
    "BaseSpec",
    "CascadePolicy",
    "FloorPolicy",
    "RoutingDecision",
    "ImageVibeClassifier",
    "ImageVibeRouter",
    "LatentRouter",
    "PromptEmbeddingRouter",
    "ScoreRouter",
    "TaskClassifier",
]

__version__ = "0.1.0"
