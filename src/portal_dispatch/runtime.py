"""Runtime: HF backend with PorTAL adapter materialization + per-choice scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from portallib import ChoiceExample, PortalModel
from portallib.evaluation import PortalEvaluator, PortalInjector


@dataclass
class HFBackend:
    """Reference backend: loads a base once, materializes the task LoRA at
    request time (~50 ms) and scores multiple-choice rows."""

    model_id: str
    artifact_id: str
    dtype: str = "bfloat16"
    batch_size: int = 8
    max_prompt: int = 768

    portal: PortalModel = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)
    tokenizer: Any = field(init=False, repr=False)
    _injector: PortalInjector | None = field(init=False, default=None, repr=False)
    _evaluator: PortalEvaluator = field(init=False, repr=False)
    _lora_cache: dict[str, Any] = field(init=False, default_factory=dict, repr=False)

    def load(self) -> None:
        from portallib import PortalBase
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.portal = PortalModel.from_pretrained(self.artifact_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=getattr(torch, self.dtype))
        self.model.eval()
        self.base = PortalBase(model_id=self.model_id, model=self.model, tokenizer=self.tokenizer)
        self._injector = PortalInjector(self.model, self.portal.config)
        self._evaluator = PortalEvaluator(max_prompt=self.max_prompt, batch_size=self.batch_size)

    def materialize(self, task: str) -> Any:
        """Generate (and LRU-cache) the LoRA factors for one task."""
        if task not in self._lora_cache:
            start = time.perf_counter()
            self._lora_cache[task] = self.portal.generate(task)
            self._lora_cache[f"_{task}_ms"] = (time.perf_counter() - start) * 1000
        return self._lora_cache[task]

    @torch.no_grad()
    def score_rows(self, rows: list[ChoiceExample], task: str) -> list[list[float]]:
        """Per-choice normalized log-prob scores with the task LoRA active."""
        factors = self.materialize(task)
        assert self._injector is not None
        with self._injector.activate(factors):
            scores, _nll, _tokens = self._evaluator._score_rows(self.base, rows)
        return scores

    def close(self) -> None:
        if self._injector is not None:
            self._injector.close()
            self._injector = None
