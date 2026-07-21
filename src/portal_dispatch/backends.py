"""Completion backends behind one seam: local PorTAL-served bases and commercial APIs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    cost_usd: float
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


class CompletionBackend(Protocol):
    """One candidate model behind the routing seam."""

    name: str

    def complete(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.0) -> Completion: ...


@dataclass
class LiteLLMBackend:
    """Commercial API models via litellm (OpenAI, Anthropic, Together, Gemini, ...).

    ``model`` uses litellm naming, e.g. ``anthropic/claude-3-5-haiku-20241022`` or
    ``together_ai/meta-llama/Llama-3.2-3B-Instruct-Turbo``.  Cost comes from
    litellm's price table, with an optional per-1k-token override.
    """

    name: str
    model: str
    price_per_1k_input: float | None = None
    price_per_1k_output: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def complete(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.0) -> Completion:
        import litellm

        start = time.perf_counter()
        resp = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **self.extra,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = resp.get("usage", {}) or {}
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)
        if self.price_per_1k_input is not None or self.price_per_1k_output is not None:
            cost = (in_tok / 1000) * (self.price_per_1k_input or 0.0) + (out_tok / 1000) * (
                self.price_per_1k_output or 0.0
            )
        else:
            try:
                cost = float(litellm.completion_cost(completion_response=resp))
            except Exception:
                cost = 0.0
        return Completion(
            text=resp["choices"][0]["message"]["content"] or "",
            model=self.model,
            cost_usd=cost,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )


@dataclass
class PortalLocalBackend:
    """A local PorTAL-served base as a generative candidate.

    Wraps ``runtime.HFBackend``: materializes the route's task LoRA and generates.
    ``cost_per_1k_tokens`` should reflect amortized GPU/CPU cost so commercial and
    local tiers rank on one axis.
    """

    name: str
    hf_backend: Any
    task: str
    cost_per_1k_tokens: float = 0.0

    def complete(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.0) -> Completion:
        import torch

        backend = self.hf_backend
        factors = backend.materialize(self.task)
        tokenizer = backend.tokenizer
        start = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt")
        with backend._injector.activate(factors), torch.no_grad():
            output = backend.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature or None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        total_tokens = int(inputs["input_ids"].shape[1] + len(new_tokens))
        return Completion(
            text=text,
            model=self.name,
            cost_usd=(total_tokens / 1000) * self.cost_per_1k_tokens,
            latency_ms=latency_ms,
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=len(new_tokens),
        )
