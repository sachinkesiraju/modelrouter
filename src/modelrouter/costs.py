"""GPU-amortized $/request cost model for the local tier.

Parameter-proportional costs (cost ∝ parameter count) overstate how much
cheaper small models are on a dedicated GPU: what actually matters is how many
requests per hour each model can push through the same card. This module
converts measured serving throughput (saturated output tokens/sec per model on
a given GPU, e.g. from ``experiments/exp02_vllm_bench`` ``load_bench``) plus an
hourly GPU price into a $/request figure:

    usd_per_request(m) = (tokens_per_request / throughput_tok_s[m])
                         * gpu_hourly_usd / 3600

Assumptions (documented, not hidden):
- The GPU is saturated (throughput measured at peak concurrency); idle time is
  not attributed to any model.
- Requests are priced by output tokens; prefill is folded into the measured
  end-to-end decode throughput.
- Throughput is measured on the bare base model. The vLLM LoRA execution path
  adds a roughly uniform multiplicative overhead (~23% single-request, smaller
  batched; see exp02), which cancels out of *relative* costs between ladder
  models and therefore does not change routing savings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuCostModel:
    """$/request from measured tokens/sec on a fixed hourly-priced GPU."""

    gpu_hourly_usd: float
    throughput_tok_s: dict[str, float]
    gpu: str = ""

    def usd_per_request(self, model: str, tokens_per_request: float) -> float:
        return tokens_per_request / self.throughput_tok_s[model] * self.gpu_hourly_usd / 3600.0

    def relative_costs(self) -> dict[str, float]:
        """Per-request costs normalized so the cheapest model costs 1.0.

        Tokens-per-request and the GPU price cancel: relative cost is just the
        inverse throughput ratio.
        """
        fastest = max(self.throughput_tok_s.values())
        return {m: fastest / tps for m, tps in self.throughput_tok_s.items()}

    @staticmethod
    def from_load_bench(path: str | Path, gpu_hourly_usd: float) -> "GpuCostModel":
        """Build from a combined ``load_bench.json`` written by exp02's ``load_bench_ladder``."""
        combined = json.loads(Path(path).read_text())
        throughput = {model: r["peak_output_tok_per_s"] for model, r in combined.items()}
        gpus = {r.get("gpu", "") for r in combined.values()}
        if len(gpus) != 1:
            raise ValueError(f"load bench mixes GPUs: {sorted(gpus)}")
        return GpuCostModel(gpu_hourly_usd=gpu_hourly_usd, throughput_tok_s=throughput, gpu=gpus.pop())
