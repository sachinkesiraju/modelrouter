"""Production gateway: per-route YAML policies, API keys, fallbacks, shadow mode.

Config format (routes.yaml):

```yaml
api_keys: []                # empty = open gateway; else list of accepted bearer keys
default_limits: {requests_per_minute: 120}   # optional; applies to keys without overrides
key_limits:                 # optional per-key quotas / rate limits
  some-key: {requests_per_minute: 60, requests_per_day: 10000}
routes:
  default:
    shadow_mode: true       # log the routing decision but always serve default_model
    default_model: capable
    floor: 1.2              # cost-bias knob for the floor policy
    abstain_below: 0.55     # task-classifier confidence guard -> default_model
    task: null              # known-skill routes pin the task here
    router_artifact: null   # optional joblib PromptEmbeddingRouter
    fallback_order: [capable, cheap]
    models:
      - {name: cheap, backend: litellm, model: "together_ai/...", price_per_1k_input: 0.0002}
      - {name: capable, backend: litellm, model: "anthropic/claude-3-5-haiku-20241022"}
```
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .backends import Completion, CompletionBackend, LiteLLMBackend, PortalLocalBackend, VLLMBackend
from .runtime import HFBackend
from .dispatch import BaseSpec, FloorPolicy, RoutingDecision
from .tracing import TraceJournal


@dataclass
class ModelSpec:
    name: str
    backend: str
    model: str
    price_per_1k_input: float | None = None
    price_per_1k_output: float | None = None
    rank_cost: float | None = None  # relative cost used for routing order; defaults to price
    artifact_id: str | None = None  # PorTAL artifact id for backend: portal
    backend_kwargs: dict[str, Any] = field(default_factory=dict)  # passed to HFBackend


@dataclass
class RouteConfig:
    name: str
    models: list[ModelSpec]
    default_model: str
    shadow_mode: bool = True
    floor: float = 1.2
    abstain_below: float | None = None
    task: str | None = None
    router_artifact: str | None = None
    fallback_order: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KeyLimits:
    requests_per_minute: int | None = None
    requests_per_day: int | None = None


class RateLimiter:
    """Fixed-window per-key request counter for minute and day quotas."""

    def __init__(self, limits: dict[str, KeyLimits], default: KeyLimits | None = None) -> None:
        self.limits = limits
        self.default = default
        self._windows: dict[tuple[str, str], tuple[int, int]] = {}  # (key, period) -> (window_id, count)

    def _limits_for(self, key: str) -> KeyLimits | None:
        return self.limits.get(key, self.default)

    def check(self, key: str, now: float | None = None) -> float | None:
        """Record one request; return None if allowed, else seconds until retry."""
        limits = self._limits_for(key)
        if limits is None:
            return None
        now = time.time() if now is None else now
        for period, seconds, cap in (("minute", 60, limits.requests_per_minute),
                                     ("day", 86400, limits.requests_per_day)):
            if cap is None:
                continue
            window = int(now // seconds)
            prev_window, count = self._windows.get((key, period), (window, 0))
            if prev_window != window:
                count = 0
            if count >= cap:
                return (window + 1) * seconds - now
            self._windows[(key, period)] = (window, count + 1)
        return None


@dataclass
class GatewayConfig:
    routes: dict[str, RouteConfig]
    api_keys: list[str] = field(default_factory=list)
    key_limits: dict[str, KeyLimits] = field(default_factory=dict)
    default_limits: KeyLimits | None = None

    @staticmethod
    def from_yaml(path: str | Path) -> "GatewayConfig":
        raw = yaml.safe_load(Path(path).read_text())
        routes: dict[str, RouteConfig] = {}
        for route_name, r in (raw.get("routes") or {}).items():
            models = [ModelSpec(**m) for m in r["models"]]
            routes[route_name] = RouteConfig(
                name=route_name,
                models=models,
                default_model=r.get("default_model", models[-1].name),
                shadow_mode=bool(r.get("shadow_mode", True)),
                floor=float(r.get("floor", 1.2)),
                abstain_below=r.get("abstain_below"),
                task=r.get("task"),
                router_artifact=r.get("router_artifact"),
                fallback_order=list(r.get("fallback_order") or [m.name for m in reversed(models)]),
            )
        key_limits = {k: KeyLimits(**v) for k, v in (raw.get("key_limits") or {}).items()}
        default_limits = KeyLimits(**raw["default_limits"]) if raw.get("default_limits") else None
        return GatewayConfig(routes=routes, api_keys=list(raw.get("api_keys") or []),
                             key_limits=key_limits, default_limits=default_limits)


def build_backends(route: RouteConfig, overrides: dict[str, CompletionBackend] | None = None) -> dict[str, CompletionBackend]:
    backends: dict[str, CompletionBackend] = {}
    for spec in route.models:
        if overrides and spec.name in overrides:
            backends[spec.name] = overrides[spec.name]
        elif spec.backend == "litellm":
            backends[spec.name] = LiteLLMBackend(
                name=spec.name,
                model=spec.model,
                price_per_1k_input=spec.price_per_1k_input,
                price_per_1k_output=spec.price_per_1k_output,
            )
        elif spec.backend == "portal":
            if not spec.artifact_id:
                raise ValueError(f"backend 'portal' for {spec.name!r} requires artifact_id")
            if route.task is None:
                raise ValueError(f"backend 'portal' for {spec.name!r} requires route.task")
            hf = HFBackend(
                model_id=spec.model,
                artifact_id=spec.artifact_id,
                **spec.backend_kwargs,
            )
            hf.load()
            backends[spec.name] = PortalLocalBackend(
                name=spec.name,
                hf_backend=hf,
                task=route.task,
                cost_per_1k_tokens=_rank_cost(spec),
            )
        elif spec.backend == "vllm":
            if not spec.artifact_id:
                raise ValueError(f"backend 'vllm' for {spec.name!r} requires artifact_id")
            if route.task is None:
                raise ValueError(f"backend 'vllm' for {spec.name!r} requires route.task")
            vllm = VLLMBackend(
                name=spec.name,
                model=spec.model,
                artifact_id=spec.artifact_id,
                task=route.task,
                cost_per_1k_tokens=_rank_cost(spec),
                dtype=spec.backend_kwargs.get("dtype", "bfloat16"),
                backend_kwargs={k: v for k, v in spec.backend_kwargs.items() if k != "dtype"},
            )
            vllm.load()
            backends[spec.name] = vllm
        else:
            raise ValueError(f"unknown backend type {spec.backend!r} for model {spec.name!r}")
    return backends


def _rank_cost(spec: ModelSpec) -> float:
    if spec.rank_cost is not None:
        return spec.rank_cost
    return (spec.price_per_1k_input or 0.0) + (spec.price_per_1k_output or 0.0)


class ChatRequest(BaseModel):
    model: str = "route/default"
    messages: list[dict[str, str]]
    max_tokens: int = 256
    temperature: float = 0.0
    task: str | None = None


@dataclass
class RouteRuntime:
    config: RouteConfig
    backends: dict[str, CompletionBackend]
    bases: list[BaseSpec]
    router: Any = None
    embed: Any = None
    task_classifier: Any = None

    def decide(self, prompt: str, task: str | None) -> RoutingDecision:
        cfg = self.config
        # Abstain-to-capable: never route cheap when the task signal is uncertain.
        if task is None and cfg.task is None and self.task_classifier is not None and cfg.abstain_below:
            conf = self.task_classifier.confidence([prompt])[0]
            if conf < cfg.abstain_below:
                return RoutingDecision(chosen=cfg.default_model, scores={"task_confidence": conf},
                                       reason=f"abstain_task_conf<{cfg.abstain_below}")
        if self.router is not None and self.embed is not None:
            probs = {name: float(p[0]) for name, p in self.router.predict_proba(self.embed([prompt])).items()
                     if name in self.backends}
            if probs:
                return FloorPolicy(floor=cfg.floor).decide(probs, self.bases)
        return RoutingDecision(chosen=cfg.default_model, scores={}, reason="no_router_default")

    def serve(self, prompt: str, task: str | None, *, max_tokens: int, temperature: float,
              journal: TraceJournal | None) -> tuple[Completion, RoutingDecision, str]:
        cfg = self.config
        decision = self.decide(prompt, task)
        serve_model = cfg.default_model if cfg.shadow_mode else decision.chosen
        chain = [serve_model] + [m for m in cfg.fallback_order if m != serve_model]
        errors: dict[str, str] = {}
        for name in chain:
            try:
                completion = self.backends[name].complete(prompt, max_tokens=max_tokens,
                                                          temperature=temperature)
                if journal is not None:
                    journal.emit(route=cfg.name, prompt=prompt[:500], task=task or cfg.task,
                                 decision=decision.chosen, decision_reason=decision.reason,
                                 decision_scores=decision.scores, shadow_mode=cfg.shadow_mode,
                                 served=name, fallbacks_tried=list(errors),
                                 cost_usd=completion.cost_usd, latency_ms=completion.latency_ms,
                                 input_tokens=completion.input_tokens,
                                 output_tokens=completion.output_tokens)
                return completion, decision, name
            except Exception as exc:  # noqa: BLE001 - fall back through the chain
                errors[name] = str(exc)
        raise RuntimeError(f"all backends failed for route {cfg.name!r}: {json.dumps(errors)[:500]}")


def create_production_app(
    config: GatewayConfig,
    *,
    journal: TraceJournal | None = None,
    backend_overrides: dict[str, dict[str, CompletionBackend]] | None = None,
    routers: dict[str, Any] | None = None,
    embed: Any = None,
    task_classifier: Any = None,
) -> Any:
    from fastapi import FastAPI, Header, HTTPException

    runtimes: dict[str, RouteRuntime] = {}
    for name, route in config.routes.items():
        overrides = (backend_overrides or {}).get(name)
        router = (routers or {}).get(name)
        if router is None and route.router_artifact:
            from .routing import PromptEmbeddingRouter

            router = PromptEmbeddingRouter()
            router.load(route.router_artifact)
        runtimes[name] = RouteRuntime(
            config=route,
            backends=build_backends(route, overrides),
            bases=[BaseSpec(m.name, _rank_cost(m), m.model) for m in route.models],
            router=router,
            embed=embed,
            task_classifier=task_classifier,
        )

    app = FastAPI(title="modelrouter gateway")
    limiter = RateLimiter(config.key_limits, config.default_limits)

    def check_key(authorization: str | None) -> str:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if config.api_keys and token not in config.api_keys:
            raise HTTPException(status_code=401, detail="invalid API key")
        return token

    def check_quota(key: str) -> None:
        retry_after = limiter.check(key)
        if retry_after is not None:
            raise HTTPException(status_code=429, detail="rate limit exceeded",
                                headers={"Retry-After": str(max(1, int(retry_after + 1)))})

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        if not runtimes:
            raise HTTPException(status_code=503, detail="no routes configured")
        return {
            "status": "ready",
            "routes": {name: {"models": [b.name for b in rt.bases],
                              "router_loaded": rt.router is not None,
                              "shadow_mode": rt.config.shadow_mode}
                       for name, rt in runtimes.items()},
        }

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check_key(authorization)
        return {"object": "list",
                "data": [{"id": f"route/{name}", "object": "model"} for name in runtimes]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        key = check_key(authorization)
        check_quota(key)
        route_name = req.model.removeprefix("route/") if req.model.startswith("route/") else "default"
        if route_name not in runtimes:
            raise HTTPException(status_code=404, detail=f"unknown route {route_name!r}")
        runtime = runtimes[route_name]
        prompt = "\n".join(m["content"] for m in req.messages if m.get("role") == "user")
        completion, decision, served = runtime.serve(
            prompt, req.task, max_tokens=req.max_tokens, temperature=req.temperature, journal=journal
        )
        return {
            "id": f"pd-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "model": completion.model,
            "usage": {"prompt_tokens": completion.input_tokens,
                      "completion_tokens": completion.output_tokens,
                      "total_tokens": completion.input_tokens + completion.output_tokens},
            "modelrouter": {
                "route": route_name,
                "decision": decision.chosen,
                "reason": decision.reason,
                "served": served,
                "shadow_mode": runtime.config.shadow_mode,
                "cost_usd": completion.cost_usd,
                "latency_ms": completion.latency_ms,
            },
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": completion.text},
                         "finish_reason": "stop"}],
        }

    return app
