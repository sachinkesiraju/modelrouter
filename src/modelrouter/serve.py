"""Production gateway: per-route YAML policies, API keys, fallbacks, shadow mode.

Config format (routes.yaml):

```yaml
api_keys: []                # empty = open gateway; else list of accepted bearer keys
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

from .backends import Completion, CompletionBackend, LiteLLMBackend
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


@dataclass
class GatewayConfig:
    routes: dict[str, RouteConfig]
    api_keys: list[str] = field(default_factory=list)

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
        return GatewayConfig(routes=routes, api_keys=list(raw.get("api_keys") or []))


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

    def check_key(authorization: str | None) -> None:
        if not config.api_keys:
            return
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token not in config.api_keys:
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check_key(authorization)
        return {"object": "list",
                "data": [{"id": f"route/{name}", "object": "model"} for name in runtimes]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check_key(authorization)
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
