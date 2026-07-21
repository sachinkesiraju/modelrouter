"""OpenAI-compatible gateway: classify task -> route -> materialize -> score/answer.

Runs with any object satisfying the small ``ScoringBackend`` protocol, so unit
tests can exercise the full route with a mock backend and no model download.
"""

from __future__ import annotations

from typing import Any, Protocol

from portallib import ChoiceExample
from pydantic import BaseModel

from .dispatch import BaseSpec, FloorPolicy
from .tracing import TraceJournal


class ScoringBackend(Protocol):
    def score_rows(self, rows: list[ChoiceExample], task: str) -> list[list[float]]: ...


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[dict[str, str]]
    choices_options: list[str] | None = None
    task: str | None = None


def create_app(
    backends: dict[str, ScoringBackend],
    bases: list[BaseSpec],
    router: Any,
    task_classifier: Any,
    embed: Any,
    policy: FloorPolicy | None = None,
    journal: TraceJournal | None = None,
) -> Any:
    from fastapi import FastAPI

    policy = policy or FloorPolicy(floor=1.2)
    app = FastAPI(title="modelrouter gateway")

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": b.name, "object": "model"} for b in bases]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict[str, Any]:
        prompt = "\n".join(m["content"] for m in req.messages if m.get("role") == "user")
        task = req.task or task_classifier.predict([prompt])[0]
        emb = embed([prompt])
        probs = {name: float(p[0]) for name, p in router.predict_proba(emb).items()}
        decision = policy.decide(probs, bases)
        backend = backends[decision.chosen]
        choices = tuple(req.choices_options) if req.choices_options else (" Yes", " No")
        row = ChoiceExample(task=task, prompt=prompt, choices=choices, gold_idx=0)
        scores = backend.score_rows([row], task)[0]
        answer = choices[max(range(len(scores)), key=lambda i: scores[i])]
        if journal is not None:
            journal.emit(
                prompt=prompt[:200], task=task, candidates=probs, chosen=decision.chosen,
                reason=decision.reason, scores=scores,
            )
        return {
            "object": "chat.completion",
            "model": decision.chosen,
            "modelrouter": {"task": task, "chosen": decision.chosen, "reason": decision.reason},
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
        }

    return app
