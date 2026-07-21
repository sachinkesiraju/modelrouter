import numpy as np
from fastapi.testclient import TestClient
from portallib import ChoiceExample

from portal_dispatch.dispatch import BaseSpec
from portal_dispatch.gateway import create_app
from portal_dispatch.tracing import TraceJournal


class MockBackend:
    def __init__(self, best_idx: int) -> None:
        self.best_idx = best_idx

    def score_rows(self, rows: list[ChoiceExample], task: str) -> list[list[float]]:
        return [[0.0 if i == self.best_idx else -1.0 for i in range(len(r.choices))] for r in rows]


class MockRouter:
    def predict_proba(self, emb):
        return {"cheap": np.array([0.9]), "capable": np.array([0.8])}


class MockClassifier:
    def predict(self, prompts):
        return ["rte"] * len(prompts)


def test_gateway_routes_and_traces(tmp_path):
    journal = TraceJournal(tmp_path / "traces.jsonl")
    app = create_app(
        backends={"cheap": MockBackend(0), "capable": MockBackend(1)},
        bases=[BaseSpec("cheap", 1.0), BaseSpec("capable", 2.0)],
        router=MockRouter(),
        task_classifier=MockClassifier(),
        embed=lambda prompts: np.zeros((len(prompts), 4)),
        journal=journal,
    )
    client = TestClient(app)

    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert {m["id"] for m in resp.json()["data"]} == {"cheap", "capable"}

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Is the sky blue?"}],
              "choices_options": [" Yes", " No"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["portal_dispatch"]["chosen"] == "cheap"
    assert body["choices"][0]["message"]["content"] == " Yes"

    traces = journal.read()
    assert len(traces) == 1
    assert traces[0]["chosen"] == "cheap"
