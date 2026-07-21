import numpy as np
import pytest
from fastapi.testclient import TestClient

from modelrouter.backends import Completion
from modelrouter.learning import label_traces, retrain_from_traces
from modelrouter.serve import GatewayConfig, create_production_app
from modelrouter.tracing import TraceJournal

CONFIG_YAML = """
api_keys: ["secret-key"]
routes:
  default:
    shadow_mode: {shadow}
    default_model: capable
    floor: 1.2
    fallback_order: [capable, cheap]
    models:
      - {{name: cheap, backend: litellm, model: "together_ai/fake-small", price_per_1k_input: 0.0001}}
      - {{name: capable, backend: litellm, model: "anthropic/fake-big", price_per_1k_input: 0.001}}
"""


class FakeBackend:
    def __init__(self, name: str, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.calls = 0

    def complete(self, prompt, *, max_tokens=256, temperature=0.0):
        self.calls += 1
        if self.fail:
            raise RuntimeError("backend down")
        return Completion(text=f"answer from {self.name}", model=self.name,
                          cost_usd=0.001, latency_ms=5.0, input_tokens=10, output_tokens=5)


class FakeRouter:
    def predict_proba(self, emb):
        return {"cheap": np.array([0.9]), "capable": np.array([0.85])}


def make_app(tmp_path, shadow: bool, cheap_fail=False, capable_fail=False, router=None):
    config_path = tmp_path / "routes.yaml"
    config_path.write_text(CONFIG_YAML.format(shadow=str(shadow).lower()))
    config = GatewayConfig.from_yaml(config_path)
    journal = TraceJournal(tmp_path / "traces.jsonl")
    backends = {"cheap": FakeBackend("cheap", cheap_fail), "capable": FakeBackend("capable", capable_fail)}
    app = create_production_app(
        config, journal=journal,
        backend_overrides={"default": backends},
        routers={"default": router} if router else None,
        embed=lambda prompts: np.zeros((len(prompts), 4)),
    )
    return app, backends, journal


AUTH = {"Authorization": "Bearer secret-key"}


def test_api_key_required(tmp_path):
    app, _, _ = make_app(tmp_path, shadow=True)
    client = TestClient(app)
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/models", headers=AUTH).status_code == 200


def test_shadow_mode_serves_default_but_logs_decision(tmp_path):
    app, backends, journal = make_app(tmp_path, shadow=True, router=FakeRouter())
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", headers=AUTH,
                       json={"messages": [{"role": "user", "content": "hi"}]})
    body = resp.json()
    assert body["modelrouter"]["served"] == "capable"       # shadow: default served
    assert body["modelrouter"]["decision"] == "cheap"        # but router chose cheap
    assert backends["capable"].calls == 1 and backends["cheap"].calls == 0
    trace = journal.read()[0]
    assert trace["decision"] == "cheap" and trace["served"] == "capable" and trace["shadow_mode"]


def test_live_mode_serves_router_choice(tmp_path):
    app, backends, _ = make_app(tmp_path, shadow=False, router=FakeRouter())
    client = TestClient(app)
    body = client.post("/v1/chat/completions", headers=AUTH,
                       json={"messages": [{"role": "user", "content": "hi"}]}).json()
    assert body["modelrouter"]["served"] == "cheap"
    assert backends["cheap"].calls == 1


def test_fallback_chain_on_backend_failure(tmp_path):
    app, backends, journal = make_app(tmp_path, shadow=False, cheap_fail=True, router=FakeRouter())
    client = TestClient(app)
    body = client.post("/v1/chat/completions", headers=AUTH,
                       json={"messages": [{"role": "user", "content": "hi"}]}).json()
    assert body["modelrouter"]["served"] == "capable"
    assert backends["cheap"].calls == 1 and backends["capable"].calls == 1
    assert journal.read()[0]["fallbacks_tried"] == ["cheap"]


def test_unknown_route_404(tmp_path):
    app, _, _ = make_app(tmp_path, shadow=True)
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", headers=AUTH,
                       json={"model": "route/nope", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 404


class FakeClassifier:
    def __init__(self, conf: float) -> None:
        self.conf = conf

    def confidence(self, prompts):
        return [self.conf] * len(prompts)


def test_abstain_to_capable_on_low_task_confidence(tmp_path):
    config_path = tmp_path / "routes.yaml"
    config_path.write_text(CONFIG_YAML.format(shadow="false").replace(
        "floor: 1.2", "floor: 1.2\n    abstain_below: 0.6"))
    config = GatewayConfig.from_yaml(config_path)
    backends = {"cheap": FakeBackend("cheap"), "capable": FakeBackend("capable")}
    app = create_production_app(
        config, journal=None, backend_overrides={"default": backends},
        routers={"default": FakeRouter()},
        embed=lambda prompts: np.zeros((len(prompts), 4)),
        task_classifier=FakeClassifier(0.4),
    )
    client = TestClient(app)
    body = client.post("/v1/chat/completions", headers=AUTH,
                       json={"messages": [{"role": "user", "content": "hi"}]}).json()
    # Router prefers cheap, but low task confidence abstains to the default model.
    assert body["modelrouter"]["decision"] == "capable"
    assert "abstain" in body["modelrouter"]["reason"]


def test_retrain_from_traces(tmp_path):
    journal = TraceJournal(tmp_path / "traces.jsonl")
    rng = np.random.default_rng(0)
    for i in range(60):
        journal.emit(prompt=f"prompt {i}", served="cheap", outcome=int(i % 2 == 0))
    for i in range(60):
        journal.emit(prompt=f"prompt {i}", served="capable", outcome=1)
    traces = label_traces(journal, grader=lambda r: r.get("outcome"))
    meta = retrain_from_traces(
        traces, tmp_path / "routers",
        embed=lambda prompts: rng.normal(size=(len(prompts), 8)),
    )
    assert meta["examples_per_model"] == {"cheap": 60, "capable": 60}
    assert (tmp_path / "routers" / "latest.json").exists()


def test_retrain_requires_min_examples(tmp_path):
    with pytest.raises(ValueError):
        retrain_from_traces(
            [{"prompt": "p", "served": "cheap", "outcome": 1}], tmp_path / "r",
            embed=lambda prompts: np.zeros((len(prompts), 4)),
        )
