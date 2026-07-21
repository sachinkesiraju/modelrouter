import numpy as np
from fastapi.testclient import TestClient

from modelrouter.backends import Completion
from modelrouter.serve import GatewayConfig, KeyLimits, RateLimiter, create_production_app

CONFIG_YAML = """
api_keys: ["key-a", "key-b"]
default_limits: {requests_per_minute: 100}
key_limits:
  key-a: {requests_per_minute: 2, requests_per_day: 5}
routes:
  default:
    shadow_mode: false
    default_model: capable
    models:
      - {name: capable, backend: litellm, model: "anthropic/fake-big"}
"""


class FakeBackend:
    name = "capable"

    def complete(self, prompt, *, max_tokens=256, temperature=0.0):
        return Completion(text="ok", model="capable", cost_usd=0.0, latency_ms=1.0,
                          input_tokens=1, output_tokens=1)


def make_app(tmp_path):
    config_path = tmp_path / "routes.yaml"
    config_path.write_text(CONFIG_YAML)
    config = GatewayConfig.from_yaml(config_path)
    return create_production_app(
        config, backend_overrides={"default": {"capable": FakeBackend()}},
        embed=lambda prompts: np.zeros((len(prompts), 4)),
    )


def chat(client, key):
    return client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                       json={"messages": [{"role": "user", "content": "hi"}]})


def test_healthz_and_readyz_open(tmp_path):
    client = TestClient(make_app(tmp_path))
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready"
    assert ready["routes"]["default"]["models"] == ["capable"]
    assert ready["routes"]["default"]["shadow_mode"] is False


def test_per_key_rate_limit_429_with_retry_after(tmp_path):
    client = TestClient(make_app(tmp_path))
    assert chat(client, "key-a").status_code == 200
    assert chat(client, "key-a").status_code == 200
    resp = chat(client, "key-a")
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1
    # Other keys are unaffected (fall back to default_limits).
    assert chat(client, "key-b").status_code == 200


def test_rate_limiter_windows():
    limiter = RateLimiter({"k": KeyLimits(requests_per_minute=1, requests_per_day=2)})
    assert limiter.check("k", now=0.0) is None
    assert limiter.check("k", now=1.0) is not None      # minute window full
    assert limiter.check("k", now=61.0) is None          # new minute window
    assert limiter.check("k", now=122.0) is not None     # day quota (2) exhausted
    assert limiter.check("k", now=86401.0) is None       # next day
    assert limiter.check("unknown", now=0.0) is None     # no limits configured


def test_rate_limiter_default_limits():
    limiter = RateLimiter({}, default=KeyLimits(requests_per_minute=1))
    assert limiter.check("any", now=0.0) is None
    assert limiter.check("any", now=1.0) is not None
