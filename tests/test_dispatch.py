import numpy as np

from modelrouter.dispatch import BaseSpec, CascadePolicy, FloorPolicy
from modelrouter.eval import bootstrap_ci, check_kill_criteria, policy_stats
from modelrouter.routing import LatentRouter, PromptEmbeddingRouter, ScoreRouter, score_features

BASES = [BaseSpec("cheap", 1.0), BaseSpec("capable", 2.0)]


def test_floor_policy_prefers_cheap_within_floor():
    policy = FloorPolicy(floor=1.2)
    assert policy.decide({"cheap": 0.85, "capable": 0.9}, BASES).chosen == "cheap"
    assert policy.decide({"cheap": 0.4, "capable": 0.9}, BASES).chosen == "capable"


def test_floor_one_is_argmax():
    policy = FloorPolicy(floor=1.0)
    assert policy.decide({"cheap": 0.89, "capable": 0.9}, BASES).chosen == "capable"
    assert policy.decide({"cheap": 0.9, "capable": 0.9}, BASES).chosen == "cheap"


def test_cascade_policy():
    policy = CascadePolicy(threshold=0.5)
    assert policy.decide(0.7, BASES).chosen == "cheap"
    assert policy.decide(0.3, BASES).chosen == "capable"


def test_score_features_shape_and_margin():
    f = score_features([-1.0, -2.0, -3.0])
    assert f.shape == (4,)
    assert f[0] == -1.0 and f[2] == 1.0


def test_score_router_learns_separable_signal():
    rng = np.random.default_rng(0)
    n = 400
    margin = rng.uniform(0, 2, n)
    labels = (margin > 1.0).astype(int)
    feats = np.stack([np.array([-1.0, -1.0 - m, m, 0.5]) for m in margin])
    router = ScoreRouter()
    router.fit({"cheap": feats}, {"cheap": labels})
    probs = router.predict_proba({"cheap": feats})["cheap"]
    assert ((probs > 0.5).astype(int) == labels).mean() > 0.95


def test_prompt_router_and_policy_stats():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(200, 8))
    labels = {"cheap": (emb[:, 0] > 0).astype(int), "capable": np.ones(200, dtype=int)}
    router = PromptEmbeddingRouter()
    router.fit(emb, labels)
    probs = router.predict_proba(emb)
    chosen = np.array(["cheap" if probs["cheap"][i] >= probs["capable"][i] - 0.5 else "capable"
                       for i in range(200)])
    stats = policy_stats("test", chosen, labels, {"cheap": 1.0, "capable": 2.0}, "capable")
    assert 0.0 <= stats.accuracy <= 1.0
    assert stats.savings >= 0.0


def test_latent_router_loo():
    rng = np.random.default_rng(2)
    latents = rng.normal(size=(14, 16))
    labels = (latents[:, 0] > 0).astype(int)
    acc = LatentRouter.loo_accuracy(latents, labels)
    assert 0.0 <= acc <= 1.0


def test_kill_criteria_and_bootstrap():
    n = 100
    correct = {"cheap": np.ones(n, dtype=int), "capable": np.ones(n, dtype=int)}
    chosen = np.array(["cheap"] * 30 + ["capable"] * 70)
    stats = policy_stats("mix", chosen, correct, {"cheap": 1.0, "capable": 2.0}, "capable")
    kill = check_kill_criteria(stats)
    assert kill["passed"] == (stats.savings >= 0.15 and stats.drop_vs_capable <= 0.03)
    ci = bootstrap_ci(chosen, correct, {"cheap": 1.0, "capable": 2.0}, "capable", iters=50)
    assert ci["savings"][0] <= stats.savings <= ci["savings"][1] + 1e-9
