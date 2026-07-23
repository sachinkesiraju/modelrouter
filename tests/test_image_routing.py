import numpy as np

from modelrouter.dispatch import BaseSpec, FloorPolicy
from modelrouter.eval import check_kill_criteria, policy_stats
from modelrouter.image_routing import ImageVibeClassifier, ImageVibeRouter


def test_image_vibe_router_learns_synthetic_signal():
    """Router should learn to map style keywords to the right model."""
    prompts = []
    gold = []
    for _ in range(8):
        prompts.extend(["a cat, anime style", "a cat, photorealistic", "a cat, logo", "a simple cat"])
        gold.extend(["anime", "photo", "logo", "cheap"])
    labels = {m: np.array([1 if g == m else 0 for g in gold]) for m in ("anime", "photo", "logo", "cheap")}
    router = ImageVibeRouter()
    router.fit(prompts, labels)
    preds = router.predict(["a cat, anime style", "a cat, photorealistic", "a cat, logo", "a simple cat"])
    assert preds[0] == "anime"
    assert preds[1] == "photo"
    assert preds[2] == "logo"
    assert preds[3] == "cheap"


def test_image_vibe_router_save_load(tmp_path):
    prompts = ["x anime", "y photo"]
    labels = {
        "anime": np.array([1, 0]),
        "photo": np.array([0, 1]),
    }
    router = ImageVibeRouter()
    router.fit(prompts, labels)
    path = str(tmp_path / "router.joblib")
    router.save(path)

    loaded = ImageVibeRouter()
    loaded.load(path)
    assert loaded.predict(["x anime"]) == ["anime"]


def test_image_vibe_classifier_and_abstain_signal():
    prompts = ["anime style cat", "photorealistic cat", "logo for a cat cafe"] * 4
    vibes = ["anime", "photo", "logo"] * 4
    clf = ImageVibeClassifier()
    clf.fit(prompts, vibes)
    assert clf.accuracy(prompts, vibes) == 1.0
    assert clf.predict(["anime style cat"]) == ["anime"]
    # Confidence is the max class probability; predicted class should be correct.
    assert all(c > 1.0 / 3.0 for c in clf.confidence(prompts))


def test_image_vibe_router_dispatch_policy_stats():
    """End-to-end: router + FloorPolicy + eval on a tiny ladder."""
    prompts_train = [
        "a simple cat",
        "an anime cat",
        "a photorealistic cat",
        "a logo for a cat cafe",
    ] * 4
    gold = ["cheap", "anime", "photo", "logo"] * 4
    costs = {"cheap": 0.1, "anime": 0.2, "photo": 0.4, "logo": 0.6}
    labels = {m: np.array([1 if g == m else 0 for g in gold]) for m in costs}
    router = ImageVibeRouter()
    router.fit(prompts_train, labels)

    prompts_test = [
        "a simple cat",
        "an anime cat",
        "a photorealistic cat",
        "a logo for a cat cafe",
    ]
    gold_test = ["cheap", "anime", "photo", "logo"]
    correct = {m: np.array([1 if g == m else 0 for g in gold_test]) for m in costs}
    probs = router.predict_proba(prompts_test)
    bases = [BaseSpec(name=m, cost=c) for m, c in costs.items()]
    policy = FloorPolicy(floor=1.2)
    chosen = np.array([policy.decide({m: probs[m][i] for m in costs}, bases).chosen for i in range(4)])
    stats = policy_stats("test", chosen, correct, costs, capable="logo")
    assert stats.savings > 0.0
    assert check_kill_criteria(stats)["quality_ok"]
