"""Routers: predict per-base correctness (or the task) from cheap signals."""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def _fit_binary(clf, x: np.ndarray, y: np.ndarray):
    """Fit a binary classifier, degrading to a constant predictor on single-class labels."""
    if len(np.unique(y)) < 2:
        constant = DummyClassifier(strategy="constant", constant=int(y[0]))
        constant.fit(x, y)
        return constant
    clf.fit(x, y)
    return clf


def _proba_of_one(clf, x: np.ndarray) -> np.ndarray:
    probs = clf.predict_proba(x)
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 not in classes:
        return np.zeros(len(x))
    return probs[:, classes.index(1)]


def score_features(choice_scores: list[float]) -> np.ndarray:
    """Distribution features of one base's per-choice scores: top, second, margin, entropy."""
    arr = np.asarray(choice_scores, dtype=np.float64)
    order = np.sort(arr)[::-1]
    top = order[0]
    second = order[1] if len(order) > 1 else order[0]
    probs = np.exp(arr - arr.max())
    probs = probs / probs.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return np.array([top, second, top - second, entropy])


class ScoreRouter:
    """Logistic regression on per-base score-distribution features.

    Predicts, per base, whether that base answers correctly.  Requires running
    the candidate base's forward pass, so it is an offline/oracle-grade signal.
    """

    def __init__(self) -> None:
        self.models: dict[str, LogisticRegression] = {}

    def fit(self, features: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> None:
        for base, x in features.items():
            self.models[base] = _fit_binary(LogisticRegression(max_iter=2000), x, labels[base])

    def predict_proba(self, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {base: _proba_of_one(self.models[base], x) for base, x in features.items()}

    def save(self, path: str) -> None:
        joblib.dump(self.models, path)

    def load(self, path: str) -> None:
        self.models = joblib.load(path)


class PromptEmbeddingRouter:
    """Per-base correctness classifiers on sentence-transformer prompt embeddings.

    Production path: routes from the prompt alone, no candidate forward pass.
    The ``delta`` knob biases the cheap-vs-capable comparison at decision time.
    """

    def __init__(self, hidden: int = 0) -> None:
        self.hidden = hidden
        self.models: dict[str, LogisticRegression | MLPClassifier] = {}

    def _new_model(self) -> LogisticRegression | MLPClassifier:
        if self.hidden:
            return MLPClassifier(hidden_layer_sizes=(self.hidden,), max_iter=500, random_state=0)
        return LogisticRegression(max_iter=2000)

    def fit(self, embeddings: np.ndarray, labels: dict[str, np.ndarray]) -> None:
        for base, y in labels.items():
            self.models[base] = _fit_binary(self._new_model(), embeddings, y)

    def predict_proba(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        return {base: _proba_of_one(clf, embeddings) for base, clf in self.models.items()}

    def save(self, path: str) -> None:
        joblib.dump({"hidden": self.hidden, "models": self.models}, path)

    def load(self, path: str) -> None:
        blob = joblib.load(path)
        self.hidden = blob["hidden"]
        self.models = blob["models"]


class LatentRouter:
    """Predict from the PorTAL task latent ``z`` whether the cheap base is at
    least as good as the capable base for an entire task (zero-shot task routing)."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=2000)

    def fit(self, latents: np.ndarray, cheap_is_good_enough: np.ndarray) -> None:
        self.model.fit(latents, cheap_is_good_enough)

    def predict(self, latents: np.ndarray) -> np.ndarray:
        return self.model.predict(latents)

    def predict_proba(self, latents: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(latents)[:, 1]

    @staticmethod
    def loo_accuracy(latents: np.ndarray, labels: np.ndarray) -> float:
        """Leave-one-task-out accuracy of the cheap-is-good-enough predictor."""
        n = len(labels)
        correct = 0
        for i in range(n):
            mask = np.arange(n) != i
            if len(np.unique(labels[mask])) < 2:
                pred = int(labels[mask][0])
            else:
                clf = LogisticRegression(max_iter=2000)
                clf.fit(latents[mask], labels[mask])
                pred = int(clf.predict(latents[i : i + 1])[0])
            correct += int(pred == int(labels[i]))
        return correct / n


@dataclass
class TaskClassifier:
    """Prompt -> task-id classifier on sentence-transformer embeddings."""

    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model: LogisticRegression = field(default_factory=lambda: LogisticRegression(max_iter=2000))
    _encoder: object = field(default=None, repr=False)

    def _encode(self, prompts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.encoder_name)
        return np.asarray(self._encoder.encode(prompts, show_progress_bar=False))

    def fit(self, prompts: list[str], tasks: list[str]) -> None:
        self.model.fit(self._encode(prompts), tasks)

    def predict(self, prompts: list[str]) -> list[str]:
        return list(self.model.predict(self._encode(prompts)))

    def confidence(self, prompts: list[str]) -> list[float]:
        """Max class probability per prompt — the abstain-to-capable signal."""
        probs = self.model.predict_proba(self._encode(prompts))
        return [float(p.max()) for p in probs]

    def accuracy(self, prompts: list[str], tasks: list[str]) -> float:
        preds = self.predict(prompts)
        return float(np.mean([p == t for p, t in zip(preds, tasks)]))
