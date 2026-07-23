"""Image generation routing: predict the cheapest model that matches the desired ability/vibe.

The signal is human preference for a generated image rather than exact-match correctness,
so ``ImageVibeRouter`` is a multi-class preference predictor over prompt embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class ImageVibeRouter:
    """Predict per-image-model win probability from a prompt embedding.

    Trained on a preference dataset where each example is ``(prompt, best_model)``.
    The router fits one binary classifier per registered model, exactly mirroring
    ``PromptEmbeddingRouter``, but the label is "this model produces the most
    preferred image for this prompt" instead of "this model answers correctly".

    The encoder can be any sentence-transformer or CLIP/SigLIP-style text model.
    """

    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hidden: int = 0
    models: dict[str, LogisticRegression | MLPClassifier] = field(default_factory=dict)
    _encoder: Any = field(default=None, repr=False)

    def _new_model(self) -> LogisticRegression | MLPClassifier:
        if self.hidden:
            return MLPClassifier(hidden_layer_sizes=(self.hidden,), max_iter=500, random_state=0)
        return LogisticRegression(max_iter=2000)

    def _encode(self, prompts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.encoder_name)
        return np.asarray(self._encoder.encode(prompts, show_progress_bar=False))

    def fit(self, prompts: list[str], labels: dict[str, np.ndarray]) -> None:
        """Fit one binary classifier per model.

        ``labels[model]`` is a 0/1 array of length ``len(prompts)`` indicating whether
        ``model`` is the most preferred model for the corresponding prompt.
        """
        embeddings = self._encode(prompts)
        for model, y in labels.items():
            self.models[model] = _fit_binary(self._new_model(), embeddings, np.asarray(y))

    def predict_proba(self, prompts: list[str]) -> dict[str, np.ndarray]:
        """Return per-model win probabilities for each prompt."""
        embeddings = self._encode(prompts)
        return {model: _proba_of_one(clf, embeddings) for model, clf in self.models.items()}

    def predict(self, prompts: list[str]) -> list[str]:
        """Return the model with the highest win probability for each prompt."""
        probs = self.predict_proba(prompts)
        models = list(probs.keys())
        arr = np.stack([probs[m] for m in models], axis=1)
        return [models[i] for i in np.argmax(arr, axis=1)]

    def save(self, path: str) -> None:
        joblib.dump({"encoder_name": self.encoder_name, "hidden": self.hidden, "models": self.models}, path)

    def load(self, path: str) -> None:
        blob = joblib.load(path)
        self.encoder_name = blob["encoder_name"]
        self.hidden = blob["hidden"]
        self.models = blob["models"]


@dataclass
class ImageVibeClassifier:
    """Prompt -> coarse vibe bucket (photo, anime, logo, painting, ...).

    Used to label traces and as an abstain-to-capable signal when the requested
    vibe is ambiguous.
    """

    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model: LogisticRegression = field(default_factory=lambda: LogisticRegression(max_iter=2000))
    _encoder: Any = field(default=None, repr=False)

    def _encode(self, prompts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.encoder_name)
        return np.asarray(self._encoder.encode(prompts, show_progress_bar=False))

    def fit(self, prompts: list[str], vibes: list[str]) -> None:
        self.model.fit(self._encode(prompts), vibes)

    def predict(self, prompts: list[str]) -> list[str]:
        return list(self.model.predict(self._encode(prompts)))

    def confidence(self, prompts: list[str]) -> list[float]:
        """Max class probability per prompt — the abstain-to-capable signal."""
        probs = self.model.predict_proba(self._encode(prompts))
        return [float(p.max()) for p in probs]

    def accuracy(self, prompts: list[str], vibes: list[str]) -> float:
        preds = self.predict(prompts)
        return float(np.mean([p == v for p, v in zip(preds, vibes)]))
