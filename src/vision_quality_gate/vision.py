from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class VisionResult:
    model: object
    metrics: dict[str, float]
    abstention_rate: float


def make_dataset(n: int = 600, size: int = 24, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create simple grayscale inspection images: clean, scratch, or spot."""
    rng = np.random.default_rng(seed)
    images = rng.normal(.15, .04, (n, size, size)).clip(0, 1)
    labels = rng.integers(0, 3, n)
    for idx, label in enumerate(labels):
        if label == 1:
            row = rng.integers(4, size - 4)
            images[idx, row - 1:row + 2, 3:-3] += rng.uniform(.45, .8)
        elif label == 2:
            r, c = rng.integers(4, size - 4, 2)
            yy, xx = np.ogrid[:size, :size]
            mask = (yy - r) ** 2 + (xx - c) ** 2 <= rng.integers(3, 7) ** 2
            images[idx][mask] += rng.uniform(.45, .8)
    return images.clip(0, 1).astype("float32"), labels


def extract_features(images: np.ndarray) -> np.ndarray:
    """Use intensity, row/column gradients, and block statistics as features."""
    gradients_y = np.abs(np.diff(images, axis=1)).mean(axis=(1, 2))
    gradients_x = np.abs(np.diff(images, axis=2)).mean(axis=(1, 2))
    blocks = images.reshape(images.shape[0], 6, 4, 6, 4).mean(axis=(2, 4))
    return np.column_stack([images.mean(axis=(1, 2)), images.std(axis=(1, 2)), gradients_y, gradients_x, blocks.reshape(len(images), -1)])


def train_quality_gate(images: np.ndarray, labels: np.ndarray, seed: int = 42, confidence_threshold: float = .60) -> VisionResult:
    if not 0 < confidence_threshold < 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    X = extract_features(images)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=.25, random_state=seed, stratify=labels)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600, random_state=seed))
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    confidence = probabilities.max(axis=1)
    predictions = model.classes_[probabilities.argmax(axis=1)]
    abstain = confidence < confidence_threshold
    safe_predictions = predictions[~abstain]
    safe_labels = y_test[~abstain]
    metrics = {
        "coverage": float((~abstain).mean()),
        "accuracy_on_covered": float(accuracy_score(safe_labels, safe_predictions)) if len(safe_labels) else 0.0,
        "macro_f1_on_covered": float(f1_score(safe_labels, safe_predictions, average="macro", zero_division=0)) if len(safe_labels) else 0.0,
    }
    return VisionResult(model, metrics, float(abstain.mean()))


def evaluate(seed: int = 42, confidence_threshold: float = .60) -> dict[str, float]:
    images, labels = make_dataset(seed=seed)
    result = train_quality_gate(images, labels, seed=seed, confidence_threshold=confidence_threshold)
    return {**result.metrics, "abstention_rate": result.abstention_rate}


def evaluate_thresholds(seed: int = 42, thresholds: tuple[float, ...] = (.50, .60, .70, .80)) -> list[dict[str, float]]:
    """Compare coverage and covered-set quality across abstention thresholds."""
    if not thresholds:
        raise ValueError("thresholds cannot be empty")
    return [
        {"confidence_threshold": threshold, **evaluate(seed=seed, confidence_threshold=threshold)}
        for threshold in thresholds
    ]
