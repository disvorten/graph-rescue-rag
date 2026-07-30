from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Standardizer":
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
        )


class BinaryLogisticRegression:
    def __init__(
        self,
        *,
        epochs: int = 350,
        learning_rate: float = 0.05,
        l2: float = 0.01,
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.standardizer: Standardizer | None = None
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.constant_probability: float | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "BinaryLogisticRegression":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
            raise ValueError("Expected a non-empty 2D feature matrix and matching labels")
        unique = np.unique(y)
        if len(unique) == 1:
            self.constant_probability = float(np.clip(unique[0], 1e-5, 1 - 1e-5))
            self.standardizer = Standardizer.fit(x)
            self.weights = np.zeros(x.shape[1], dtype=np.float64)
            return self

        self.standardizer = Standardizer.fit(x)
        z = self.standardizer.transform(x)
        self.weights = np.zeros(z.shape[1], dtype=np.float64)
        positive_weight = len(y) / max(1.0, 2.0 * float(np.sum(y)))
        negative_weight = len(y) / max(1.0, 2.0 * float(np.sum(1.0 - y)))
        sample_weights = np.where(y > 0.5, positive_weight, negative_weight)

        for _ in range(self.epochs):
            probabilities = sigmoid(z @ self.weights + self.bias)
            error = (probabilities - y) * sample_weights
            weight_gradient = (z.T @ error) / len(y) + self.l2 * self.weights
            bias_gradient = float(np.mean(error))
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.constant_probability is not None:
            return np.full(len(x), self.constant_probability, dtype=np.float64)
        if self.standardizer is None or self.weights is None:
            raise RuntimeError("Model is not fitted")
        return sigmoid(self.standardizer.transform(x) @ self.weights + self.bias)

    def to_dict(self) -> dict[str, Any]:
        if self.standardizer is None or self.weights is None:
            raise RuntimeError("Model is not fitted")
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "standardizer": self.standardizer.to_dict(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "constant_probability": self.constant_probability,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BinaryLogisticRegression":
        model = cls(
            epochs=int(value["epochs"]),
            learning_rate=float(value["learning_rate"]),
            l2=float(value["l2"]),
        )
        model.standardizer = Standardizer.from_dict(value["standardizer"])
        model.weights = np.asarray(value["weights"], dtype=np.float64)
        model.bias = float(value["bias"])
        model.constant_probability = value.get("constant_probability")
        return model


class IsotonicCalibrator:
    """Pool-adjacent-violators calibration with piecewise-linear inference."""

    def __init__(self):
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, probabilities: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        p = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if not len(p):
            raise ValueError("Calibration data is empty")
        order = np.argsort(p)
        p, y = p[order], y[order]
        blocks: list[dict[str, float]] = []
        for probability, label in zip(p, y):
            blocks.append(
                {
                    "weight": 1.0,
                    "sum": float(label),
                    "x_sum": float(probability),
                }
            )
            while (
                len(blocks) >= 2
                and blocks[-2]["sum"] / blocks[-2]["weight"]
                > blocks[-1]["sum"] / blocks[-1]["weight"]
            ):
                right = blocks.pop()
                left = blocks.pop()
                blocks.append(
                    {
                        "weight": left["weight"] + right["weight"],
                        "sum": left["sum"] + right["sum"],
                        "x_sum": left["x_sum"] + right["x_sum"],
                    }
                )
        self.x = np.asarray(
            [block["x_sum"] / block["weight"] for block in blocks], dtype=np.float64
        )
        self.y = np.asarray(
            [block["sum"] / block["weight"] for block in blocks], dtype=np.float64
        )
        if len(self.x) == 1:
            self.x = np.asarray([0.0, 1.0])
            self.y = np.asarray([self.y[0], self.y[0]])
        return self

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        if self.x is None or self.y is None:
            raise RuntimeError("Calibrator is not fitted")
        return np.interp(
            np.asarray(probabilities, dtype=np.float64),
            self.x,
            self.y,
            left=self.y[0],
            right=self.y[-1],
        )

    def to_dict(self) -> dict[str, list[float]]:
        if self.x is None or self.y is None:
            raise RuntimeError("Calibrator is not fitted")
        return {"x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IsotonicCalibrator":
        calibrator = cls()
        calibrator.x = np.asarray(value["x"], dtype=np.float64)
        calibrator.y = np.asarray(value["y"], dtype=np.float64)
        return calibrator


class IdentityCalibrator:
    def fit(
        self, probabilities: Sequence[float], labels: Sequence[int]
    ) -> "IdentityCalibrator":
        if not len(probabilities):
            raise ValueError("Calibration data is empty")
        return self

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        return np.clip(
            np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6
        )

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IdentityCalibrator":
        return cls()


class PlattCalibrator:
    """Unweighted logistic calibration on the classifier log-odds."""

    def __init__(
        self,
        *,
        epochs: int = 800,
        learning_rate: float = 0.05,
        l2: float = 0.01,
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.mean = 0.0
        self.scale = 1.0
        self.weight = 1.0
        self.bias = 0.0

    @staticmethod
    def _logits(probabilities: Sequence[float]) -> np.ndarray:
        values = np.clip(
            np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6
        )
        return np.log(values / (1.0 - values))

    def fit(
        self, probabilities: Sequence[float], labels: Sequence[int]
    ) -> "PlattCalibrator":
        logits = self._logits(probabilities)
        y = np.asarray(labels, dtype=np.float64)
        if not len(logits):
            raise ValueError("Calibration data is empty")
        self.mean = float(np.mean(logits))
        self.scale = float(np.std(logits))
        if self.scale < 1e-8:
            self.scale = 1.0
        z = (logits - self.mean) / self.scale
        prevalence = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
        self.weight = 0.0 if len(np.unique(y)) == 1 else 1.0
        self.bias = float(np.log(prevalence / (1.0 - prevalence)))
        if len(np.unique(y)) == 1:
            return self

        for _ in range(self.epochs):
            predicted = sigmoid(self.weight * z + self.bias)
            error = predicted - y
            weight_gradient = float(np.mean(error * z)) + self.l2 * self.weight
            bias_gradient = float(np.mean(error))
            self.weight -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient
        return self

    def transform(self, probabilities: Sequence[float]) -> np.ndarray:
        logits = self._logits(probabilities)
        z = (logits - self.mean) / self.scale
        return np.clip(sigmoid(self.weight * z + self.bias), 1e-6, 1.0 - 1e-6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "mean": self.mean,
            "scale": self.scale,
            "weight": self.weight,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlattCalibrator":
        model = cls(
            epochs=int(value.get("epochs", 800)),
            learning_rate=float(value.get("learning_rate", 0.05)),
            l2=float(value.get("l2", 0.01)),
        )
        model.mean = float(value["mean"])
        model.scale = float(value["scale"])
        model.weight = float(value["weight"])
        model.bias = float(value["bias"])
        return model


def _make_calibrator(name: str):
    if name == "identity":
        return IdentityCalibrator()
    if name == "platt":
        return PlattCalibrator()
    if name == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibration method: {name}")


def _calibration_log_loss(
    probabilities: Sequence[float], labels: Sequence[int]
) -> float:
    values = np.clip(
        np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6
    )
    targets = np.asarray(labels, dtype=np.float64)
    return float(
        -np.mean(
            targets * np.log(values)
            + (1.0 - targets) * np.log(1.0 - values)
        )
    )


def select_calibration_method(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    folds: int = 5,
    seed: int = 42,
) -> tuple[str, dict[str, float]]:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if len(p) != len(y) or not len(p):
        raise ValueError("Calibration probabilities and labels must be non-empty")
    methods = ("identity", "platt", "isotonic")
    fold_count = max(2, min(int(folds), len(p)))
    rng = np.random.default_rng(seed)
    validation_folds: list[list[int]] = [[] for _ in range(fold_count)]
    for label in sorted(set(y.tolist())):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for offset, index in enumerate(indices.tolist()):
            validation_folds[offset % fold_count].append(index)
    validation_folds = [values for values in validation_folds if values]

    scores: dict[str, float] = {}
    all_indices = np.arange(len(p))
    for method in methods:
        predictions = np.full(len(p), np.nan, dtype=np.float64)
        for validation in validation_folds:
            validation_indices = np.asarray(validation, dtype=np.int64)
            train_mask = np.ones(len(p), dtype=bool)
            train_mask[validation_indices] = False
            train_indices = all_indices[train_mask]
            if not len(train_indices):
                predictions[validation_indices] = p[validation_indices]
                continue
            calibrator = _make_calibrator(method).fit(
                p[train_indices], y[train_indices]
            )
            predictions[validation_indices] = calibrator.transform(
                p[validation_indices]
            )
        missing = np.isnan(predictions)
        predictions[missing] = p[missing]
        scores[method] = _calibration_log_loss(predictions, y)

    preference = {"identity": 0, "platt": 1, "isotonic": 2}
    selected = min(methods, key=lambda name: (scores[name], preference[name]))
    return selected, scores


class MRVModel:
    def __init__(
        self,
        *,
        epochs: int = 350,
        learning_rate: float = 0.05,
        l2: float = 0.01,
    ):
        kwargs = dict(epochs=epochs, learning_rate=learning_rate, l2=l2)
        self.add_support = BinaryLogisticRegression(**kwargs)
        self.complete = BinaryLogisticRegression(**kwargs)
        self.reader_gain = BinaryLogisticRegression(**kwargs)
        self.harmful = BinaryLogisticRegression(**kwargs)
        self.feature_names: list[str] = []

    def fit(
        self,
        features: np.ndarray,
        add_support_labels: np.ndarray,
        complete_labels: np.ndarray,
        harmful_labels: np.ndarray,
        feature_names: Sequence[str],
        *,
        reader_gain_labels: np.ndarray | None = None,
    ) -> "MRVModel":
        self.feature_names = list(feature_names)
        self.add_support.fit(features, add_support_labels)
        self.complete.fit(features, complete_labels)
        self.reader_gain.fit(
            features,
            (
                reader_gain_labels
                if reader_gain_labels is not None
                else np.maximum(add_support_labels, complete_labels)
            ),
        )
        self.harmful.fit(features, harmful_labels)
        return self

    def predict(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.add_support.predict_proba(features),
            self.complete.predict_proba(features),
            self.reader_gain.predict_proba(features),
            self.harmful.predict_proba(features),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "add_support": self.add_support.to_dict(),
                    "complete": self.complete.to_dict(),
                    "reader_gain": self.reader_gain.to_dict(),
                    "harmful": self.harmful.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "MRVModel":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls()
        model.feature_names = list(value["feature_names"])
        model.add_support = BinaryLogisticRegression.from_dict(value["add_support"])
        model.complete = BinaryLogisticRegression.from_dict(value["complete"])
        model.reader_gain = BinaryLogisticRegression.from_dict(
            value.get("reader_gain", value["complete"])
        )
        model.harmful = BinaryLogisticRegression.from_dict(value["harmful"])
        return model


class GateModel:
    def __init__(
        self,
        *,
        epochs: int = 350,
        learning_rate: float = 0.05,
        l2: float = 0.01,
        calibration_method: str = "auto",
        calibration_folds: int = 5,
        calibration_seed: int = 42,
        threshold_bootstrap_samples: int = 400,
        threshold_quantile: float = 0.10,
    ):
        self.classifier = BinaryLogisticRegression(
            epochs=epochs, learning_rate=learning_rate, l2=l2
        )
        self.calibration_method = calibration_method
        self.calibration_folds = calibration_folds
        self.calibration_seed = calibration_seed
        self.threshold_bootstrap_samples = threshold_bootstrap_samples
        self.threshold_quantile = threshold_quantile
        self.calibrator = IdentityCalibrator()
        self.selected_calibration_method = "identity"
        self.calibration_scores: dict[str, float] = {}
        self.threshold = 0.5
        self.feature_names: list[str] = []

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        calibration_features: np.ndarray,
        calibration_labels: np.ndarray,
        feature_names: Sequence[str],
        target_recall: float = 0.95,
    ) -> "GateModel":
        self.feature_names = list(feature_names)
        self.classifier.fit(train_features, train_labels)
        raw = self.classifier.predict_proba(calibration_features)
        method = self.calibration_method
        if method == "auto":
            method, self.calibration_scores = select_calibration_method(
                raw,
                calibration_labels,
                folds=self.calibration_folds,
                seed=self.calibration_seed,
            )
        elif method not in {"identity", "platt", "isotonic"}:
            raise ValueError(f"Unknown calibration method: {method}")
        self.selected_calibration_method = method
        self.calibrator = _make_calibrator(method).fit(raw, calibration_labels)
        calibrated = self.calibrator.transform(raw)
        self.threshold = bootstrap_threshold_for_recall(
            calibrated,
            calibration_labels,
            target_recall,
            samples=self.threshold_bootstrap_samples,
            quantile=self.threshold_quantile,
            seed=self.calibration_seed,
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raw = self.classifier.predict_proba(features)
        return self.calibrator.transform(raw)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "threshold": self.threshold,
                    "calibration_method": self.calibration_method,
                    "selected_calibration_method": self.selected_calibration_method,
                    "calibration_folds": self.calibration_folds,
                    "calibration_seed": self.calibration_seed,
                    "threshold_bootstrap_samples": self.threshold_bootstrap_samples,
                    "threshold_quantile": self.threshold_quantile,
                    "calibration_scores": self.calibration_scores,
                    "classifier": self.classifier.to_dict(),
                    "calibrator": self.calibrator.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GateModel":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            calibration_method=value.get("calibration_method", "isotonic"),
            calibration_folds=int(value.get("calibration_folds", 5)),
            calibration_seed=int(value.get("calibration_seed", 42)),
            threshold_bootstrap_samples=int(
                value.get("threshold_bootstrap_samples", 0)
            ),
            threshold_quantile=float(value.get("threshold_quantile", 0.10)),
        )
        model.feature_names = list(value["feature_names"])
        model.threshold = float(value["threshold"])
        model.classifier = BinaryLogisticRegression.from_dict(value["classifier"])
        selected = value.get("selected_calibration_method", "isotonic")
        model.selected_calibration_method = selected
        model.calibration_scores = {
            str(name): float(score)
            for name, score in value.get("calibration_scores", {}).items()
        }
        calibrator_types = {
            "identity": IdentityCalibrator,
            "platt": PlattCalibrator,
            "isotonic": IsotonicCalibrator,
        }
        model.calibrator = calibrator_types[selected].from_dict(value["calibrator"])
        return model


def threshold_for_recall(
    probabilities: Sequence[float], labels: Sequence[int], target_recall: float
) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    positives = int(np.sum(y))
    if positives == 0:
        return 1.0
    best = 0.0
    for threshold in sorted(set(p.tolist()), reverse=True):
        predicted = p >= threshold
        recall = float(np.sum(predicted & (y == 1)) / positives)
        if recall >= target_recall:
            best = float(threshold)
            break
    return best


def bootstrap_threshold_for_recall(
    probabilities: Sequence[float],
    labels: Sequence[int],
    target_recall: float,
    *,
    samples: int = 400,
    quantile: float = 0.10,
    seed: int = 42,
) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    positive_probabilities = p[y == 1]
    if not len(positive_probabilities):
        return 1.0
    baseline = threshold_for_recall(p, y, target_recall)
    if samples <= 0 or len(positive_probabilities) < 2:
        return baseline
    rng = np.random.default_rng(seed)
    thresholds = np.empty(samples, dtype=np.float64)
    positive_labels = np.ones(len(positive_probabilities), dtype=np.int64)
    for index in range(samples):
        sampled = rng.choice(
            positive_probabilities, size=len(positive_probabilities), replace=True
        )
        thresholds[index] = threshold_for_recall(
            sampled, positive_labels, target_recall
        )
    conservative = float(
        np.quantile(thresholds, float(np.clip(quantile, 0.0, 1.0)))
    )
    return min(baseline, conservative)
