"""Fitting and evaluation primitives for Models V and T (STRATEGY.md §5–§6).

Deliberately minimal: standardized ridge and L2 logistic regression in numpy,
per-time-bucket fitting, window-level walk-forward evaluation, and block
bootstrap over windows. No sklearn dependency; coefficients serialize to JSON
so the live runner and replay adjusters can load them without numpy at all.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# seconds_to_close buckets; quoting stops at the 90s close cutoff.
TIME_BUCKETS: tuple[tuple[float, float], ...] = ((90.0, 300.0), (300.0, 600.0), (600.0, 900.0))

BOOK_FEATURES = (
    "imb_top1",
    "imb_top3",
    "microprice_minus_mid",
    "flow_1s",
    "flow_5s",
    "bid_size_delta",
    "ask_size_delta",
    "spread",
    "gbm_fair_minus_mid",
    "jump_ratio",
)
EXT_FEATURES = (
    "ext_basis_bps",
    "ext_micro_lead_bps",
    "ext_imbalance",
    "ext_dispersion_bps",
)
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "book": BOOK_FEATURES,
    "book_ext": BOOK_FEATURES + EXT_FEATURES,
}


def bucket_of(seconds_to_close: float) -> Optional[tuple[float, float]]:
    for low, high in TIME_BUCKETS:
        if low <= seconds_to_close < high:
            return (low, high)
    return None


@dataclass
class LinearModel:
    """Standardized linear model: y ≈ intercept + Σ coef·(x − mean)/std."""

    kind: str  # "ridge" | "logistic"
    features: tuple[str, ...]
    means: list[float]
    stds: list[float]
    coefs: list[float]
    intercept: float
    bucket: tuple[float, float]
    n_train: int
    meta: dict = field(default_factory=dict)

    def raw_score(self, values: Sequence[Optional[float]]) -> Optional[float]:
        total = self.intercept
        for value, mean, std, coef in zip(values, self.means, self.stds, self.coefs):
            if value is None:
                return None
            total += coef * (value - mean) / (std if std > 0 else 1.0)
        return total

    def predict(self, values: Sequence[Optional[float]]) -> Optional[float]:
        score = self.raw_score(values)
        if score is None:
            return None
        if self.kind == "logistic":
            return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
        return score

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "features": list(self.features),
            "means": self.means,
            "stds": self.stds,
            "coefs": self.coefs,
            "intercept": self.intercept,
            "bucket": list(self.bucket),
            "n_train": self.n_train,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, row: dict) -> "LinearModel":
        return cls(
            kind=str(row["kind"]),
            features=tuple(row["features"]),
            means=[float(v) for v in row["means"]],
            stds=[float(v) for v in row["stds"]],
            coefs=[float(v) for v in row["coefs"]],
            intercept=float(row["intercept"]),
            bucket=(float(row["bucket"][0]), float(row["bucket"][1])),
            n_train=int(row["n_train"]),
            meta=dict(row.get("meta") or {}),
        )


def save_models(models: list[LinearModel], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([model.to_dict() for model in models], indent=1) + "\n")


def load_models(path: str | Path) -> list[LinearModel]:
    return [LinearModel.from_dict(row) for row in json.loads(Path(path).read_text())]


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds = np.where(stds > 1e-12, stds, 1.0)
    return (x - means) / stds, means, stds


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    features: tuple[str, ...],
    bucket: tuple[float, float],
    l2: float = 1.0,
) -> LinearModel:
    xs, means, stds = _standardize(x)
    n, k = xs.shape
    gram = xs.T @ xs + l2 * np.eye(k)
    coefs = np.linalg.solve(gram, xs.T @ (y - y.mean()))
    return LinearModel(
        kind="ridge",
        features=features,
        means=means.tolist(),
        stds=stds.tolist(),
        coefs=coefs.tolist(),
        intercept=float(y.mean()),
        bucket=bucket,
        n_train=n,
        meta={"l2": l2},
    )


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    features: tuple[str, ...],
    bucket: tuple[float, float],
    l2: float = 1.0,
    iterations: int = 50,
) -> LinearModel:
    """L2-regularized logistic regression via Newton-Raphson."""
    xs, means, stds = _standardize(x)
    n, k = xs.shape
    design = np.hstack([np.ones((n, 1)), xs])
    weights = np.zeros(k + 1)
    penalty = l2 * np.eye(k + 1)
    penalty[0, 0] = 0.0  # do not shrink the intercept
    for _ in range(iterations):
        z = np.clip(design @ weights, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        gradient = design.T @ (p - y) + penalty @ weights
        w_diag = np.maximum(p * (1 - p), 1e-9)
        hessian = (design * w_diag[:, None]).T @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        weights = weights - step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return LinearModel(
        kind="logistic",
        features=features,
        means=means.tolist(),
        stds=stds.tolist(),
        coefs=weights[1:].tolist(),
        intercept=float(weights[0]),
        bucket=bucket,
        n_train=n,
        meta={"l2": l2, "base_rate": float(y.mean())},
    )


def chronological_split(
    window_ids: list[str], *, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[list[str], list[str], list[str]]:
    """Train/validation/holdout by chronological window order (STRATEGY.md §6)."""
    ordered = sorted(set(window_ids))
    n = len(ordered)
    train_end = max(1, int(n * train_frac))
    val_end = max(train_end + 1, int(n * (train_frac + val_frac)))
    return ordered[:train_end], ordered[train_end:val_end], ordered[val_end:]


def block_bootstrap_ci(
    per_window_values: list[float],
    *,
    n_boot: int = 2_000,
    seed: int = 7,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(mean, low, high) resampling whole windows with replacement."""
    if not per_window_values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(per_window_values)
    mean = sum(per_window_values) / n
    draws = []
    for _ in range(n_boot):
        sample = [per_window_values[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    draws.sort()
    low = draws[int(alpha / 2 * n_boot)]
    high = draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return mean, low, high


def per_window_ic(
    predictions: dict[str, list[tuple[float, float]]],
) -> dict[str, float]:
    """Pearson IC per window id from (prediction, outcome) pairs."""
    result = {}
    for window_id, pairs in predictions.items():
        if len(pairs) < 30:
            continue
        xs = np.array([p for p, _ in pairs])
        ys = np.array([y for _, y in pairs])
        sx, sy = xs.std(), ys.std()
        if sx > 0 and sy > 0:
            result[window_id] = float(((xs - xs.mean()) * (ys - ys.mean())).mean() / (sx * sy))
    return result
