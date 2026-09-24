"""
Fit / predict / uncertainty API for the Stage-II surrogate.

Persists under ``artifacts/surrogate/`` (model + metrics JSON).
Primary targets (locked): SR, CR, TR.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from raise_core.surrogate.targets import DEFAULT_TARGET_KEYS, HIGHWAY_TARGET_KEYS

__all__ = ("DEFAULT_TARGET_KEYS", "HIGHWAY_TARGET_KEYS", "SurrogateModel", "SurrogatePrediction")


@dataclass
class SurrogatePrediction:
    """Point estimate plus a crude uncertainty score in ``[0, +inf)``."""

    y_hat: Dict[str, float]
    uncertainty: float
    model_id: str = ""


def default_model_dir() -> str:
    return "artifacts/surrogate"


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def vectorize_features(
    features: Sequence[Dict[str, Any]],
    *,
    columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert feature dicts to a dense float matrix.

    When ``columns`` is None, infer a stable column list from the batch
    (including expanded ``behavior_fingerprint[*]``).
    """
    if not features:
        return np.zeros((0, 0), dtype=np.float64), list(columns or [])

    max_fp = 0
    for row in features:
        fp = row.get("behavior_fingerprint") or []
        if isinstance(fp, (list, tuple)):
            max_fp = max(max_fp, len(fp))

    if columns is None:
        base = [
            "code_len",
            "score1",
            "score1_raw",
            "score1_holdout",
            "score1_train",
            "score1_degen",
            "score1_rejected",
            "score1_worst_mean",
        ]
        cols = list(base) + [f"fp_{i}" for i in range(max_fp)]
    else:
        cols = list(columns)

    matrix = np.full((len(features), len(cols)), np.nan, dtype=np.float64)
    col_index = {name: i for i, name in enumerate(cols)}

    for r_i, row in enumerate(features):
        for name in (
            "code_len",
            "score1",
            "score1_raw",
            "score1_holdout",
            "score1_train",
            "score1_degen",
            "score1_worst_mean",
        ):
            if name not in col_index:
                continue
            val = row.get(name)
            if _is_finite_number(val):
                matrix[r_i, col_index[name]] = float(val)

        if "score1_rejected" in col_index:
            rejected = row.get("score1_rejected")
            if rejected is None:
                matrix[r_i, col_index["score1_rejected"]] = np.nan
            else:
                matrix[r_i, col_index["score1_rejected"]] = 1.0 if bool(rejected) else 0.0

        fp = row.get("behavior_fingerprint") or []
        if isinstance(fp, (list, tuple)):
            for i, value in enumerate(fp):
                key = f"fp_{i}"
                if key in col_index and _is_finite_number(value):
                    matrix[r_i, col_index[key]] = float(value)

    # sklearn RF cannot take NaN in older versions; impute column medians.
    for c_i in range(matrix.shape[1]):
        col = matrix[:, c_i]
        finite = col[np.isfinite(col)]
        fill = float(np.median(finite)) if finite.size else 0.0
        bad = ~np.isfinite(col)
        if np.any(bad):
            matrix[bad, c_i] = fill

    return matrix, cols


def _label_matrix(
    labels: Sequence[Dict[str, Any]],
    target_keys: Sequence[str],
) -> np.ndarray:
    out = np.zeros((len(labels), len(target_keys)), dtype=np.float64)
    for i, row in enumerate(labels):
        for j, key in enumerate(target_keys):
            val = row.get(key, row.get(key.lower()))
            if not _is_finite_number(val):
                raise ValueError(f"Label row {i} missing finite target {key!r}")
            out[i, j] = float(val)
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    # Average ranks with scipy if available; else simple argsort ranks.
    try:
        from scipy.stats import spearmanr

        coef, _ = spearmanr(a, b)
        return float(coef) if coef is not None and math.isfinite(float(coef)) else float("nan")
    except Exception:  # noqa: BLE001
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        if ra.std() < 1e-12 or rb.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])


def _grouped_split(
    features: Sequence[Dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """80/20-ish split grouped by code_hash when possible."""
    n = len(features)
    indices = np.arange(n)
    if n < 2:
        return indices, np.array([], dtype=int)

    rng = np.random.RandomState(int(seed))
    groups: Dict[str, List[int]] = {}
    for i, row in enumerate(features):
        key = str(row.get("code_hash") or f"row_{i}")
        groups.setdefault(key, []).append(i)
    keys = list(groups.keys())
    rng.shuffle(keys)
    n_val_groups = max(1, int(round(len(keys) * float(val_fraction)))) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val_groups])
    train_idx = [i for k, ids in groups.items() if k not in val_keys for i in ids]
    val_idx = [i for k, ids in groups.items() if k in val_keys for i in ids]
    if not train_idx or not val_idx:
        # Fallback random split.
        perm = rng.permutation(n)
        cut = max(1, int(round(n * (1.0 - val_fraction))))
        cut = min(max(cut, 1), n - 1)
        return perm[:cut], perm[cut:]
    return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)


@dataclass
class SurrogateModel:
    """Offline-trained multi-output predictor (SR, CR, TR)."""

    target_keys: Tuple[str, ...] = DEFAULT_TARGET_KEYS
    n_bags: int = 5
    rf_estimators: int = 64
    random_seed: int = 425
    model_id: str = "surrogate_rf_v1"
    feature_columns: List[str] = field(default_factory=list)
    _bags: List[Any] = field(default_factory=list, repr=False)
    # Per-target scale (train std) so uncertainty is unitless / comparable.
    _target_scales: List[float] = field(default_factory=list, repr=False)

    def fit(
        self,
        features: Sequence[Dict[str, Any]],
        labels: Sequence[Dict[str, Any]],
        *,
        target_keys: Sequence[str] = DEFAULT_TARGET_KEYS,
        val_fraction: float = 0.2,
    ) -> Dict[str, Any]:
        if len(features) != len(labels):
            raise ValueError("features and labels length mismatch")
        if len(features) < 1:
            raise ValueError("need at least one labeled example to fit")

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor

        self.target_keys = tuple(str(k) for k in target_keys) or DEFAULT_TARGET_KEYS
        x_all, cols = vectorize_features(features)
        self.feature_columns = cols
        y_all = _label_matrix(labels, self.target_keys)

        train_idx, val_idx = _grouped_split(
            features, val_fraction=val_fraction, seed=self.random_seed
        )
        x_train, y_train = x_all[train_idx], y_all[train_idx]
        if val_idx.size:
            x_val, y_val = x_all[val_idx], y_all[val_idx]
        else:
            x_val, y_val = x_train, y_train

        # Scale for uncertainty: max(std, floor) so m/s and rates share a [0,1]-ish u.
        scales: List[float] = []
        for j in range(y_train.shape[1]):
            std = float(np.std(y_train[:, j]))
            # Floors: rates ~0.05, continuous targets use larger floors so
            # raw std of mean_progress (~100s) does not dominate gate u.
            key = self.target_keys[j] if j < len(self.target_keys) else ""
            if key in ("SR", "CR", "TR", "soft_success", "scalar"):
                floor = 0.05
            elif key in ("mean_speed", "ITR", "itr"):
                floor = 5.0
            elif key in ("mean_progress", "PL", "pl"):
                floor = 100.0
            else:
                floor = 0.05
            scales.append(max(std, floor))
        self._target_scales = scales

        self._bags = []
        for bag in range(int(self.n_bags)):
            seed = int(self.random_seed) + bag
            base = RandomForestRegressor(
                n_estimators=int(self.rf_estimators),
                random_state=seed,
                min_samples_leaf=1,
                n_jobs=1,
            )
            model = MultiOutputRegressor(base)
            # Bootstrap rows within train for diversity.
            rng = np.random.RandomState(seed)
            if len(x_train) == 1:
                boot = np.array([0])
            else:
                boot = rng.randint(0, len(x_train), size=len(x_train))
            model.fit(x_train[boot], y_train[boot])
            self._bags.append(model)

        pred_val = self._predict_mean_std(x_val)[0]
        metrics: Dict[str, Any] = {
            "n_total": int(len(features)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)) if val_idx.size else int(len(train_idx)),
            "target_keys": list(self.target_keys),
            "n_bags": int(self.n_bags),
            "model_id": self.model_id,
            "target_scales": list(self._target_scales),
            "per_target": {},
        }
        for j, key in enumerate(self.target_keys):
            y_true = y_val[:, j]
            y_hat = pred_val[:, j]
            err = y_hat - y_true
            spearman = _spearman(y_hat, y_true)
            scale = float(self._target_scales[j]) if j < len(self._target_scales) else 1.0
            metrics["per_target"][key] = {
                "mae": float(np.mean(np.abs(err))),
                "mae_normalized": float(np.mean(np.abs(err)) / scale),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "spearman": float(spearman) if math.isfinite(spearman) else None,
                "scale": scale,
            }
        return metrics

    def _predict_mean_std(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._bags:
            raise RuntimeError("SurrogateModel is not fitted / loaded")
        stack = np.stack([bag.predict(x) for bag in self._bags], axis=0)
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        return mean, std

    def predict(self, features: Dict[str, Any]) -> SurrogatePrediction:
        x, _ = vectorize_features([features], columns=self.feature_columns)
        mean, std = self._predict_mean_std(x)
        y_hat = {
            key: float(mean[0, j]) for j, key in enumerate(self.target_keys)
        }
        # Normalize per-target bag-std by train scale so multi-unit targets
        # (SR vs mean_progress) do not inflate gate uncertainty.
        if std.size:
            scales = list(self._target_scales) if self._target_scales else []
            norms: List[float] = []
            for j in range(std.shape[1]):
                s = float(std[0, j])
                scale = float(scales[j]) if j < len(scales) else 1.0
                if scale < 1e-9:
                    scale = 1.0
                norms.append(s / scale)
            uncertainty = float(np.mean(norms)) if norms else 0.0
        else:
            uncertainty = 0.0
        if not math.isfinite(uncertainty) or uncertainty < 0.0:
            uncertainty = 0.0
        return SurrogatePrediction(
            y_hat=y_hat,
            uncertainty=uncertainty,
            model_id=self.model_id,
        )

    def save(self, directory: str) -> None:
        import joblib

        os.makedirs(directory, exist_ok=True)
        payload = {
            "target_keys": list(self.target_keys),
            "n_bags": int(self.n_bags),
            "rf_estimators": int(self.rf_estimators),
            "random_seed": int(self.random_seed),
            "model_id": self.model_id,
            "feature_columns": list(self.feature_columns),
            "bags": self._bags,
            "target_scales": list(self._target_scales),
        }
        joblib.dump(payload, os.path.join(directory, "model.joblib"))
        with open(os.path.join(directory, "feature_columns.json"), "w", encoding="utf-8") as fh:
            json.dump({"columns": list(self.feature_columns)}, fh, indent=2)
            fh.write("\n")
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "target_keys": list(self.target_keys),
                    "n_bags": int(self.n_bags),
                    "rf_estimators": int(self.rf_estimators),
                    "random_seed": int(self.random_seed),
                    "model_id": self.model_id,
                    "target_scales": list(self._target_scales),
                },
                fh,
                indent=2,
            )
            fh.write("\n")

    @classmethod
    def load(cls, directory: str) -> "SurrogateModel":
        import joblib

        path = os.path.join(directory, "model.joblib")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing surrogate model: {path}")
        payload = joblib.load(path)
        model = cls(
            target_keys=tuple(payload.get("target_keys") or DEFAULT_TARGET_KEYS),
            n_bags=int(payload.get("n_bags") or 5),
            rf_estimators=int(payload.get("rf_estimators") or 64),
            random_seed=int(payload.get("random_seed") or 425),
            model_id=str(payload.get("model_id") or "surrogate_rf_v1"),
            feature_columns=list(payload.get("feature_columns") or []),
        )
        model._bags = list(payload.get("bags") or [])
        model._target_scales = [float(x) for x in (payload.get("target_scales") or [])]
        if not model._bags:
            raise ValueError(f"no bags in saved model: {path}")
        # Old models without scales: fall back to unit scale (raw std).
        if not model._target_scales:
            model._target_scales = [1.0] * len(model.target_keys)
        return model
