"""Regression models with calibrated uncertainty.

A point prediction is not enough to choose the next plate. Picking compounds
to assay needs to know where the model is confident, so every model here
exposes ``predict_with_uncertainty`` and the intervals are calibrated against
held-out data rather than trusted as they come out of the ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_SEED = 42


@dataclass
class RandomForestModel:
    """Random forest whose per-tree spread is used as a raw uncertainty."""

    n_estimators: int = 500
    min_samples_leaf: int = 1
    max_features: str | float = "sqrt"
    seed: int = DEFAULT_SEED
    n_jobs: int = -1
    _model: object | None = field(default=None, init=False, repr=False)

    name = "random_forest"

    def fit(self, features: np.ndarray, target: np.ndarray) -> RandomForestModel:
        from sklearn.ensemble import RandomForestRegressor

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self._model.fit(features, target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")
        return self._model.predict(features)

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard deviation across the trees.

        The per-tree loop dominates runtime in the campaign simulation, which
        retrains and rescores a few thousand compounds every round, so the
        trees are evaluated in parallel. Each tree is deterministic once
        fitted, so this changes speed and nothing else.
        """
        if self._model is None:
            raise RuntimeError("fit() must be called before predict_with_uncertainty()")
        from joblib import Parallel, delayed

        predictions = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(tree.predict)(features) for tree in self._model.estimators_
        )
        per_tree = np.stack(predictions)
        return per_tree.mean(axis=0), per_tree.std(axis=0)


@dataclass
class GradientBoostingModel:
    """XGBoost point predictor, bagged over seeds to obtain a spread.

    Boosted trees have no natural ensemble spread, so a small bag of models
    trained on bootstrap resamples supplies one. The bag is kept deliberately
    small; its job is to rank uncertainty, and the width is calibrated
    afterwards by :class:`ConformalRegressor`.
    """

    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    n_bags: int = 5
    seed: int = DEFAULT_SEED
    n_jobs: int = -1
    _models: list = field(default_factory=list, init=False, repr=False)

    name = "gradient_boosting"

    def _make(self, seed: int):
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=seed,
            n_jobs=self.n_jobs,
            tree_method="hist",
        )

    def fit(self, features: np.ndarray, target: np.ndarray) -> GradientBoostingModel:
        rng = np.random.default_rng(self.seed)
        self._models = []
        for bag in range(self.n_bags):
            rows = rng.integers(0, len(features), size=len(features))
            model = self._make(self.seed + bag)
            model.fit(features[rows], target[rows])
            self._models.append(model)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_with_uncertainty(features)[0]

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._models:
            raise RuntimeError("fit() must be called before predict()")
        per_bag = np.stack([model.predict(features) for model in self._models])
        return per_bag.mean(axis=0), per_bag.std(axis=0)


@dataclass
class ConformalRegressor:
    """Wrap a model so its intervals have a guaranteed coverage level.

    Split conformal prediction: part of the training data is held back as a
    calibration set, the absolute residuals on it are measured, and the
    requested quantile of those residuals becomes the interval half-width.
    Scaling the residuals by the model's own uncertainty before taking the
    quantile makes the interval adaptive — wider where the ensemble disagrees
    — while keeping the finite-sample coverage guarantee.

    The guarantee assumes calibration and test data are exchangeable. Under a
    scaffold or document split they are not, by construction, so the measured
    coverage on such a split is reported as a result in its own right rather
    than asserted.
    """

    base: object
    confidence: float = 0.8
    calibration_fraction: float = 0.25
    seed: int = DEFAULT_SEED
    _quantile: float | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"conformal_{getattr(self.base, 'name', 'model')}"

    def fit(self, features: np.ndarray, target: np.ndarray) -> ConformalRegressor:
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(features))
        n_calibration = max(int(round(len(features) * self.calibration_fraction)), 10)
        if n_calibration >= len(features):
            raise ValueError(
                f"{len(features)} training rows is too few to hold out "
                f"{n_calibration} for calibration"
            )
        calibration_rows, fit_rows = order[:n_calibration], order[n_calibration:]

        self.base.fit(features[fit_rows], target[fit_rows])
        mean, spread = self.base.predict_with_uncertainty(features[calibration_rows])
        # Guard against a zero spread making the score infinite.
        scaled = np.abs(target[calibration_rows] - mean) / np.maximum(spread, 1e-6)
        level = min(
            (1.0 - self.confidence) * (1.0 + 1.0 / n_calibration), 1.0
        )
        self._quantile = float(np.quantile(scaled, 1.0 - level, method="higher"))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.base.predict(features)

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.base.predict_with_uncertainty(features)

    def predict_interval(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(prediction, lower, upper)``."""
        if self._quantile is None:
            raise RuntimeError("fit() must be called before predict_interval()")
        mean, spread = self.base.predict_with_uncertainty(features)
        half_width = self._quantile * np.maximum(spread, 1e-6)
        return mean, mean - half_width, mean + half_width


def build_model(name: str, seed: int = DEFAULT_SEED):
    """Factory used by the CLI."""
    if name == "random_forest":
        return RandomForestModel(seed=seed)
    if name == "gradient_boosting":
        return GradientBoostingModel(seed=seed)
    raise ValueError(
        f"unknown model {name!r}; expected 'random_forest' or 'gradient_boosting'"
    )


MODEL_NAMES = ("random_forest", "gradient_boosting")
