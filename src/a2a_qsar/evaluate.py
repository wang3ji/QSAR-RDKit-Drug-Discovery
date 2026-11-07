"""Metrics chosen for how the model will actually be used.

R-squared is reported because reviewers expect it, but it is not the metric
that matters for triage. A model is used to *rank* compounds and to decide
which plate to run next, so rank correlation and enrichment at a realistic
pick size are reported alongside it, and prediction error is always compared
against the assay's own reproducibility: a model whose RMSE has reached the
measurement noise floor cannot be improved by better algorithms, only by
better experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    r2: float
    rmse: float
    mae: float
    spearman: float
    kendall: float
    #: RMSE divided by the assay standard deviation. At or below 1.0 the model
    #: is as close to the truth as a repeat experiment would be.
    rmse_over_assay_sd: float | None
    enrichment_top_5pct: float
    actives_in_top_5pct: int

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "r2": self.r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "spearman": self.spearman,
            "kendall": self.kendall,
            "rmse_over_assay_sd": self.rmse_over_assay_sd,
            "enrichment_top_5pct": self.enrichment_top_5pct,
            "actives_in_top_5pct": self.actives_in_top_5pct,
        }


def enrichment_factor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fraction: float = 0.05,
    threshold: float = 8.0,
) -> tuple[float, int]:
    """Enrichment in the top-ranked fraction, and how many actives it holds.

    An enrichment of 1.0 is what picking at random would give, so the number
    states directly how many times better than random a ranked pick list is.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    actives = y_true >= threshold
    total_actives = int(actives.sum())
    n_selected = max(int(round(len(y_true) * fraction)), 1)
    if total_actives == 0:
        return float("nan"), 0

    top = np.argsort(-y_pred)[:n_selected]
    found = int(actives[top].sum())
    expected_rate = total_actives / len(y_true)
    return float((found / n_selected) / expected_rate), found


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    assay_sd: float | None = None,
    active_threshold: float = 8.0,
) -> RegressionMetrics:
    from scipy.stats import kendalltau, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    if len(y_true) < 2:
        raise ValueError("need at least 2 points to compute metrics")

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    enrichment, found = enrichment_factor(
        y_true, y_pred, fraction=0.05, threshold=active_threshold
    )
    return RegressionMetrics(
        n=len(y_true),
        r2=float(r2_score(y_true, y_pred)),
        rmse=rmse,
        mae=float(mean_absolute_error(y_true, y_pred)),
        spearman=float(spearmanr(y_true, y_pred).statistic),
        kendall=float(kendalltau(y_true, y_pred).statistic),
        rmse_over_assay_sd=(
            float(rmse / assay_sd) if assay_sd not in (None, 0) else None
        ),
        enrichment_top_5pct=enrichment,
        actives_in_top_5pct=found,
    )


def interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    """Observed coverage and mean width of prediction intervals.

    Coverage well below the nominal level means the intervals are
    overconfident, which is the dangerous direction: a campaign would pick
    compounds the model has no business being sure about.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": float(inside.mean()),
        "mean_interval_width": float(np.mean(upper - lower)),
        "median_interval_width": float(np.median(upper - lower)),
    }


def uncertainty_is_informative(
    y_true: np.ndarray, y_pred: np.ndarray, spread: np.ndarray, n_bins: int = 4
) -> dict:
    """Check that predicted uncertainty tracks real error.

    Compounds are binned by predicted spread and the actual RMSE of each bin
    is reported. A usable uncertainty gives a monotonically rising curve; a
    flat curve means the spread carries no information and should not be used
    to choose compounds.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    spread = np.asarray(spread, dtype=float)

    order = np.argsort(spread)
    bins = np.array_split(order, n_bins)
    rows = []
    for position, rows_in_bin in enumerate(bins):
        if len(rows_in_bin) == 0:
            continue
        errors = y_true[rows_in_bin] - y_pred[rows_in_bin]
        rows.append(
            {
                "bin": position,
                "n": int(len(rows_in_bin)),
                "mean_predicted_sd": float(spread[rows_in_bin].mean()),
                "actual_rmse": float(np.sqrt(np.mean(errors**2))),
            }
        )
    rmses = [row["actual_rmse"] for row in rows]
    return {
        "bins": rows,
        "monotonic": bool(all(b >= a for a, b in zip(rmses, rmses[1:], strict=False))),
        "rmse_ratio_high_over_low": (
            float(rmses[-1] / rmses[0]) if rmses and rmses[0] > 0 else float("nan")
        ),
    }
