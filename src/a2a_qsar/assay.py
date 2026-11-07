"""An empirical model of assay measurement error.

Most QSAR work treats the activity column as the truth. An assay biologist
knows it is an estimate with a standard deviation attached, produced by a
plate-based experiment with a finite dynamic range and a quality-control
record. This module makes that explicit, so that the cost of assay noise can
be *measured* in the active-learning simulation instead of being ignored.

The noise width is not assumed. It is estimated from compounds in the
annotated snapshot that were measured in more than one assay: the spread
between those independent measurements of the same structure is exactly the
inter-assay reproducibility the simulation needs.

Three effects are modelled, each one a thing that happens on a real plate:

1. **Inter-assay variability** — a Gaussian term on the log scale, whose width
   comes from the replicate estimate above.
2. **Dynamic range** — potency cannot be read beyond the top and bottom of the
   curve. Values outside the range are censored to the limit, as a real
   report would state ``> 10 uM``.
3. **Plate-level quality** — assay robustness summarised by Z-prime. A poor
   Z-prime inflates the variability term, which is how a weak assay actually
   degrades a screening campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Reported inter-laboratory reproducibility of public pIC50 data is roughly
#: 0.5-0.7 log units (Kramer et al. 2012; Kalliokoski et al. 2013). Used only
#: as a fallback when no replicates are available.
LITERATURE_PACTIVITY_SD = 0.60

#: Assays with Z-prime at or above this value are conventionally "excellent";
#: 0.5 is the usual minimum for a screening-grade assay.
ZPRIME_EXCELLENT = 0.7
ZPRIME_ACCEPTABLE = 0.5


@dataclass(frozen=True)
class NoiseEstimate:
    """Inter-assay variability estimated from repeated measurements."""

    n_replicated_compounds: int
    n_measurements: int
    median_sd: float
    pooled_sd: float
    median_range: float
    #: The value downstream simulations should use.
    recommended_sd: float
    source: str

    def to_dict(self) -> dict:
        return {
            "n_replicated_compounds": self.n_replicated_compounds,
            "n_measurements": self.n_measurements,
            "median_sd": self.median_sd,
            "pooled_sd": self.pooled_sd,
            "median_range": self.median_range,
            "recommended_sd": self.recommended_sd,
            "source": self.source,
        }


def estimate_noise(curated: pd.DataFrame) -> NoiseEstimate:
    """Estimate inter-assay variability from a curated frame.

    Expects the per-compound columns produced by :func:`a2a_qsar.data.load_annotated`:
    ``n_measurements``, ``pactivity_sd`` and ``pactivity_range``.

    The *pooled* standard deviation is the square root of the mean of the
    within-compound variances. It is the right summary to feed a Gaussian
    noise model, because variances add while standard deviations do not, and
    it is deliberately reported next to the median so that the skew caused by
    a handful of badly disagreeing compounds stays visible.
    """
    required = {"n_measurements", "pactivity_sd", "pactivity_range"}
    missing = required - set(curated.columns)
    if missing:
        raise ValueError(f"missing columns for noise estimation: {sorted(missing)}")

    replicated = curated[curated["n_measurements"] > 1]
    deviations = replicated["pactivity_sd"].dropna()
    if deviations.empty:
        return NoiseEstimate(
            n_replicated_compounds=0,
            n_measurements=int(curated["n_measurements"].sum()),
            median_sd=float("nan"),
            pooled_sd=float("nan"),
            median_range=float("nan"),
            recommended_sd=LITERATURE_PACTIVITY_SD,
            source="literature fallback (no replicates available)",
        )

    pooled = float(np.sqrt(np.mean(np.square(deviations.to_numpy()))))
    return NoiseEstimate(
        n_replicated_compounds=int(len(replicated)),
        n_measurements=int(replicated["n_measurements"].sum()),
        median_sd=float(deviations.median()),
        pooled_sd=pooled,
        median_range=float(replicated["pactivity_range"].median()),
        recommended_sd=pooled,
        source="within-compound spread across independent assays",
    )


def sd_for_zprime(base_sd: float, zprime: float) -> float:
    """Scale measurement noise by assay robustness.

    Z-prime is ``1 - 3*(sd_pos + sd_neg) / |mean_pos - mean_neg|``, so the
    control spread that remains after subtracting the signal window scales as
    ``1 - zprime``. Referencing that to an excellent assay gives the
    multiplier used here, floored at 1.0 because a very good assay does not
    make public-data heterogeneity disappear.
    """
    if not 0.0 < zprime < 1.0:
        raise ValueError(f"zprime must be in (0, 1), got {zprime}")
    multiplier = (1.0 - zprime) / (1.0 - ZPRIME_EXCELLENT)
    return float(base_sd * max(multiplier, 1.0))


@dataclass
class AssaySimulator:
    """Turn a true pActivity into what a plate would have reported.

    Parameters
    ----------
    sd:
        Inter-assay standard deviation in log units.
    lower_limit, upper_limit:
        Dynamic range of the readout in pActivity units. Measurements outside
        it are censored to the limit and flagged.
    zprime:
        Optional assay robustness; when given, ``sd`` is scaled by
        :func:`sd_for_zprime`.
    seed:
        Seed for reproducible campaigns.
    """

    sd: float = LITERATURE_PACTIVITY_SD
    lower_limit: float = 4.0
    upper_limit: float = 10.5
    zprime: float | None = None
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sd < 0:
            raise ValueError(f"sd must be non-negative, got {self.sd}")
        if self.lower_limit >= self.upper_limit:
            raise ValueError("lower_limit must be below upper_limit")
        if self.zprime is not None:
            self.sd = sd_for_zprime(self.sd, self.zprime)
        self._rng = np.random.default_rng(self.seed)

    @property
    def effective_sd(self) -> float:
        return self.sd

    def measure(self, true_pactivity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(observed, censored_flag)`` for an array of true values.

        A censored point is still usable as a label — it is the honest value a
        screening group would carry forward — but it is marked so that
        downstream code can treat it as an inequality if it chooses.
        """
        true_pactivity = np.asarray(true_pactivity, dtype=float)
        noise = (
            self._rng.normal(0.0, self.sd, size=true_pactivity.shape)
            if self.sd > 0
            else np.zeros_like(true_pactivity)
        )
        observed = true_pactivity + noise
        censored = (observed < self.lower_limit) | (observed > self.upper_limit)
        observed = np.clip(observed, self.lower_limit, self.upper_limit)
        return observed, censored

    def measure_frame(
        self, frame: pd.DataFrame, column: str = "pactivity"
    ) -> pd.DataFrame:
        """Add ``observed_pactivity`` and ``censored`` columns to a copy."""
        observed, censored = self.measure(frame[column].to_numpy())
        return frame.assign(observed_pactivity=observed, censored=censored)


def hit_confirmation_rate(
    true_pactivity: np.ndarray,
    simulator: AssaySimulator,
    threshold: float = 8.0,
    n_repeats: int = 200,
) -> dict[str, float]:
    """Fraction of calls at ``threshold`` that a second run would reproduce.

    This is the quantity that decides how much of a screening cascade is
    wasted: with a wide noise term, compounds near the threshold flip label
    between runs, so a nominal hit list contains confirmed hits, false
    positives that will fail triage, and false negatives that were never
    followed up at all.
    """
    true_pactivity = np.asarray(true_pactivity, dtype=float)
    truly_active = true_pactivity >= threshold

    called_active = np.zeros(n_repeats, dtype=float)
    true_positive = np.zeros(n_repeats, dtype=float)
    false_positive = np.zeros(n_repeats, dtype=float)
    false_negative = np.zeros(n_repeats, dtype=float)

    for repeat in range(n_repeats):
        observed, _ = simulator.measure(true_pactivity)
        called = observed >= threshold
        called_active[repeat] = called.sum()
        true_positive[repeat] = (called & truly_active).sum()
        false_positive[repeat] = (called & ~truly_active).sum()
        false_negative[repeat] = (~called & truly_active).sum()

    n_called = called_active.mean()
    return {
        "effective_sd": simulator.effective_sd,
        "n_truly_active": float(truly_active.sum()),
        "mean_n_called_active": float(n_called),
        "precision": float(true_positive.mean() / n_called) if n_called else float("nan"),
        "recall": (
            float(true_positive.mean() / truly_active.sum())
            if truly_active.any()
            else float("nan")
        ),
        "mean_false_positives": float(false_positive.mean()),
        "mean_false_negatives": float(false_negative.mean()),
    }
