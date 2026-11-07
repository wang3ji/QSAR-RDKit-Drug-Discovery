from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from a2a_qsar import assay


def _curated(sds: list[float], counts: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_measurements": counts,
            "pactivity_sd": sds,
            "pactivity_range": [sd * 2 for sd in sds],
        }
    )


def test_estimate_noise_pools_variances_not_deviations():
    # Pooled SD is sqrt(mean(var)) = sqrt((0.04 + 1.0) / 2) ~= 0.721,
    # which is above the median of 0.2 and 1.0.
    frame = _curated([0.2, 1.0], [2, 2])
    estimate = assay.estimate_noise(frame)
    assert estimate.pooled_sd == pytest.approx(np.sqrt((0.04 + 1.0) / 2))
    assert estimate.median_sd == pytest.approx(0.6)
    assert estimate.pooled_sd > estimate.median_sd
    assert estimate.n_replicated_compounds == 2
    assert estimate.recommended_sd == estimate.pooled_sd


def test_estimate_noise_ignores_unreplicated_compounds():
    frame = _curated([float("nan"), 0.5], [1, 2])
    estimate = assay.estimate_noise(frame)
    assert estimate.n_replicated_compounds == 1
    assert estimate.pooled_sd == pytest.approx(0.5)


def test_estimate_noise_falls_back_to_literature_without_replicates():
    frame = _curated([float("nan")], [1])
    estimate = assay.estimate_noise(frame)
    assert estimate.recommended_sd == assay.LITERATURE_PACTIVITY_SD
    assert "literature" in estimate.source


def test_estimate_noise_requires_columns():
    with pytest.raises(ValueError, match="missing columns"):
        assay.estimate_noise(pd.DataFrame({"pactivity": [7.0]}))


def test_sd_for_zprime_penalises_weak_assays():
    base = 0.5
    assert assay.sd_for_zprime(base, 0.9) == pytest.approx(base)  # floored at 1x
    assert assay.sd_for_zprime(base, 0.7) == pytest.approx(base)
    assert assay.sd_for_zprime(base, 0.4) > base
    assert assay.sd_for_zprime(base, 0.2) > assay.sd_for_zprime(base, 0.4)


def test_sd_for_zprime_rejects_out_of_range():
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="zprime"):
            assay.sd_for_zprime(0.5, bad)


def test_simulator_with_zero_noise_is_the_identity_inside_range():
    simulator = assay.AssaySimulator(sd=0.0, lower_limit=4.0, upper_limit=10.5)
    truth = np.array([5.0, 7.5, 9.0])
    observed, censored = simulator.measure(truth)
    assert observed == pytest.approx(truth)
    assert not censored.any()


def test_simulator_censors_outside_dynamic_range():
    simulator = assay.AssaySimulator(sd=0.0, lower_limit=5.0, upper_limit=9.0)
    observed, censored = simulator.measure(np.array([3.0, 7.0, 11.0]))
    assert observed.tolist() == [5.0, 7.0, 9.0]
    assert censored.tolist() == [True, False, True]


def test_simulator_noise_width_matches_requested_sd():
    simulator = assay.AssaySimulator(sd=0.5, lower_limit=-50, upper_limit=50, seed=7)
    truth = np.full(20000, 7.0)
    observed, _ = simulator.measure(truth)
    assert observed.std() == pytest.approx(0.5, abs=0.02)
    assert observed.mean() == pytest.approx(7.0, abs=0.02)


def test_simulator_is_reproducible_from_seed():
    first = assay.AssaySimulator(sd=0.6, seed=3).measure(np.full(50, 7.0))[0]
    second = assay.AssaySimulator(sd=0.6, seed=3).measure(np.full(50, 7.0))[0]
    assert first == pytest.approx(second)


def test_simulator_rejects_bad_configuration():
    with pytest.raises(ValueError, match="sd must be non-negative"):
        assay.AssaySimulator(sd=-0.1)
    with pytest.raises(ValueError, match="lower_limit must be below"):
        assay.AssaySimulator(lower_limit=9.0, upper_limit=5.0)


def test_measure_frame_adds_columns():
    frame = pd.DataFrame({"pactivity": [6.0, 8.0]})
    out = assay.AssaySimulator(sd=0.0).measure_frame(frame)
    assert list(out.columns) == ["pactivity", "observed_pactivity", "censored"]
    assert out["pactivity"].tolist() == [6.0, 8.0]  # original untouched


def test_hit_confirmation_degrades_as_noise_grows():
    rng = np.random.default_rng(0)
    truth = rng.normal(7.5, 1.2, size=600)

    clean = assay.hit_confirmation_rate(
        truth, assay.AssaySimulator(sd=0.0, seed=1), threshold=8.0, n_repeats=20
    )
    noisy = assay.hit_confirmation_rate(
        truth, assay.AssaySimulator(sd=1.0, seed=1), threshold=8.0, n_repeats=20
    )

    assert clean["precision"] == pytest.approx(1.0)
    assert clean["mean_false_positives"] == pytest.approx(0.0)
    assert noisy["precision"] < clean["precision"]
    assert noisy["mean_false_positives"] > 0
    assert noisy["mean_false_negatives"] > 0
    # Same number of truly active compounds in both arms; only the calls differ.
    assert clean["n_truly_active"] == noisy["n_truly_active"]
