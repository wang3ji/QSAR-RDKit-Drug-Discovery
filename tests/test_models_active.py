from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from a2a_qsar import active, evaluate, models

SMILES = [
    "CC(=O)NCCNc1[nH]c(-c2ccccc2)nc2nc(C)c(C)c1-2",
    "c1ccc(CCNc2nc3ccccc3c3cn(-c4ccccc4)nc23)cc1",
    "CCCn1c(=O)c2[nH]c(-c3ccccc3)cc2n(CCC)c1=O",
    "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1",
    "CCCC(=O)Nc1nc(-c2ccccc2)nc(-c2ccccc2)c1C#N",
    "N#Cc1sc(=S)n(-c2ccccc2)c1/N=C/c1ccc(Cl)cc1",
]


def _learnable(n: int = 240, n_features: int = 12, seed: int = 0):
    """A linear problem with noise, so models must do better than the mean."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, n_features))
    weights = rng.normal(size=n_features)
    target = features @ weights + rng.normal(0, 0.3, size=n)
    return features, target


@pytest.mark.parametrize("builder", [models.RandomForestModel, models.GradientBoostingModel])
def test_models_learn_and_report_spread(builder):
    features, target = _learnable()
    model = builder(seed=1)
    if hasattr(model, "n_estimators"):
        model.n_estimators = 60
    if hasattr(model, "n_bags"):
        model.n_bags = 3
    model.fit(features[:180], target[:180])

    prediction, spread = model.predict_with_uncertainty(features[180:])
    assert prediction.shape == (60,)
    assert spread.shape == (60,)
    assert (spread >= 0).all()
    assert spread.max() > 0
    assert prediction == pytest.approx(model.predict(features[180:]), abs=1e-6)

    from sklearn.metrics import r2_score

    assert r2_score(target[180:], prediction) > 0.3


def test_model_raises_before_fit():
    model = models.RandomForestModel()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="fit"):
        models.GradientBoostingModel().predict_with_uncertainty(np.zeros((2, 3)))


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        models.build_model("transformer")


def test_conformal_intervals_reach_nominal_coverage():
    features, target = _learnable(n=600, seed=3)
    base = models.RandomForestModel(n_estimators=80, seed=3)
    wrapped = models.ConformalRegressor(base=base, confidence=0.8, seed=3)
    wrapped.fit(features[:450], target[:450])

    prediction, lower, upper = wrapped.predict_interval(features[450:])
    coverage = evaluate.interval_coverage(target[450:], lower, upper)
    # Exchangeable data, so the guarantee should hold up to sampling error.
    assert coverage["coverage"] >= 0.70
    assert (upper > lower).all()
    assert prediction == pytest.approx((lower + upper) / 2, abs=1e-6)


def test_conformal_width_grows_with_confidence():
    features, target = _learnable(n=600, seed=4)

    def width(confidence: float) -> float:
        wrapped = models.ConformalRegressor(
            base=models.RandomForestModel(n_estimators=60, seed=4),
            confidence=confidence,
            seed=4,
        )
        wrapped.fit(features[:450], target[:450])
        _, lower, upper = wrapped.predict_interval(features[450:])
        return float(np.mean(upper - lower))

    assert width(0.95) > width(0.8) > width(0.5)


def test_conformal_rejects_bad_confidence_and_tiny_training_set():
    features, target = _learnable(n=40)
    with pytest.raises(ValueError, match="confidence"):
        models.ConformalRegressor(base=models.RandomForestModel(), confidence=1.5).fit(
            features, target
        )
    with pytest.raises(ValueError, match="too few"):
        models.ConformalRegressor(
            base=models.RandomForestModel(), calibration_fraction=0.9
        ).fit(features[:10], target[:10])


def test_conformal_raises_before_fit():
    wrapped = models.ConformalRegressor(base=models.RandomForestModel())
    with pytest.raises(RuntimeError, match="fit"):
        wrapped.predict_interval(np.zeros((2, 3)))


def test_enrichment_of_a_perfect_ranking_is_the_maximum():
    y_true = np.array([9.0] * 5 + [6.0] * 95)
    perfect, found = evaluate.enrichment_factor(y_true, y_true, fraction=0.05, threshold=8.0)
    assert found == 5
    assert perfect == pytest.approx(20.0)  # 1 / 0.05

    rng = np.random.default_rng(0)
    random_scores = rng.normal(size=len(y_true))
    random_enrichment, _ = evaluate.enrichment_factor(y_true, random_scores, 0.05, 8.0)
    assert random_enrichment < perfect


def test_regression_metrics_reference_the_assay_noise_floor():
    rng = np.random.default_rng(0)
    y_true = rng.normal(7.5, 1.0, size=300)
    y_pred = y_true + rng.normal(0, 0.6, size=300)

    metrics = evaluate.regression_metrics(y_true, y_pred, assay_sd=0.6)
    assert metrics.n == 300
    assert metrics.rmse == pytest.approx(0.6, abs=0.1)
    # Error at the measurement floor: better algorithms cannot fix this.
    assert metrics.rmse_over_assay_sd == pytest.approx(1.0, abs=0.2)
    assert metrics.spearman > 0.7

    assert evaluate.regression_metrics(y_true, y_pred).rmse_over_assay_sd is None


def test_regression_metrics_validate_input():
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate.regression_metrics(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="at least 2"):
        evaluate.regression_metrics(np.zeros(1), np.zeros(1))


def test_uncertainty_diagnostic_detects_an_informative_spread():
    rng = np.random.default_rng(1)
    spread = np.linspace(0.1, 2.0, 400)
    y_pred = np.zeros(400)
    y_true = rng.normal(0, spread)  # error grows with the reported spread

    report = evaluate.uncertainty_is_informative(y_true, y_pred, spread, n_bins=4)
    assert len(report["bins"]) == 4
    assert report["monotonic"]
    assert report["rmse_ratio_high_over_low"] > 2.0


def _pool(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 16))
    weights = rng.normal(size=16)
    raw = features @ weights
    # Rescale into a realistic pActivity window with ~10% actives.
    pactivity = 6.5 + 1.2 * (raw - raw.mean()) / raw.std()
    frame = pd.DataFrame(
        {
            "smiles": [SMILES[index % len(SMILES)] for index in range(n)],
            "pactivity": pactivity,
        }
    )
    return frame, features


def test_campaign_assays_the_expected_number_of_compounds():
    pool, features = _pool()
    result = active.run_campaign(
        pool,
        features,
        lambda: models.RandomForestModel(n_estimators=40, seed=0),
        acquisition="greedy",
        assay_sd=0.3,
        batch_size=20,
        n_rounds=4,
        n_seed=40,
        seed=0,
    )
    assert len(result.rounds) == 5  # seed round plus four batches
    assert result.rounds[0].n_assayed == 40
    assert result.rounds[-1].n_assayed == 40 + 4 * 20
    assert result.n_pool == len(pool)
    frame = result.to_frame()
    assert frame["n_true_actives_found"].is_monotonic_increasing
    assert frame["recall_of_pool_actives"].max() <= 1.0


@pytest.mark.parametrize("acquisition", active.ACQUISITION_FUNCTIONS)
def test_every_acquisition_runs(acquisition):
    pool, features = _pool()
    result = active.run_campaign(
        pool,
        features,
        lambda: models.RandomForestModel(n_estimators=30, seed=0),
        acquisition=acquisition,
        batch_size=16,
        n_rounds=3,
        n_seed=32,
        seed=0,
    )
    assert result.rounds[-1].n_assayed == 32 + 3 * 16


def test_greedy_beats_random_at_finding_actives():
    pool, features = _pool(n=400, seed=2)

    def campaign(acquisition: str) -> int:
        found = []
        for seed in (0, 1, 2):
            result = active.run_campaign(
                pool,
                features,
                lambda seed=seed: models.RandomForestModel(
                    n_estimators=40, seed=seed
                ),
                acquisition=acquisition,
                assay_sd=0.3,
                batch_size=20,
                n_rounds=5,
                n_seed=40,
                seed=seed,
            )
            found.append(result.rounds[-1].n_true_actives_found)
        return float(np.mean(found))

    assert campaign("greedy") > campaign("random")


def test_diverse_greedy_covers_more_scaffolds_than_greedy():
    pool, features = _pool(n=300, seed=5)
    scaffold_counts = {}
    for acquisition in ("greedy", "diverse_greedy"):
        result = active.run_campaign(
            pool,
            features,
            lambda: models.RandomForestModel(n_estimators=30, seed=0),
            acquisition=acquisition,
            batch_size=16,
            n_rounds=3,
            n_seed=32,
            seed=0,
        )
        scaffold_counts[acquisition] = result.rounds[-1].n_scaffolds_assayed
    assert scaffold_counts["diverse_greedy"] >= scaffold_counts["greedy"]


def test_noise_free_campaign_makes_no_false_calls():
    pool, features = _pool()
    result = active.run_campaign(
        pool,
        features,
        lambda: models.RandomForestModel(n_estimators=30, seed=0),
        acquisition="greedy",
        assay_sd=0.0,
        batch_size=16,
        n_rounds=3,
        n_seed=32,
        seed=0,
    )
    assert all(entry.n_false_calls == 0 for entry in result.rounds)
    assert all(
        entry.n_called_active == entry.n_true_actives_found for entry in result.rounds
    )


def test_noisy_campaign_produces_false_calls():
    pool, features = _pool()
    result = active.run_campaign(
        pool,
        features,
        lambda: models.RandomForestModel(n_estimators=30, seed=0),
        acquisition="greedy",
        assay_sd=1.0,
        batch_size=16,
        n_rounds=3,
        n_seed=32,
        seed=0,
    )
    assert result.rounds[-1].n_false_calls > 0


def test_campaign_validates_its_budget():
    pool, features = _pool(n=100)
    with pytest.raises(ValueError, match="pool has"):
        active.run_campaign(
            pool, features[:50], lambda: models.RandomForestModel(), n_rounds=1
        )
    with pytest.raises(ValueError, match="campaign needs"):
        active.run_campaign(
            pool,
            features,
            lambda: models.RandomForestModel(),
            batch_size=50,
            n_rounds=10,
            n_seed=40,
        )


def test_campaign_rejects_unknown_acquisition():
    pool, features = _pool()
    with pytest.raises(ValueError, match="unknown acquisition"):
        active.run_campaign(
            pool,
            features,
            lambda: models.RandomForestModel(n_estimators=20, seed=0),
            acquisition="thompson",
            batch_size=16,
            n_rounds=2,
            n_seed=32,
        )


def test_summarize_aggregates_across_seeds():
    pool, features = _pool()
    results = active.compare_acquisitions(
        pool,
        features,
        lambda: models.RandomForestModel(n_estimators=20, seed=0),
        acquisitions=("random", "greedy"),
        seeds=(0, 1),
        batch_size=16,
        n_rounds=2,
        n_seed=32,
    )
    summary = active.summarize(results, ["acquisition"])
    assert set(summary["acquisition"]) == {"random", "greedy"}
    assert (summary["n_seeds"] == 2).all()
    assert summary["actives_found_sd"].notna().any()
