"""A retrospective simulation of an iterative screening campaign.

The question this answers is not "how accurate is the model" but "how many
plates does it take to find the potent compounds, and how much does that
depend on the quality of the assay feeding it".

The simulation treats a held-out set of A2A ligands as an unscreened deck.
Their measured potencies are the ground truth and are hidden from the model.
Each round, the model is retrained on everything assayed so far, scores the
remaining deck, and a batch of compounds is selected and "assayed" through
:class:`a2a_qsar.assay.AssaySimulator`, which adds inter-assay noise and
censors anything outside the dynamic range. The model therefore learns from
the same imperfect numbers a real campaign would hand it, while scoring is
done against the truth.

Because the only difference between arms can be the acquisition rule or the
assay noise, the comparison isolates exactly one effect at a time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .assay import AssaySimulator
from .features import murcko_scaffold

LOGGER = logging.getLogger(__name__)

ACQUISITION_FUNCTIONS = (
    "random",
    "greedy",
    "ucb",
    "uncertainty",
    "diverse_greedy",
)


@dataclass
class RoundResult:
    """One iteration of the campaign."""

    round_index: int
    n_assayed: int
    n_true_actives_found: int
    n_called_active: int
    n_false_calls: int
    best_true_pactivity: float
    mean_true_pactivity_of_batch: float
    n_scaffolds_assayed: int

    def to_dict(self) -> dict:
        return {
            "round": self.round_index,
            "n_assayed": self.n_assayed,
            "n_true_actives_found": self.n_true_actives_found,
            "n_called_active": self.n_called_active,
            "n_false_calls": self.n_false_calls,
            "best_true_pactivity": self.best_true_pactivity,
            "mean_true_pactivity_of_batch": self.mean_true_pactivity_of_batch,
            "n_scaffolds_assayed": self.n_scaffolds_assayed,
        }


@dataclass
class CampaignResult:
    acquisition: str
    assay_sd: float
    seed: int
    rounds: list[RoundResult] = field(default_factory=list)
    n_pool: int = 0
    n_pool_actives: int = 0

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([entry.to_dict() for entry in self.rounds])
        frame.insert(0, "acquisition", self.acquisition)
        frame.insert(1, "assay_sd", self.assay_sd)
        frame.insert(2, "seed", self.seed)
        frame["recall_of_pool_actives"] = (
            frame["n_true_actives_found"] / self.n_pool_actives
            if self.n_pool_actives
            else np.nan
        )
        frame["fraction_of_pool_assayed"] = frame["n_assayed"] / self.n_pool
        return frame


def _select(
    acquisition: str,
    predictions: np.ndarray,
    spread: np.ndarray,
    candidate_rows: np.ndarray,
    batch_size: int,
    scaffolds: np.ndarray,
    assayed_scaffolds: set[str],
    rng: np.random.Generator,
    beta: float = 1.0,
) -> np.ndarray:
    """Choose which candidates to put on the next plate."""
    if acquisition == "random":
        return rng.choice(candidate_rows, size=batch_size, replace=False)

    if acquisition == "greedy":
        scores = predictions
    elif acquisition == "ucb":
        # Upper confidence bound: exploit the prediction, but pay for being
        # unsure, which is what lets the campaign leave a saturated series.
        scores = predictions + beta * spread
    elif acquisition == "uncertainty":
        scores = spread
    elif acquisition == "diverse_greedy":
        # Greedy, but a scaffold already on a plate is penalised, so one
        # series cannot consume the whole batch. This is the rule a project
        # team actually applies when filling a plate.
        penalty = np.array(
            [0.5 if scaffold in assayed_scaffolds else 0.0 for scaffold in scaffolds]
        )
        scores = predictions - penalty
    else:
        raise ValueError(
            f"unknown acquisition {acquisition!r}; expected one of {ACQUISITION_FUNCTIONS}"
        )

    if acquisition == "diverse_greedy":
        chosen: list[int] = []
        chosen_scaffolds: set[str] = set()
        order = np.argsort(-scores)
        for position in order:
            scaffold = scaffolds[position]
            if scaffold in chosen_scaffolds and len(chosen) < batch_size:
                continue
            chosen.append(candidate_rows[position])
            chosen_scaffolds.add(scaffold)
            if len(chosen) == batch_size:
                break
        if len(chosen) < batch_size:  # deck exhausted of fresh scaffolds
            remaining = [row for row in candidate_rows[order] if row not in set(chosen)]
            chosen.extend(remaining[: batch_size - len(chosen)])
        return np.asarray(chosen)

    return candidate_rows[np.argsort(-scores)[:batch_size]]


def run_campaign(
    pool: pd.DataFrame,
    features: np.ndarray,
    model_factory,
    acquisition: str = "ucb",
    assay_sd: float = 0.6,
    batch_size: int = 32,
    n_rounds: int = 10,
    n_seed: int = 48,
    active_threshold: float = 8.0,
    seed: int = 0,
    beta: float = 1.0,
) -> CampaignResult:
    """Run one campaign arm.

    Parameters
    ----------
    pool:
        Frame with a ``pactivity`` column (ground truth, hidden from the
        model) and a ``smiles`` column.
    features:
        Feature matrix aligned row-for-row with ``pool``.
    model_factory:
        Zero-argument callable returning a fresh unfitted model. A fresh model
        each round avoids carrying state between rounds, which would make the
        arms incomparable.
    batch_size:
        Compounds per round. 32 approximates a dose-response plate layout.
    """
    if len(pool) != len(features):
        raise ValueError(f"pool has {len(pool)} rows but features has {len(features)}")
    if n_seed + n_rounds * batch_size > len(pool):
        raise ValueError(
            f"campaign needs {n_seed + n_rounds * batch_size} compounds "
            f"but the pool has {len(pool)}"
        )

    truth = pool["pactivity"].to_numpy(dtype=float)
    scaffolds_all = np.asarray([murcko_scaffold(smi) for smi in pool["smiles"]])
    truly_active = truth >= active_threshold

    simulator = AssaySimulator(sd=assay_sd, seed=seed)
    rng = np.random.default_rng(seed)

    assayed = np.zeros(len(pool), dtype=bool)
    observed = np.full(len(pool), np.nan)

    seed_rows = rng.choice(len(pool), size=n_seed, replace=False)
    seed_observed, _ = simulator.measure(truth[seed_rows])
    assayed[seed_rows] = True
    observed[seed_rows] = seed_observed

    result = CampaignResult(
        acquisition=acquisition,
        assay_sd=assay_sd,
        seed=seed,
        n_pool=len(pool),
        n_pool_actives=int(truly_active.sum()),
    )

    def record(round_index: int, batch_rows: np.ndarray) -> None:
        called = observed[assayed] >= active_threshold
        found = truly_active[assayed]
        result.rounds.append(
            RoundResult(
                round_index=round_index,
                n_assayed=int(assayed.sum()),
                n_true_actives_found=int(found.sum()),
                n_called_active=int(called.sum()),
                n_false_calls=int((called & ~found).sum()),
                best_true_pactivity=float(truth[assayed].max()),
                mean_true_pactivity_of_batch=float(truth[batch_rows].mean()),
                n_scaffolds_assayed=int(len(set(scaffolds_all[assayed]))),
            )
        )

    record(0, seed_rows)

    for round_index in range(1, n_rounds + 1):
        candidate_rows = np.flatnonzero(~assayed)
        if len(candidate_rows) <= batch_size:
            LOGGER.info("deck exhausted at round %d", round_index)
            break

        model = model_factory()
        model.fit(features[assayed], observed[assayed])
        predictions, spread = model.predict_with_uncertainty(features[candidate_rows])

        batch_rows = _select(
            acquisition=acquisition,
            predictions=predictions,
            spread=spread,
            candidate_rows=candidate_rows,
            batch_size=batch_size,
            scaffolds=scaffolds_all[candidate_rows],
            assayed_scaffolds=set(scaffolds_all[assayed]),
            rng=rng,
            beta=beta,
        )

        batch_observed, _ = simulator.measure(truth[batch_rows])
        assayed[batch_rows] = True
        observed[batch_rows] = batch_observed
        record(round_index, batch_rows)

    return result


def compare_acquisitions(
    pool: pd.DataFrame,
    features: np.ndarray,
    model_factory,
    acquisitions: tuple[str, ...] = ACQUISITION_FUNCTIONS,
    assay_sd: float = 0.6,
    seeds: tuple[int, ...] = (0, 1, 2),
    **kwargs,
) -> pd.DataFrame:
    """Run every acquisition rule over several seeds and stack the results."""
    frames = []
    for acquisition in acquisitions:
        for seed in seeds:
            campaign = run_campaign(
                pool,
                features,
                model_factory,
                acquisition=acquisition,
                assay_sd=assay_sd,
                seed=seed,
                **kwargs,
            )
            frames.append(campaign.to_frame())
            LOGGER.info(
                "%s seed=%d: %d/%d actives after %d assayed",
                acquisition,
                seed,
                campaign.rounds[-1].n_true_actives_found,
                campaign.n_pool_actives,
                campaign.rounds[-1].n_assayed,
            )
    return pd.concat(frames, ignore_index=True)


def compare_assay_quality(
    pool: pd.DataFrame,
    features: np.ndarray,
    model_factory,
    assay_sds: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0),
    acquisition: str = "ucb",
    seeds: tuple[int, ...] = (0, 1, 2),
    **kwargs,
) -> pd.DataFrame:
    """Hold the acquisition rule fixed and vary only the assay noise.

    This is the headline experiment: everything about the AI is identical
    across arms, so any difference in how fast the campaign finds potent
    compounds is attributable to data quality alone.
    """
    frames = []
    for assay_sd in assay_sds:
        for seed in seeds:
            campaign = run_campaign(
                pool,
                features,
                model_factory,
                acquisition=acquisition,
                assay_sd=assay_sd,
                seed=seed,
                **kwargs,
            )
            frames.append(campaign.to_frame())
            LOGGER.info(
                "sd=%.2f seed=%d: %d/%d actives",
                assay_sd,
                seed,
                campaign.rounds[-1].n_true_actives_found,
                campaign.n_pool_actives,
            )
    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Mean and spread across seeds at each round."""
    return (
        results.groupby(group_columns + ["round"])
        .agg(
            n_assayed=("n_assayed", "mean"),
            actives_found_mean=("n_true_actives_found", "mean"),
            actives_found_sd=("n_true_actives_found", "std"),
            recall_mean=("recall_of_pool_actives", "mean"),
            best_pactivity_mean=("best_true_pactivity", "mean"),
            false_calls_mean=("n_false_calls", "mean"),
            scaffolds_mean=("n_scaffolds_assayed", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
