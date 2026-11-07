"""Command-line entry points.

Every number in the README comes from one of these commands, so a reader can
reproduce the whole study without opening a notebook::

    a2a-qsar curate
    a2a-qsar benchmark
    a2a-qsar assay-noise
    a2a-qsar campaign --experiment assay-quality
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import active, assay, data, evaluate, features, models, splits

REPORTS = Path(__file__).resolve().parents[2] / "reports"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {path}")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"wrote {path} ({len(frame)} rows)")


def cmd_curate(args: argparse.Namespace) -> int:
    modelling, pool, reports = data.build_dataset()
    _write_csv(modelling, PROCESSED / "modelling_set.csv")
    _write_csv(pool, PROCESSED / "screening_pool.csv")
    _write_json(
        {
            "target": data.TARGET_CHEMBL_ID,
            "uniprot": data.TARGET_UNIPROT,
            "modelling_set_compounds": len(modelling),
            "screening_pool_compounds": len(pool),
            "curation": [report.to_dict() for report in reports],
        },
        REPORTS / "curation.json",
    )
    return 0


def cmd_assay_noise(args: argparse.Namespace) -> int:
    modelling, pool, _ = data.build_dataset()
    estimate = assay.estimate_noise(modelling)
    print(json.dumps(estimate.to_dict(), indent=2))

    confirmation = {}
    for label, sd in (
        ("perfect", 0.0),
        ("excellent", 0.3),
        ("empirical", estimate.recommended_sd),
        ("poor", 1.0),
    ):
        simulator = assay.AssaySimulator(sd=sd, seed=0)
        confirmation[label] = assay.hit_confirmation_rate(
            pool["pactivity"].to_numpy(),
            simulator,
            threshold=args.active_threshold,
            n_repeats=args.repeats,
        )

    _write_json(
        {"noise_estimate": estimate.to_dict(), "hit_confirmation": confirmation},
        REPORTS / "assay_noise.json",
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    modelling, pool, _ = data.build_dataset()
    estimate = assay.estimate_noise(modelling)
    logging.info("assay sd estimate: %.3f", estimate.recommended_sd)

    rows = []
    details = {}
    for representation in args.representations:
        matrix = features.featurize(modelling["smiles"], representation=representation)
        pool_matrix = features.featurize(pool["smiles"], representation=representation)

        for split in splits.all_splits(modelling, test_fraction=0.2, seed=args.seed):
            for model_name in args.models:
                model = models.build_model(model_name, seed=args.seed)
                wrapped = models.ConformalRegressor(
                    base=model, confidence=args.confidence, seed=args.seed
                )
                wrapped.fit(
                    matrix.values[split.train_index],
                    modelling["pactivity"].to_numpy()[split.train_index],
                )
                y_true = modelling["pactivity"].to_numpy()[split.test_index]
                prediction, lower, upper = wrapped.predict_interval(
                    matrix.values[split.test_index]
                )
                _, spread = wrapped.predict_with_uncertainty(
                    matrix.values[split.test_index]
                )

                metrics = evaluate.regression_metrics(
                    y_true,
                    prediction,
                    assay_sd=estimate.recommended_sd,
                    active_threshold=args.active_threshold,
                )
                coverage = evaluate.interval_coverage(y_true, lower, upper)
                row = {
                    "representation": representation,
                    "split": split.name,
                    "model": model_name,
                    "n_train": split.sizes[0],
                    "n_test": split.sizes[1],
                    **metrics.to_dict(),
                    **coverage,
                    "nominal_coverage": args.confidence,
                }
                rows.append(row)
                key = f"{representation}|{split.name}|{model_name}"
                details[key] = evaluate.uncertainty_is_informative(
                    y_true, prediction, spread
                )
                print(
                    f"{key}: R2={metrics.r2:.3f} RMSE={metrics.rmse:.3f} "
                    f"rho={metrics.spearman:.3f} coverage={coverage['coverage']:.2f}"
                )

        # External validation: train on everything, test on the independent set.
        for model_name in args.models:
            model = models.build_model(model_name, seed=args.seed)
            model.fit(matrix.values, modelling["pactivity"].to_numpy())
            prediction = model.predict(pool_matrix.values)
            metrics = evaluate.regression_metrics(
                pool["pactivity"].to_numpy(),
                prediction,
                assay_sd=estimate.recommended_sd,
                active_threshold=args.active_threshold,
            )
            rows.append(
                {
                    "representation": representation,
                    "split": "external",
                    "model": model_name,
                    "n_train": len(modelling),
                    "n_test": len(pool),
                    **metrics.to_dict(),
                    "coverage": None,
                    "mean_interval_width": None,
                    "median_interval_width": None,
                    "nominal_coverage": None,
                }
            )
            print(
                f"{representation}|external|{model_name}: R2={metrics.r2:.3f} "
                f"rho={metrics.spearman:.3f} EF5%={metrics.enrichment_top_5pct:.2f}"
            )

    frame = pd.DataFrame(rows)
    _write_csv(frame, REPORTS / "benchmark.csv")
    _write_json(
        {
            "assay_sd": estimate.recommended_sd,
            "uncertainty_diagnostics": details,
        },
        REPORTS / "benchmark_diagnostics.json",
    )
    return 0


def cmd_campaign(args: argparse.Namespace) -> int:
    _, pool, _ = data.build_dataset()
    matrix = features.featurize(pool["smiles"], representation=args.representation)

    def factory():
        return models.build_model(args.model, seed=args.seed)

    common = dict(
        batch_size=args.batch_size,
        n_rounds=args.rounds,
        n_seed=args.seed_compounds,
        active_threshold=args.active_threshold,
    )
    seeds = tuple(range(args.replicates))

    if args.experiment in ("acquisition", "both"):
        results = active.compare_acquisitions(
            pool, matrix.values, factory, assay_sd=args.assay_sd, seeds=seeds, **common
        )
        _write_csv(results, REPORTS / "campaign_acquisition.csv")
        summary = active.summarize(results, ["acquisition"])
        _write_csv(summary, REPORTS / "campaign_acquisition_summary.csv")
        print(
            summary[summary["round"] == summary["round"].max()][
                ["acquisition", "n_assayed", "actives_found_mean", "recall_mean"]
            ].to_string(index=False)
        )

    if args.experiment in ("assay-quality", "both"):
        results = active.compare_assay_quality(
            pool,
            matrix.values,
            factory,
            assay_sds=tuple(args.assay_sds),
            acquisition=args.acquisition,
            seeds=seeds,
            **common,
        )
        _write_csv(results, REPORTS / "campaign_assay_quality.csv")
        summary = active.summarize(results, ["assay_sd"])
        _write_csv(summary, REPORTS / "campaign_assay_quality_summary.csv")
        print(
            summary[summary["round"] == summary["round"].max()][
                [
                    "assay_sd",
                    "n_assayed",
                    "actives_found_mean",
                    "actives_found_sd",
                    "recall_mean",
                    "false_calls_mean",
                ]
            ].to_string(index=False)
        )

    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from . import plots

    produced = plots.plot_all()
    if not produced:
        print("no figures produced; run 'curate', 'benchmark' and 'campaign' first")
        return 1
    for path in produced:
        print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a2a-qsar",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=8.0,
        help="pActivity at or above which a compound counts as active (default: 8.0 = 10 nM)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    curate = subparsers.add_parser(
        "curate", help="curate the snapshots and write processed data"
    )
    curate.set_defaults(func=cmd_curate)

    noise = subparsers.add_parser(
        "assay-noise", help="estimate inter-assay variability and its effect on hit calling"
    )
    noise.add_argument("--repeats", type=int, default=200)
    noise.set_defaults(func=cmd_assay_noise)

    benchmark = subparsers.add_parser(
        "benchmark", help="train and evaluate across representations, splits and models"
    )
    benchmark.add_argument(
        "--representations", nargs="+", default=["morgan", "combined"]
    )
    benchmark.add_argument("--models", nargs="+", default=list(models.MODEL_NAMES))
    benchmark.add_argument("--confidence", type=float, default=0.8)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.set_defaults(func=cmd_benchmark)

    campaign = subparsers.add_parser(
        "campaign", help="simulate an iterative screening campaign"
    )
    campaign.add_argument(
        "--experiment",
        choices=["acquisition", "assay-quality", "both"],
        default="both",
    )
    campaign.add_argument("--model", default="random_forest", choices=models.MODEL_NAMES)
    campaign.add_argument("--representation", default="morgan")
    campaign.add_argument("--acquisition", default="ucb", choices=active.ACQUISITION_FUNCTIONS)
    campaign.add_argument("--assay-sd", type=float, default=0.6)
    campaign.add_argument(
        "--assay-sds", nargs="+", type=float, default=[0.0, 0.3, 0.6, 1.0]
    )
    campaign.add_argument("--batch-size", type=int, default=32)
    campaign.add_argument("--rounds", type=int, default=10)
    campaign.add_argument("--seed-compounds", type=int, default=48)
    campaign.add_argument("--replicates", type=int, default=3)
    campaign.add_argument("--seed", type=int, default=42)
    campaign.set_defaults(func=cmd_campaign)

    figures = subparsers.add_parser(
        "figures", help="render the README figures from the reports/ CSVs"
    )
    figures.set_defaults(func=cmd_figures)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    np.seterr(all="ignore")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
