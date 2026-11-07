"""Figures for the README.

Every figure is generated from a committed CSV in ``reports/``, so a figure
cannot disagree with the numbers it is drawn from.

Colour choices follow one rule: colour encodes what the series *is*. Assay
noise is an ordered magnitude, so it gets one hue stepped light to dark.
Acquisition rules are unordered identities, so they get distinct hues in a
fixed order. Every line is labelled directly as well as in the legend, so no
reading of these figures depends on telling two colours apart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parents[2] / "reports"
FIGURES = REPORTS / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: One hue, light to dark: assay noise is an ordered quantity.
SEQUENTIAL = ("#86b6ef", "#5598e7", "#2a78d6", "#184f95")

#: Fixed categorical order; never cycled, never reassigned by rank.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")

ACQUISITION_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "ucb": "UCB",
    "uncertainty": "Uncertainty",
    "diverse_greedy": "Diverse greedy",
}

SPLIT_LABELS = {
    "random": "Random\nsplit",
    "scaffold": "Scaffold\nsplit",
    "document": "Publication\nsplit",
    "external": "External\nset",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )
    return plt


def _despine(axes):
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(axis="x", visible=False)


def _label_line(axes, x, y, text, color, plt):
    """Direct label at the end of a line, so identity is never colour alone."""
    axes.annotate(
        text,
        xy=(x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9,
        color=color,
        fontweight="medium",
    )


def plot_assay_quality(path: Path | None = None) -> Path:
    """Actives found per compound assayed, at four levels of assay noise."""
    plt = _style()
    source = path or REPORTS / "campaign_assay_quality_summary.csv"
    frame = pd.read_csv(source).sort_values(["assay_sd", "round"])

    figure, axes = plt.subplots(figsize=(7.6, 4.6))
    noise_levels = sorted(frame["assay_sd"].unique())
    for position, sd in enumerate(noise_levels):
        subset = frame[frame["assay_sd"] == sd]
        color = SEQUENTIAL[min(position, len(SEQUENTIAL) - 1)]
        label = f"sd = {sd:.1f}"
        if abs(sd) < 1e-9:
            label += "  (noise-free)"
        axes.plot(
            subset["n_assayed"],
            subset["actives_found_mean"],
            color=color,
            marker="o",
            markersize=4,
            label=label,
        )
        axes.fill_between(
            subset["n_assayed"],
            subset["actives_found_mean"] - subset["actives_found_sd"].fillna(0),
            subset["actives_found_mean"] + subset["actives_found_sd"].fillna(0),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        last = subset.iloc[-1]
        _label_line(
            axes, last["n_assayed"], last["actives_found_mean"], f"{sd:.1f}", color, plt
        )

    axes.set_title("A weaker assay costs the campaign real hits")
    axes.set_xlabel("Compounds assayed")
    axes.set_ylabel("Potent compounds found (pActivity ≥ 8)")
    axes.legend(title="Inter-assay SD (log units)", title_fontsize=9, loc="upper left")
    _despine(axes)
    axes.margins(x=0.08)
    figure.text(
        0.01,
        -0.02,
        "Identical model and acquisition rule in every arm; only the simulated "
        "measurement error differs. Mean of 3 seeds, band = ±1 SD.",
        fontsize=8,
        color=INK_MUTED,
        ha="left",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "assay_quality.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


def plot_acquisition(path: Path | None = None) -> Path:
    """How the compound-selection rule changes campaign yield."""
    plt = _style()
    source = path or REPORTS / "campaign_acquisition_summary.csv"
    frame = pd.read_csv(source).sort_values(["acquisition", "round"])

    order = [
        name
        for name in ("random", "uncertainty", "diverse_greedy", "ucb", "greedy")
        if name in set(frame["acquisition"])
    ]

    figure, axes = plt.subplots(figsize=(7.6, 4.6))
    for position, name in enumerate(order):
        subset = frame[frame["acquisition"] == name]
        color = CATEGORICAL[position % len(CATEGORICAL)]
        axes.plot(
            subset["n_assayed"],
            subset["actives_found_mean"],
            color=color,
            marker="o",
            markersize=4,
            label=ACQUISITION_LABELS.get(name, name),
        )
        last = subset.iloc[-1]
        _label_line(
            axes,
            last["n_assayed"],
            last["actives_found_mean"],
            ACQUISITION_LABELS.get(name, name),
            color,
            plt,
        )

    axes.set_title("Choosing the next plate by model score, not at random")
    axes.set_xlabel("Compounds assayed")
    axes.set_ylabel("Potent compounds found (pActivity ≥ 8)")
    axes.legend(loc="upper left")
    _despine(axes)
    axes.margins(x=0.22)
    figure.text(
        0.01,
        -0.02,
        "Mean of 3 seeds. Measurement error held at the empirical estimate "
        "(SD 0.60 log units) in every arm.",
        fontsize=8,
        color=INK_MUTED,
        ha="left",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "acquisition.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


def plot_generalization(path: Path | None = None) -> Path:
    """The gap between a random split and a realistic one."""
    plt = _style()
    source = path or REPORTS / "benchmark.csv"
    frame = pd.read_csv(source)
    frame = frame[
        (frame["model"] == "gradient_boosting") & (frame["representation"] == "morgan")
    ]
    available = set(frame["split"])
    order = [
        name
        for name in ("random", "scaffold", "document", "external")
        if name in available
    ]
    frame = frame.set_index("split").loc[order].reset_index()

    figure, axes = plt.subplots(figsize=(7.6, 4.4))
    positions = range(len(frame))
    width = 0.36
    gap = 0.02  # 2px-equivalent surface gap between adjacent bars

    for offset, (column, label, color) in enumerate(
        [
            ("r2", "R² (absolute accuracy)", CATEGORICAL[0]),
            ("spearman", "Spearman ρ (rank order)", CATEGORICAL[1]),
        ]
    ):
        xs = [
            position + (offset - 0.5) * (width + gap) for position in positions
        ]
        axes.bar(xs, frame[column], width=width, color=color, label=label, linewidth=0)
        for x, value in zip(xs, frame[column], strict=False):
            axes.annotate(
                f"{value:.2f}",
                xy=(x, value),
                xytext=(0, 4 if value >= 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK_SECONDARY,
            )

    axes.axhline(0, color=AXIS, linewidth=1)
    axes.set_xticks(list(positions))
    axes.set_xticklabels([SPLIT_LABELS.get(name, name) for name in frame["split"]])
    axes.set_title("Rank order survives where absolute prediction does not")
    axes.set_ylabel("Score")
    axes.set_ylim(-0.5, 1.0)
    axes.legend(loc="upper right")
    _despine(axes)
    figure.text(
        0.01,
        -0.04,
        "XGBoost on Morgan counts. Left to right: increasingly realistic "
        "evaluation. A publication-level split holds out whole series together "
        "with their assay campaign.",
        fontsize=8,
        color=INK_MUTED,
        ha="left",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "generalization.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


def plot_noise_floor(
    benchmark_path: Path | None = None, assay_sd: float | None = None
) -> Path:
    """Model error against the reproducibility of the measurement itself."""
    plt = _style()
    import json

    frame = pd.read_csv(benchmark_path or REPORTS / "benchmark.csv")
    frame = frame[
        (frame["model"] == "gradient_boosting")
        & (frame["representation"] == "morgan")
        & frame["rmse"].notna()
        & (frame["split"] != "external")
    ]
    available = set(frame["split"])
    order = [name for name in ("random", "scaffold", "document") if name in available]
    frame = frame.set_index("split").loc[order].reset_index()

    if assay_sd is None:
        payload = json.loads((REPORTS / "assay_noise.json").read_text())
        assay_sd = payload["noise_estimate"]["recommended_sd"]

    figure, axes = plt.subplots(figsize=(6.6, 4.2))
    positions = list(range(len(frame)))
    axes.bar(positions, frame["rmse"], width=0.5, color=CATEGORICAL[0], linewidth=0)
    for position, value in zip(positions, frame["rmse"], strict=False):
        axes.annotate(
            f"{value:.2f}",
            xy=(position, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    axes.axhline(assay_sd, color=INK, linewidth=1.5, linestyle=(0, (4, 3)))
    axes.annotate(
        f"Measurement floor: inter-assay SD {assay_sd:.2f}",
        xy=(len(frame) - 0.5, assay_sd),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
        fontweight="medium",
    )

    axes.set_xticks(positions)
    axes.set_xticklabels([SPLIT_LABELS.get(name, name) for name in frame["split"]])
    axes.set_title("On known chemistry the model is already as good as the assay")
    axes.set_ylabel("RMSE (log units)")
    _despine(axes)
    figure.text(
        0.01,
        -0.04,
        "A bar at or below the dashed line means the model's error is "
        "indistinguishable from repeating the experiment. Floor estimated from "
        "170 compounds measured in more than one assay.",
        fontsize=8,
        color=INK_MUTED,
        ha="left",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "noise_floor.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


def plot_all() -> list[Path]:
    produced = []
    for name, function in (
        ("assay_quality", plot_assay_quality),
        ("acquisition", plot_acquisition),
        ("generalization", plot_generalization),
        ("noise_floor", plot_noise_floor),
    ):
        try:
            produced.append(function())
        except FileNotFoundError as error:
            print(f"skipped {name}: {error}")
    return produced
