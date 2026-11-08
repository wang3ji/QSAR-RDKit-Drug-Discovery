"""Figures for the README.

Every figure is generated from a committed CSV in ``reports/``, so a figure
cannot disagree with the numbers it is drawn from.

Colour encodes what a series *is*. Assay noise is an ordered magnitude, so it
gets one hue stepped light to dark. Acquisition rules are unordered
identities, so they get distinct hues in a fixed order. Every line carries a
direct label as well as a legend entry, so no reading of these figures depends
on telling two colours apart.
"""

from __future__ import annotations

import json
import textwrap
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
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
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


def _despine(axes) -> None:
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(axis="x", visible=False)


def _caption(figure, text: str, width: int = 94) -> None:
    """A wrapped note under the axes, in muted ink."""
    figure.text(
        0.01,
        -0.02,
        "\n".join(textwrap.wrap(text, width)),
        fontsize=8,
        color=INK_MUTED,
        ha="left",
        va="top",
    )


def _end_labels(axes, entries) -> None:
    """Right-margin labels, spread by rank so tightly bunched lines stay legible.

    Values at the end of these campaigns sit within a few percent of each
    other, so labels anchored to the data points would overlap. Ranking them
    down the right margin keeps every series named in its own colour, which is
    what stops identity resting on colour alone.
    """
    if not entries:
        return
    ordered = sorted(entries, key=lambda item: -item[1])
    step = 0.075
    top = 0.5 + step * (len(ordered) - 1) / 2
    for position, (_, _, text, color) in enumerate(ordered):
        axes.annotate(
            text,
            xy=(1.01, top - position * step),
            xycoords="axes fraction",
            va="center",
            ha="left",
            fontsize=9,
            color=color,
            annotation_clip=False,
        )


def plot_assay_quality(path: Path | None = None) -> Path:
    """Campaign yield at four levels of simulated assay noise."""
    plt = _style()
    source = path or REPORTS / "campaign_assay_quality_summary.csv"
    frame = pd.read_csv(source).sort_values(["assay_sd", "round"])
    n_seeds = int(frame["n_seeds"].max())

    figure, axes = plt.subplots(figsize=(7.8, 4.6))
    noise_levels = sorted(frame["assay_sd"].unique())
    entries = []
    for position, sd in enumerate(noise_levels):
        subset = frame[frame["assay_sd"] == sd]
        color = SEQUENTIAL[min(position, len(SEQUENTIAL) - 1)]
        label = f"SD {sd:.1f}" + ("  (noise-free)" if abs(sd) < 1e-9 else "")
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
            alpha=0.13,
            linewidth=0,
        )
        last = subset.iloc[-1]
        entries.append(
            (last["n_assayed"], last["actives_found_mean"], f"SD {sd:.1f}", color)
        )

    axes.set_title("Going from a perfect assay to a poor one costs 8% of the hits")
    axes.set_xlabel("Compounds assayed")
    axes.set_ylabel("Potent compounds found (pActivity ≥ 8)")
    axes.legend(title="Inter-assay SD (log units)", title_fontsize=9, loc="upper left")
    _despine(axes)
    axes.set_xlim(left=30)
    axes.margins(x=0.03)
    _end_labels(axes, entries)
    _caption(
        figure,
        "Identical model and acquisition rule in every arm; only the simulated "
        f"measurement error differs. Mean of {n_seeds} seeds, band = ±1 SD. Arms "
        "share seeds, so they are compared paired: SD 0.0 against SD 1.0 is "
        "17.8 compounds (Wilcoxon p = 0.014). Adjacent arms are not separable "
        "at this number of seeds.",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "assay_quality.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


def plot_false_calls(path: Path | None = None) -> Path:
    """Compounds wrongly believed active, by assay noise."""
    plt = _style()
    source = path or REPORTS / "campaign_assay_quality_summary.csv"
    frame = pd.read_csv(source)
    final = frame[frame["round"] == frame["round"].max()].sort_values("assay_sd")
    n_seeds = int(frame["n_seeds"].max())

    figure, axes = plt.subplots(figsize=(6.8, 4.0))
    positions = list(range(len(final)))
    colors = [SEQUENTIAL[min(index, len(SEQUENTIAL) - 1)] for index in positions]
    axes.bar(positions, final["false_calls_mean"], width=0.52, color=colors, linewidth=0)
    for position, value in zip(positions, final["false_calls_mean"], strict=True):
        axes.annotate(
            f"{value:.0f}",
            xy=(position, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=INK_SECONDARY,
        )

    axes.set_xticks(positions)
    axes.set_xticklabels([f"{sd:.1f}" for sd in final["assay_sd"]])
    axes.set_xlabel("Inter-assay SD (log units)")
    axes.set_ylabel("Compounds wrongly called active")
    axes.set_title("What assay noise reliably produces is false confidence")
    axes.set_ylim(0, max(final["false_calls_mean"].max() * 1.25, 1))
    _despine(axes)
    _caption(
        figure,
        f"Out of {int(final['n_assayed'].iloc[0])} compounds assayed per campaign, "
        f"mean of {n_seeds} seeds. Each of these was measured at or above the "
        "potency threshold without actually being potent, so it consumes "
        "follow-up before failing confirmation.",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "false_calls.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


def plot_acquisition(path: Path | None = None) -> Path:
    """How the compound-selection rule changes campaign yield."""
    plt = _style()
    source = path or REPORTS / "campaign_acquisition_summary.csv"
    frame = pd.read_csv(source).sort_values(["acquisition", "round"])
    n_seeds = int(frame["n_seeds"].max())

    available = set(frame["acquisition"])
    order = [
        name
        for name in ("random", "uncertainty", "diverse_greedy", "ucb", "greedy")
        if name in available
    ]

    figure, axes = plt.subplots(figsize=(7.8, 4.6))
    entries = []
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
        entries.append(
            (
                last["n_assayed"],
                last["actives_found_mean"],
                ACQUISITION_LABELS.get(name, name),
                color,
            )
        )

    axes.set_title("Choosing the next plate by model score, not at random")
    axes.set_xlabel("Compounds assayed")
    axes.set_ylabel("Potent compounds found (pActivity ≥ 8)")
    axes.legend(loc="upper left")
    _despine(axes)
    axes.set_xlim(left=30)
    axes.margins(x=0.03)
    _end_labels(axes, entries)
    _caption(
        figure,
        f"Mean of {n_seeds} seeds. Measurement error held at the empirical "
        "estimate (SD 0.60 log units) in every arm, so the only thing that "
        "differs between these curves is which compounds were chosen.",
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

    figure, axes = plt.subplots(figsize=(7.4, 4.3))
    positions = range(len(frame))
    width = 0.36
    gap = 0.02  # a visible surface gap between adjacent bars

    for offset, (column, label, color) in enumerate(
        [
            ("r2", "R² (absolute accuracy)", CATEGORICAL[0]),
            ("spearman", "Spearman ρ (rank order)", CATEGORICAL[1]),
        ]
    ):
        xs = [position + (offset - 0.5) * (width + gap) for position in positions]
        axes.bar(xs, frame[column], width=width, color=color, label=label, linewidth=0)
        for x, value in zip(xs, frame[column], strict=True):
            text = "0.00" if abs(value) < 0.005 else f"{value:.2f}"
            axes.annotate(
                text,
                xy=(x, max(value, 0.0)),
                xytext=(0, 5),
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
    lowest = min(frame["r2"].min(), frame["spearman"].min(), 0.0)
    axes.set_ylim(min(lowest * 1.4, -0.08), 1.05)
    axes.legend(loc="upper right")
    _despine(axes)
    _caption(
        figure,
        "XGBoost on Morgan counts. Left to right: increasingly realistic "
        "evaluation. A publication-level split holds out whole series together "
        "with the assay campaign that measured them.",
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

    figure, axes = plt.subplots(figsize=(6.8, 4.2))
    positions = list(range(len(frame)))
    axes.bar(positions, frame["rmse"], width=0.5, color=CATEGORICAL[0], linewidth=0)
    # Values sit inside the bars so the space above stays clear for the
    # reference line, which is the thing this figure is actually about.
    for position, value in zip(positions, frame["rmse"], strict=True):
        axes.annotate(
            f"{value:.2f}",
            xy=(position, value),
            xytext=(0, -8),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10,
            color="#ffffff",
            fontweight="bold",
        )

    axes.set_ylim(0, frame["rmse"].max() * 1.3)
    axes.axhline(assay_sd, color=INK, linewidth=1.5, linestyle=(0, (4, 3)))
    # Anchored in axes fraction horizontally and data units vertically, so the
    # label always sits on the line wherever the bars happen to land.
    axes.text(
        0.015,
        assay_sd + frame["rmse"].max() * 0.02,
        f"Measurement floor\n(inter-assay SD {assay_sd:.2f})",
        transform=axes.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9,
        linespacing=1.4,
        color=INK,
    )

    axes.set_xticks(positions)
    axes.set_xticklabels([SPLIT_LABELS.get(name, name) for name in frame["split"]])
    axes.set_title("On known chemistry the model is already as good as the assay")
    axes.set_ylabel("RMSE (log units)")
    _despine(axes)
    _caption(
        figure,
        "A bar at or below the dashed line means the model's error is "
        "indistinguishable from repeating the experiment, so no change of "
        "algorithm improves it. Floor estimated from 170 compounds measured in "
        "more than one assay.",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "noise_floor.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


def plot_all() -> list[Path]:
    produced = []
    for name, function in (
        ("noise_floor", plot_noise_floor),
        ("generalization", plot_generalization),
        ("acquisition", plot_acquisition),
        ("assay_quality", plot_assay_quality),
        ("false_calls", plot_false_calls),
    ):
        try:
            produced.append(function())
        except FileNotFoundError as error:
            print(f"skipped {name}: {error}")
    return produced
