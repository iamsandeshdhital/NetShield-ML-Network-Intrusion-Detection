"""Figure generation for the NetShield-ML report.

All figures are rendered against one light surface with a fixed categorical
palette assigned by slot, never cycled, so a model keeps the same colour in
every chart.  Direct labels are applied selectively - the leader, the extreme,
or the value the axis cannot carry - and the exact numbers behind every figure
are written to ``results/tables/`` as CSV, which is also what satisfies the
contrast-relief requirement for the third (aqua) slot.

Authors: Anurag Jha, Sandesh Dhital
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from src import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
GRID = "#e6e5e1"

# Categorical slots, in fixed order. Slots 1-3 clear the all-pairs CVD and
# normal-vision separation floors in light mode, which is why the comparison
# stays at three models rather than cycling a fourth hue.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

# Single blue hue, light to dark, for continuous magnitude.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "netshield_blue",
    ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"],
)

BASE_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.titlecolor": TEXT_PRIMARY,
    "text.color": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "legend.frameon": False,
    "figure.dpi": config.FIGURE_DPI,
}


def _apply_style() -> None:
    plt.rcParams.update(BASE_RC)


def _save(fig: plt.Figure, name: str) -> Path:
    path = config.FIGURES_DIR / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(path, bbox_inches="tight", dpi=config.FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved figure %s", path.name)
    return path


def _round_bar_ends(ax, bars, radius_px: float = 4.0,
                    horizontal: bool = False) -> None:
    """Redraw bars rounded only at the data end.

    A rounded box replaces each bar and a square patch is laid over the
    baseline end, so the mark grows out of the axis rather than floating above
    it.  The radius is given in reference (96 dpi) pixels and rescaled to the
    figure's own resolution, so the corner looks the same on a 200 dpi export
    as it would on screen.
    """
    fig = ax.figure
    fig.canvas.draw()
    radius_px = radius_px * fig.dpi / 96.0

    for bar in bars:
        x, y = bar.get_xy()
        width, height = bar.get_width(), bar.get_height()
        if width <= 0 or height <= 0:
            continue

        colour = bar.get_facecolor()

        origin = ax.transData.transform((0, 0))
        shifted = ax.transData.inverted().transform(
            (origin[0] + radius_px, origin[1] + radius_px)
        )
        base = ax.transData.inverted().transform(origin)
        radius_x = abs(shifted[0] - base[0])
        radius_y = abs(shifted[1] - base[1])

        radius = min(radius_x, radius_y, width / 2, height / 2)
        if radius <= 0:
            continue

        bar.set_visible(False)

        rounded = FancyBboxPatch(
            (x + radius, y + radius),
            max(width - 2 * radius, 1e-12),
            max(height - 2 * radius, 1e-12),
            boxstyle=f"round,pad={radius},rounding_size={radius}",
            linewidth=0,
            facecolor=colour,
            mutation_aspect=1,
            zorder=2,
        )
        ax.add_patch(rounded)

        if horizontal:
            cap = Rectangle((x, y), min(radius * 2, width), height,
                            linewidth=0, facecolor=colour, zorder=3)
        else:
            cap = Rectangle((x, y), width, min(radius * 2, height),
                            linewidth=0, facecolor=colour, zorder=3)
        ax.add_patch(cap)


def _swatch_legend(ax, labels: list[str], colours: list[str],
                   marker: bool = False) -> None:
    """Place a legend above the plot using proxy handles.

    ``_round_bar_ends`` hides the original bar artists, which would otherwise
    leave the legend with labels but no colour swatches - identity carried by
    text alone.  Proxy handles keep the swatch and the label together, and
    seating the legend above the axes stops it colliding with tall marks.
    """
    if marker:
        handles = [
            Line2D([], [], linestyle="none", marker="o", markersize=7,
                   markerfacecolor=c, markeredgecolor=SURFACE, markeredgewidth=1.4,
                   label=label)
            for label, c in zip(labels, colours)
        ]
    else:
        handles = [Patch(facecolor=c, edgecolor="none", label=label)
                   for label, c in zip(labels, colours)]

    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncols=min(len(labels), 4), fontsize=8, handlelength=1.1,
              handleheight=1.1, columnspacing=1.4, borderaxespad=0)


# --------------------------------------------------------------------------
# Exploratory figures
# --------------------------------------------------------------------------
def plot_class_distribution(train: pd.DataFrame, test: pd.DataFrame) -> Path:
    """Attack-category composition of the two official splits."""
    _apply_style()

    order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    train_counts = train["attack_category"].value_counts().reindex(order, fill_value=0)
    test_counts = test["attack_category"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    positions = np.arange(len(order), dtype=float)
    bar_width = 0.30
    gap = 0.02  # surface gap between adjacent fills

    left = ax.bar(positions - (bar_width + gap) / 2, train_counts.to_numpy(),
                  bar_width, label="KDDTrain+", color=SERIES[0])
    right = ax.bar(positions + (bar_width + gap) / 2, test_counts.to_numpy(),
                   bar_width, label="KDDTest+", color=SERIES[1])

    ax.set_yscale("symlog")
    ax.set_ylim(0, max(train_counts.max(), test_counts.max()) * 4)

    for bars, counts in ((left, train_counts), (right, test_counts)):
        for bar, value in zip(bars, counts.to_numpy()):
            ax.annotate(f"{value:,}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7.5, color=TEXT_SECONDARY)

    _round_bar_ends(ax, list(left) + list(right))

    ax.set_xticks(positions)
    ax.set_xticklabels(order)
    ax.set_ylabel("Records (log scale)")
    # Counts span four orders of magnitude, so the axis is logarithmic and
    # cannot be read precisely; the value labels carry what it cannot.
    ax.set_title("Traffic composition of the NSL-KDD splits", pad=30, fontsize=11)
    ax.grid(axis="x", visible=False)
    _swatch_legend(ax, ["KDDTrain+", "KDDTest+"], [SERIES[0], SERIES[1]])

    return _save(fig, "01_class_distribution")


# --------------------------------------------------------------------------
# Model comparison
# --------------------------------------------------------------------------
def plot_model_comparison(metrics: pd.DataFrame, split_name: str,
                          filename: str) -> Path:
    """Cleveland dot plot comparing the three models across four measures.

    Dots rather than bars: the scores sit within a few points of one another,
    and zero-based bars of that data are visually identical.  A dot encodes
    position rather than length, so the axis may legitimately start above zero
    and show the differences that actually separate the models.  Exact values
    live in ``results/tables/03_model_metrics.csv`` and in the report table, so
    only the leading model on each row is direct-labelled here.
    """
    _apply_style()

    subset = metrics[metrics["Split"] == split_name]
    measures = ["Accuracy", "Precision", "Recall", "F1"]
    model_names = list(subset["Model"])

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    rows = np.arange(len(measures), dtype=float)[::-1]

    values_by_model = {
        model: [float(subset[subset["Model"] == model].iloc[0][m]) for m in measures]
        for model in model_names
    }

    low = min(min(v) for v in values_by_model.values())
    high = max(max(v) for v in values_by_model.values())
    span = max(high - low, 0.02)
    left, right = max(0.0, low - span * 0.35), min(1.0, high + span * 0.45)

    # Connector rule joining each row's dots, so the spread reads at a glance.
    for row, measure in zip(rows, measures):
        row_values = [values_by_model[m][measures.index(measure)] for m in model_names]
        ax.plot([min(row_values), max(row_values)], [row, row],
                color=GRID, linewidth=2, zorder=1, solid_capstyle="round")

    for index, model in enumerate(model_names):
        ax.scatter(values_by_model[model], rows,
                   s=110, color=SERIES[index % len(SERIES)],
                   edgecolors=SURFACE, linewidths=1.8, zorder=3)

    # Direct-label the leader on each measure only.
    for position, measure in enumerate(measures):
        best = max(model_names, key=lambda m: values_by_model[m][position])
        best_value = values_by_model[best][position]
        ax.annotate(f"{best_value:.3f}", (best_value, rows[position]),
                    textcoords="offset points", xytext=(11, 0),
                    va="center", fontsize=8, color=TEXT_SECONDARY)

    ax.set_yticks(rows)
    ax.set_yticklabels(measures)
    ax.set_xlim(left, right)
    ax.set_ylim(-0.6, len(measures) - 0.4)
    ax.set_xlabel("Score  (axis starts above zero - dots encode position, not length)")
    ax.set_title(f"Model comparison - {split_name}", pad=30, fontsize=11)
    ax.grid(axis="y", visible=False)
    _swatch_legend(ax, model_names, SERIES[:len(model_names)], marker=True)

    return _save(fig, filename)


def plot_confusion_matrices(results, split_name: str, filename: str) -> Path:
    """One annotated matrix per model, sharing a single sequential blue ramp."""
    _apply_style()

    subset = [r for r in results if r.split_name == split_name]
    fig, axes = plt.subplots(1, len(subset), figsize=(3.4 * len(subset), 3.4))
    axes = np.atleast_1d(axes)

    for ax, result in zip(axes, subset):
        matrix = result.confusion
        normalised = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(normalised, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                # Ink flips to the light token only where the fill is dark.
                colour = "#ffffff" if normalised[i, j] > 0.55 else TEXT_PRIMARY
                ax.text(j, i, f"{matrix[i, j]:,}\n{normalised[i, j]:.1%}",
                        ha="center", va="center", fontsize=8.5, color=colour)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(config.TARGET_NAMES)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(config.TARGET_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(result.model_name, fontsize=10, pad=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"Confusion matrices - {split_name}", fontsize=11, y=1.03)
    fig.tight_layout()
    return _save(fig, filename)


def plot_roc_curves(results, split_name: str, filename: str) -> Path:
    """ROC overlay; each curve is named in the legend with its AUC."""
    from src.evaluate import roc_points

    _apply_style()
    subset = [r for r in results if r.split_name == split_name]

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 4)), linewidth=1.2,
            color=TEXT_MUTED, label="_nolegend_")

    for index, result in enumerate(subset):
        points = roc_points(result)
        if points is None:
            continue
        fpr, tpr = points
        ax.plot(fpr, tpr, linewidth=2, color=SERIES[index % len(SERIES)],
                label=f"{result.model_name}  AUC {result.roc_auc:.4f}")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves - {split_name}", pad=12, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)

    return _save(fig, filename)


# --------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------
def plot_feature_importance(importances: pd.Series,
                            top_n: int = config.TOP_N_IMPORTANCES,
                            filename: str = "05_feature_importance_top10") -> Path:
    """Top-N Random Forest importances as a single-series horizontal bar chart.

    One series, so no legend box is needed - the title names the measure.  The
    axis carries the magnitudes; only the three leading features are
    direct-labelled, with the full ranking in
    ``results/tables/07_feature_importance.csv``.
    """
    _apply_style()

    top = importances.head(top_n).iloc[::-1]
    label_cutoff = float(importances.head(3).min())

    fig, ax = plt.subplots(figsize=(7.6, 0.42 * top_n + 1.4))
    ax.set_xlim(0, float(top.max()) * 1.18)

    bars = ax.barh(np.arange(len(top), dtype=float), top.to_numpy(),
                   height=0.62, color=SERIES[0])

    for bar, value in zip(bars, top.to_numpy()):
        if value >= label_cutoff:
            ax.annotate(f"{value:.4f}",
                        (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        textcoords="offset points", xytext=(6, 0),
                        va="center", fontsize=8, color=TEXT_SECONDARY)

    _round_bar_ends(ax, bars, horizontal=True)

    ax.set_yticks(np.arange(len(top), dtype=float))
    ax.set_yticklabels(list(top.index))
    ax.set_xlabel("Mean decrease in Gini impurity")
    ax.set_title(f"Top {top_n} features by Random Forest importance",
                 pad=12, fontsize=11)
    ax.grid(axis="y", visible=False)

    return _save(fig, filename)


def plot_mutual_information(mi_scores: pd.Series, top_n: int = 15) -> Path:
    """Filter-stage ranking, shown alongside the wrapper-stage importances."""
    _apply_style()

    top = mi_scores.head(top_n).iloc[::-1]
    label_cutoff = float(mi_scores.head(3).min())

    fig, ax = plt.subplots(figsize=(7.6, 0.36 * top_n + 1.4))
    ax.set_xlim(0, float(top.max()) * 1.18)

    bars = ax.barh(np.arange(len(top), dtype=float), top.to_numpy(),
                   height=0.6, color=SERIES[2])

    for bar, value in zip(bars, top.to_numpy()):
        if value >= label_cutoff:
            ax.annotate(f"{value:.3f}",
                        (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        textcoords="offset points", xytext=(6, 0),
                        va="center", fontsize=8, color=TEXT_SECONDARY)

    _round_bar_ends(ax, bars, horizontal=True)

    ax.set_yticks(np.arange(len(top), dtype=float))
    ax.set_yticklabels(list(top.index))
    ax.set_xlabel("Mutual information with the detection target (nats)")
    ax.set_title(f"Top {top_n} features by mutual information", pad=12, fontsize=11)
    ax.grid(axis="y", visible=False)

    return _save(fig, "04_mutual_information_top15")


def plot_category_detection(per_category: pd.DataFrame, split_name: str,
                            filename: str) -> Path:
    """Detection rate per attack family - where aggregate accuracy hides misses."""
    _apply_style()

    subset = per_category[
        (per_category["Split"] == split_name)
        & (~per_category["Category"].str.startswith("Normal"))
    ]
    categories = sorted(subset["Category"].unique())
    models = list(dict.fromkeys(subset["Model"]))

    fig, ax = plt.subplots(figsize=(7.8, 3.9))
    positions = np.arange(len(categories), dtype=float)
    bar_width = 0.20
    gap = 0.02

    ax.set_ylim(0, 1.1)

    all_bars = []
    for index, model in enumerate(models):
        rates = []
        for category in categories:
            match = subset[(subset["Model"] == model)
                           & (subset["Category"] == category)]
            rates.append(float(match["Detection Rate"].iloc[0]) if len(match) else 0.0)

        offset = (index - (len(models) - 1) / 2) * (bar_width + gap)
        bars = ax.bar(positions + offset, rates, bar_width,
                      label=model, color=SERIES[index % len(SERIES)])
        all_bars.extend(bars)
        # Label only the misses - the bars that are the point of the figure.
        for bar, value in zip(bars, rates):
            if value < 0.5:
                ax.annotate(f"{value:.2f}",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            textcoords="offset points", xytext=(0, 3),
                            ha="center", fontsize=7, color=TEXT_SECONDARY)

    _round_bar_ends(ax, all_bars)

    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Detection rate within family")
    ax.set_title(f"Detection rate by attack family - {split_name}",
                 pad=30, fontsize=11)
    ax.grid(axis="x", visible=False)
    _swatch_legend(ax, models, SERIES[:len(models)])

    return _save(fig, filename)
