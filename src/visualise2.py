"""
Shared plotting style utilities for the thesis (Chapter 5 and onward).

Usage:
    from thesis_plot_style import set_thesis_style, plot_bar_chart

    set_thesis_style()  # call once, at the top of a notebook/script
    fig, ax = plot_bar_chart(df["config_name"], df["test_loss"], xlabel="Config", ylabel="Test Loss")
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from src.run_types import AVAIL_SPLITS

# ---------------------------------------------------------------------------
# Colour definitions
# ---------------------------------------------------------------------------

# Fixed, distinct neutrals for "control" bars (no augmentation applied).
# Matched case-insensitively against category names.
CONTROL_COLORS = {
    "baseline": "#4D4D4D",  # darker grey
    "no_aug": "#999999",    # lighter grey
}

DEFAULT_ACCENT = "#0072B2"  # used for every non-control bar


def suggest_palette(
    categories: Sequence[str],
    recognise_controls: bool = True,
) -> list[str]:
    """
    Suggest a colour list aligned to `categories`, one colour per entry.

    Entries matching CONTROL_COLORS (case-insensitive: "baseline", "no_aug")
    get their fixed neutral, if recognise_controls. Every other entry gets
    the same DEFAULT_ACCENT colour.
    """
    if not recognise_controls:
        return [DEFAULT_ACCENT] * len(categories)

    return [CONTROL_COLORS.get(cat.lower(), DEFAULT_ACCENT) for cat in categories]


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

def set_thesis_style() -> None:
    """Apply consistent rcParams for all thesis figures. Call once per session."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# ---------------------------------------------------------------------------
# Chart function
# ---------------------------------------------------------------------------

def plot_bar_chart(
    x: ArrayLike,
    y: ArrayLike,
    horizontal: bool = False,
    palette: Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 4.5),
    rotation: int = 45,
    show_values: bool = True,
    value_fmt: str = "%.3f",
    ax=None,
):
    """
    Plot a bar chart with consistent thesis styling.

    x: category labels (e.g. df["config_name"], or a plain list of strings).
    y: values, aligned to x (e.g. df["test_loss"]).
    palette: list of colours, one per bar. If None, defaults to a single
        accent colour for every bar. Use suggest_palette() to generate one.
    horizontal: if True, uses barh with categories on the y-axis (no tick
        rotation needed -- useful for long category labels).
    show_values: if True (default), annotate each bar with its value.
    value_fmt: printf-style format string for the value labels.
    """
    categories = np.asarray(x).tolist()
    values = np.asarray(y).tolist()
    n = len(categories)

    if palette is None:
        palette = suggest_palette(categories)
    elif len(palette) != n:
        raise ValueError(f"palette has {len(palette)} colours but there are {n} bars.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if horizontal:
        container = ax.barh(categories, values, color=palette)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        if xlabel:
            ax.set_ylabel(xlabel)
        if ylabel:
            ax.set_xlabel(ylabel)
        if show_values:
            ax.bar_label(container, fmt=value_fmt, padding=3)
            ax.set_xlim(0, max(values) * 1.15)  # headroom for labels
    else:
        container = ax.bar(categories, values, color=palette)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")
        if show_values:
            ax.bar_label(container, fmt=value_fmt, padding=3)
            ax.set_ylim(0, max(values) * 1.15)  # headroom for labels

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax

SPLIT_NAME_MAP: dict[AVAIL_SPLITS, str] = {
    'asl100' : 'WLASL-100',
    'asl300' : 'WLASL-300',
    'asl1000' : 'WLASL-1000',
    'asl2000' : 'WLASL-2000',
    'asl100_cutoff_9' : 'WLASL-100',
    'asl300_cutoff_9' : 'WLASL-300',
    'asl1000_cutoff_9': 'WLASL-1000',
    'asl2000_cutoff_9': 'WLASL-2000',
    'asl100_worst': 'WLASL-100 Worst',
    'asl100_bottom': 'WLASL-100 Fewest'
}

def split_name_mapper(split: AVAIL_SPLITS) -> str:
    """Map split name to plot ready name"""
    return SPLIT_NAME_MAP[split]

# ---------------------------------------------------------------------------
