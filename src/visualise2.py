"""
Shared plotting style utilities for the thesis (Chapter 5 and onward).

Usage:
    from thesis_plot_style import set_thesis_style, plot_bar_chart

    set_thesis_style()  # call once, at the top of a notebook/script
    fig, ax = plot_bar_chart(df["config_name"], df["test_loss"], xlabel="Config", ylabel="Test Loss")
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure, FigureBase, SubFigure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from numpy.typing import ArrayLike
from scipy import stats
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from typing_extensions import TypedDict, Unpack

from src.configs import get_class_list
from src.models import get_model
from src.preprocess import Instance

# locals
from src.run_types import (
    AVAIL_SETS,
    AVAIL_SPLITS,
    RAW_DIR,
    BaseRes,
    CentreCropConfig,
    InstanceTopK,
    MinInfo,
    OG_Sampler,
)
from src.testing import (
    load_test_sizes,
    setup_data,
    test_instance_topk,
    test_topk_clsrep,
)
from src.utils import load_rgb_frames_from_video
from src.video_dataset import (
    VideoDataset,
    get_transform,
    get_video_path,
    get_wlasl_info,
)

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

# Okabe-Ito colourblind-safe qualitative palette, used for multi-line charts
# (e.g. comparing several training/validation curves in one figure).
LINE_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#E69F00",  # orange
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
VALUE_FMT = "%.2f"
LOSS_FMT = "%.3f"
COUNT_FMT = "%d"
FIGSIZE = (8, 4.5)

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
    figsize: tuple[float, float] = FIGSIZE,
    rotation: int = 45,
    show_values: bool = True,
    value_fmt: str = VALUE_FMT,
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
            ax.bar_label(container, fmt=value_fmt, padding=3, fontsize=plt.rcParams["ytick.labelsize"])
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
            ax.bar_label(container, fmt=value_fmt, padding=3, fontsize=plt.rcParams["xtick.labelsize"])
            ax.set_ylim(0, max(values) * 1.15)  # headroom for labels

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def add_iqr_lines(
    ax,
    values: ArrayLike,
    value_fmt: str = VALUE_FMT,
    horizontal: bool = False,
    mean_color: str = "black",
    quartile_color: str = "brown",
) -> None:
    """
    Overlay mean +/- std and lower/upper quartile reference lines on an
    existing bar chart axes (e.g. the `ax` returned by plot_bar_chart), with
    the legend placed outside the axes so it never occludes the bars (see
    src/info/plotting_conventions.md).

    values: the data the bars represent (e.g. df["Test Loss"]) -- mean/std/
        quartiles are computed from this, not read back from the bars.
    horizontal: must match the `horizontal` passed to plot_bar_chart -- lines
        are drawn with axhline when False (vertical bars, value on the
        y-axis) or axvline when True (horizontal bars, value on the x-axis).
    """
    values_arr = np.asarray(values, dtype=float)
    mean = values_arr.mean()
    std = values_arr.std(ddof=1)
    q1 = np.percentile(values_arr, 25)
    q3 = np.percentile(values_arr, 75)

    line_fn = ax.axvline if horizontal else ax.axhline

    line_fn(
        mean, color=mean_color, linestyle="--", linewidth=1,
        label=f"Mean: {value_fmt % mean} $\\pm$ {value_fmt % std}",
    )
    line_fn(
        q1, color=quartile_color, linestyle="--", linewidth=1,
        label=f"Lower quartile: {value_fmt % q1}",
    )
    line_fn(
        q3, color=quartile_color, linestyle="--", linewidth=1,
        label=f"Upper quartile: {value_fmt % q3}",
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)


def plot_grouped_bar_chart(
    x: ArrayLike,
    ys: dict[str, ArrayLike],
    palette: Sequence[str] | None = None,
    legend_labels: dict[str, str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = FIGSIZE,
    width: float = 0.2,
    rotation: int = 45,
    show_values: bool = True,
    value_fmt: str = VALUE_FMT,
    ax=None,
):
    """
    Plot a grouped bar chart comparing several series across categories, with
    consistent thesis styling. Intended for e.g. comparing top-1/top-5/top-10
    accuracy across dataset splits or model configs, or a per-class metric
    (e.g. f1-score) across dataset splits.

    x: category labels, one group of bars per entry (e.g. dataset splits, or
        glosses -- see suggest_palette() for many-category colouring).
    ys: mapping from series name to values aligned to x (e.g. {"Top-1":
        df["Top-1"], "Top-5": df["Top-5"]}, or one entry per dataset split
        when comparing a per-class metric across splits).
    palette: list of colours, one per series. Defaults to LINE_PALETTE,
        cycling if there are more series than palette entries.
    legend_labels: optional mapping from series name to display label, for
        renaming series in the legend without renaming keys of `ys`.
    show_values: if True (default), annotate each bar with its value.
    value_fmt: printf-style format string for the value labels.
    """
    categories = np.asarray(x).tolist()
    n = len(categories)

    series_names = list(ys.keys())
    n_series = len(series_names)

    if palette is None:
        palette = [LINE_PALETTE[i % len(LINE_PALETTE)] for i in range(n_series)]
    elif len(palette) != n_series:
        raise ValueError(f"palette has {len(palette)} colours but there are {n_series} series.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_pos = np.arange(n)

    for i, name in enumerate(series_names):
        label = legend_labels.get(name, name) if legend_labels else name
        values = np.asarray(ys[name]).tolist()
        container = ax.bar(x_pos + i * width, values, width, label=label, color=palette[i])
        if show_values:
            ax.bar_label(container, fmt=value_fmt, padding=3, fontsize=plt.rcParams["xtick.labelsize"])

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos + width * (n_series - 1) / 2)
    ax.set_xticklabels(categories)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")
    if show_values:
        max_val = max(np.nanmax(np.asarray(v, dtype=float)) for v in ys.values())
        ax.set_ylim(0, max_val * 1.15)  # headroom for labels

    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    fig.tight_layout()
    return fig, ax


def plot_stacked_bar_chart(
    x: ArrayLike,
    ys: dict[str, ArrayLike],
    palette: Sequence[str] | None = None,
    legend_labels: dict[str, str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = FIGSIZE,
    width: float = 0.5,
    rotation: int = 45,
    show_values: bool = True,
    value_fmt: str = COUNT_FMT,
    label_color: str = "white",
    min_label_frac: float = 0.03,
    ax=None,
):
    """
    Plot a stacked bar chart comparing several series across categories, with
    consistent thesis styling. Intended for e.g. comparing total instance
    counts per split, broken down by dataset subset (train/test/val).

    x: category labels (e.g. dataset splits).
    ys: mapping from series name to values aligned to x (e.g. {"train":
        [...], "test": [...], "val": [...]}), stacked in insertion order.
    palette: list of colours, one per series. Defaults to LINE_PALETTE,
        cycling if there are more series than palette entries.
    legend_labels: optional mapping from series name to display label, for
        renaming series in the legend without renaming keys of `ys`.
    show_values: if True (default), annotate each stacked segment with its
        value, centred within the segment.
    value_fmt: printf-style format string for the in-segment value labels.
        Defaults to COUNT_FMT ("%d"), suited to instance/sample counts.
    label_color: text colour for the in-bar value labels (default white,
        suited to the darker LINE_PALETTE colours).
    min_label_frac: segments shorter than this fraction of the tallest
        stacked bar have their value label omitted, since it wouldn't fit
        within the segment (e.g. a small split's bars are much shorter than
        the largest split's when several splits share one y-axis).
    """
    categories = np.asarray(x).tolist()
    n = len(categories)

    series_names = list(ys.keys())
    n_series = len(series_names)

    if palette is None:
        palette = [LINE_PALETTE[i % len(LINE_PALETTE)] for i in range(n_series)]
    elif len(palette) != n_series:
        raise ValueError(f"palette has {len(palette)} colours but there are {n_series} series.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_pos = np.arange(n)
    bottoms = np.zeros(n)

    all_values = {name: np.asarray(ys[name], dtype=float) for name in series_names}
    totals = sum(all_values.values())
    label_threshold = min_label_frac * totals.max() #type: ignore

    for i, name in enumerate(series_names):
        label = legend_labels.get(name, name) if legend_labels else name
        values = all_values[name]
        container = ax.bar(x_pos, values, width, bottom=bottoms, label=label, color=palette[i])
        if show_values:
            for bar, val, bot in zip(container, values, bottoms):
                if val < label_threshold:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bot + val / 2,
                    value_fmt % val,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=label_color,
                    fontweight="bold",
                )
        bottoms += values

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")

    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    fig.tight_layout()
    return fig, ax


def plot_loss_curves(
    x: ArrayLike,
    ys: dict[str, ArrayLike],
    palette: Sequence[str] | None = None,
    legend_labels: dict[str, str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = FIGSIZE,
    linewidth: float = 1.8,
    ax=None,
):
    """
    Plot one line per series against a shared step/x axis, with consistent
    thesis styling. Intended for comparing training/validation curves across
    multiple runs or configs (e.g. best_val_loss.csv).

    x: shared step/x-axis values (e.g. df["Step"]).
    ys: mapping from series name to values aligned to x (e.g. {"MViTv2-B":
        df["MViTv2-B"], "MViTv2-S": df["MViTv2-S"]}). Values may contain NaNs
        where that run didn't log at a given step (e.g. runs logging on
        offset step counts) -- NaNs are dropped on a per-series basis so each
        line stays continuous.
    palette: list of colours, one per series. Defaults to LINE_PALETTE,
        cycling if there are more series than palette entries.
    legend_labels: optional mapping from series name to display label, for
        renaming series in the legend without renaming keys of `ys`.
    """
    series_names = list(ys.keys())
    n = len(series_names)

    if palette is None:
        palette = [LINE_PALETTE[i % len(LINE_PALETTE)] for i in range(n)]
    elif len(palette) != n:
        raise ValueError(f"palette has {len(palette)} colours but there are {n} series.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_arr = np.asarray(x)
    for color, name in zip(palette, series_names):
        values = np.asarray(ys[name])
        mask = ~pd.isna(values)
        label = legend_labels.get(name, name) if legend_labels else name
        ax.plot(x_arr[mask], values[mask], color=color, linewidth=linewidth, label=label)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_metric_correlation(
    x: ArrayLike,
    y: ArrayLike,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "F1 score",
    figsize: tuple[float, float] = FIGSIZE,
    color: str = DEFAULT_ACCENT,
    fit_color: str = LINE_PALETTE[1],
    shade_by_density: bool = True,
    y_clip: tuple[float, float] | None = None,
    legend_top_y: float | None = None,
    log_x: bool = False,
    ax=None,
):
    """
    Scatter a metric (e.g. per-gloss F1 score) against another variable
    (e.g. per-gloss instance/signer count), with a linear trend line
    annotated with Kendall's tau, with consistent thesis styling. Intended
    for e.g. correlating per-class recognition performance with dataset
    statistics.

    x: values for the x-axis (e.g. per-gloss instance or signer counts).
    y: metric values aligned to x (e.g. per-gloss F1 scores).
    shade_by_density: if True (default), points sharing an exact (x, y)
        coordinate are shaded darker, via a white-to-`color` ramp -- useful
        when many categories (e.g. glosses) collide on the same point. If
        False, every point is drawn in `color` at a flat alpha.
    fit_color: colour of the linear-fit line. Defaults to the vermillion
        entry of LINE_PALETTE for contrast against `color`.
    y_clip: if given, clamps the linear-fit line to this (min, max) range
        (e.g. (0, 1) for a bounded metric like F1 score) so extrapolation
        can't draw it outside the metric's valid range. Does not affect the
        scattered points themselves.
    legend_top_y: if given, the legend (bottom-right corner of the axes) is
        raised so its top edge aligns with this y-axis data value, instead
        of sitting flush against the bottom of the axes -- useful to clear
        a cluster of low-value points near the bottom-right corner.
    log_x: if True, draw the x-axis on a log scale and fit the trend line
        against log10(x) (e.g. for log-uniform sweep hyperparameters such as
        learning rates). Kendall's tau is rank-based, so it is unaffected.

    Returns (fig, ax, tau, p_value) -- tau/p_value from scipy's Kendall's
    tau, so callers can report them alongside the figure.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    tau, p_value = stats.kendalltau(x_arr, y_arr)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if shade_by_density:
        coord_df = pd.DataFrame({"x": x_arr, "y": y_arr})
        point_counts = coord_df.groupby(["x", "y"])["x"].transform("size").to_numpy(dtype=float)
        norm_counts = point_counts / point_counts.max()
        norm_counts = 0.4 + 0.6 * norm_counts  # remap to [0.4, 1.0] instead of [0, 1.0]
        cmap = LinearSegmentedColormap.from_list("metric_correlation", ["#FFFFFF", color])
        ax.scatter(x_arr, y_arr, c=norm_counts, cmap=cmap, vmin=0, vmax=1, s=60, edgecolors="none")
    else:
        ax.scatter(x_arr, y_arr, color=color, alpha=0.6, edgecolors="white", linewidths=0.4, s=60)

    # Fit (and truncate the fit range) in log10 space when log_x, mapping back
    # to data space only to draw the line.
    x_fit = np.log10(x_arr) if log_x else x_arr
    coeffs = np.polyfit(x_fit, y_arr, deg=1)
    slope, intercept = coeffs
    x_min, x_max = x_fit.min(), x_fit.max()
    if y_clip is not None and slope != 0:
        # Truncate the x-range at whichever clip bound the line reaches first,
        # rather than clamping y -- clamping y instead would draw a flat
        # horizontal segment past the crossing point.
        x_at_lo = (y_clip[0] - intercept) / slope
        x_at_hi = (y_clip[1] - intercept) / slope
        x_at_lo, x_at_hi = sorted((x_at_lo, x_at_hi))
        x_min = max(x_min, x_at_lo)
        x_max = min(x_max, x_at_hi)
    x_line = np.linspace(x_min, x_max, 100)
    y_line = np.polyval(coeffs, x_line)
    if log_x:
        x_line = 10**x_line
        ax.set_xscale("log")
    ax.plot(
        x_line, y_line, color=fit_color, linewidth=1.8,
        label=f"Linear fit (Kendall $\\tau$ = {tau:.3f}, p = {p_value:.3e})",
    )

    ax.grid(linestyle="--", alpha=0.3)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    # Deliberate deviation from the outside-axes legend convention used elsewhere in this
    # module: scatter correlation plots leave a corner opposite the trend largely empty, so
    # an in-axes legend there avoids occluding points without wasting horizontal space on an
    # outside legend. Check the rendered figure when reusing this for a differently-shaped
    # trend -- "lower right" only stays clear for a positive x/y correlation.
    if legend_top_y is not None:
        y_lo, y_hi = ax.get_ylim()
        top_frac = (legend_top_y - y_lo) / (y_hi - y_lo)
        ax.legend(loc="upper right", bbox_to_anchor=(1, top_frac))
    else:
        ax.legend(loc="lower right")

    fig.tight_layout()
    return fig, ax, tau, p_value


def save_fig(
    fig: Figure,
    path: str | Path,
    dpi: int | None = None,
    bbox_inches: str | None = "tight",
) -> Path:
    """
    Save a figure to disk, creating parent directories as needed.

    fig: the figure to save (e.g. returned by plot_bar_chart, plot_grouped_bar_chart,
        plot_loss_curves).
    path: destination file path, e.g. "outputs/my_plot.pdf".
    dpi: passed to fig.savefig. Defaults to None, which falls back to the
        "savefig.dpi" rcParam (300, set by set_thesis_style).
    bbox_inches: passed to fig.savefig. Defaults to "tight" so labels and
        legends outside the axes aren't clipped, matching set_thesis_style's
        "savefig.bbox" rcParam.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    savefig_kwargs: dict[str, Any] = {"bbox_inches": bbox_inches}
    if dpi is not None:
        savefig_kwargs["dpi"] = dpi

    fig.savefig(path, **savefig_kwargs)
    return path


# ---------------------------------------------------------------------------
# BBox visualisation
# ---------------------------------------------------------------------------

AverageMethod = Literal["mean", "median"]


def _average_bboxes(instances: Sequence[Instance], method: AverageMethod) -> list[Instance]:
    """Collapse instances to one per class, with a mean/median-averaged bbox."""
    groups: dict[str, list[Instance]] = defaultdict(list)
    for inst in instances:
        groups[inst.label_name].append(inst)

    avg_fn = np.mean if method == "mean" else np.median
    averaged = []
    for insts in groups.values():
        boxes = np.array([inst.bbox for inst in insts], dtype=float)
        avg_box = avg_fn(boxes, axis=0).round().astype(int).tolist()
        averaged.append(insts[0].model_copy(update={"bbox": avg_box}))
    return averaged


def plot_bboxes_on_canvas(
    instances: Sequence[Instance],
    average: bool = True,
    method: AverageMethod = "mean",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    frame_size: tuple[int, int] = (256, 256),
    ax=None,
):
    """
    Draw each class's bounding box outline on a blank video-frame-sized
    canvas, one colour per class, with consistent thesis styling. Intended
    for spotting how consistent/central the cropped signer region is across
    classes in a split/set.

    instances: bboxes to draw, one per class if `average` (the common case --
        drawing every raw instance bbox is only useful for debugging a single
        class's bbox spread).
    average: if True (default), collapse `instances` to one bbox per class
        first, via `method` ("mean" or "median" of the class's bboxes).
    figsize: defaults to a size matching `frame_size`'s aspect ratio (square
        for the default 256x256 canvas), unlike other visualise2 charts which
        default to the wide FIGSIZE -- a non-square canvas here would distort
        the drawn boxes.
    frame_size: (width, height) of the canvas the boxes are drawn on --
        matches the video frame size the bboxes were computed against
        (WLASL precut clips are 256x256).
    """
    width, height = frame_size
    if figsize is None:
        side = FIGSIZE[1]
        figsize = (side * width / height, side)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image coordinates: y increases downward

    if average:
        instances = _average_bboxes(instances, method)

    unique_labels = sorted({inst.label_name for inst in instances})
    cmap = plt.get_cmap("tab20", len(unique_labels))
    colour_map = {label: cmap(i) for i, label in enumerate(unique_labels)}

    for inst in instances:
        x1, y1, x2, y2 = inst.bbox
        colour = colour_map[inst.label_name]
        ax.add_patch(
            Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1, edgecolor=colour, facecolor=(*colour[:3], 0.05),
            )
        )

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig, ax


def plot_dimension_distributions(
    instances: Sequence[Instance],
    bins: int = 30,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4.5),
    ax=None,
):
    """
    Plot histograms of bbox width and height across a set of instances, each
    annotated with mean/median/quartile lines, with consistent thesis
    styling. Intended for sanity-checking how much bbox dimensions vary
    within a dataset split/set.

    Returns (fig, axes) -- axes is a length-2 array (width, height), unlike
    other visualise2 charts, since this plot is inherently two histograms
    side by side.
    """
    widths = [inst.bbox[2] - inst.bbox[0] for inst in instances]
    heights = [inst.bbox[3] - inst.bbox[1] for inst in instances]

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = ax[0].figure
        axes = ax

    stat_lines: list[tuple[str, Callable[[ArrayLike], float], str]] = [ # type: ignore 
        ("mean", np.mean, "red"),
        ("median", np.median, "blue"),
        ("lower quartile", lambda v: np.percentile(v, 25), "brown"), # type: ignore
        ("upper quartile", lambda v: np.percentile(v, 75), "brown"), # type: ignore
    ]

    for cur_ax, data, label, colour in zip(
        axes, [widths, heights], ["Width (px)", "Height (px)"], LINE_PALETTE[:2]
    ):
        cur_ax.hist(data, bins=bins, color=colour, edgecolor="white")
        for stat_name, stat_fn, line_colour in stat_lines:
            stat_val = stat_fn(data)
            cur_ax.axvline(
                stat_val, color=line_colour, linestyle="--",
                label=f"{stat_name}: {stat_val:.1f} (px)",
            )
        cur_ax.set_xlabel(label)
        cur_ax.set_ylabel("Count")
        cur_ax.grid(axis="y", linestyle="--", alpha=0.3)
        cur_ax.legend()

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------

SPLIT_NAME_MAP: dict[AVAIL_SPLITS, str] = {
    'asl100' : 'WLASL-100',
    'asl300' : 'WLASL-300',
    'asl1000' : 'WLASL-1000',
    'asl2000' : 'WLASL-2000',
    'asl100_cutoff_9' : 'WLASL-100',
    'asl300_cutoff_9' : 'WLASL-300',
    'asl1000_cutoff_9': 'WLASL-1000',
    'asl2000_cutoff_9': 'WLASL-2000',
    'asl100_worst': 'Worst-100',
    'asl100_bottom': 'Fewest-100'
}

def split_name_mapper(split: AVAIL_SPLITS) -> str:
    """Map split name to plot ready name"""
    return SPLIT_NAME_MAP[split]

# ---------------------------------------------------------------------------
# Frame visualiser
# ---------------------------------------------------------------------------

class MiniSetKwargsRequired(TypedDict):
    cls_idx: int
    all_sets: dict[str, Any]
    set_name: AVAIL_SETS
    split_name: AVAIL_SPLITS


class MiniSetKwargs(MiniSetKwargsRequired, total=False):
    classes: list[str]
    target_length: int
    frame_size: int
    logger: Logger

visualise_logger = logging.getLogger(__name__)


def load_instance_frames(instance: Instance, video_dir: Path = RAW_DIR) -> Tensor:
    """Every frame of an instance's labelled range, untransformed: (T, C, H, W), RGB, uint8.

    Unlike `MiniSet`/`FrameFetcher`, nothing is subsampled or cropped, so this is the
    clip to pass to `animate_frames` to watch it at its real length and speed.
    """
    return load_rgb_frames_from_video(
        get_video_path(instance.video_id, video_dir), instance.frame_start, instance.frame_end
    )

class MiniSet(Dataset):
    def __init__(
        self,
        cls_idx: int,
        all_sets: dict[str, Any],
        set_name: AVAIL_SETS,
        split_name: AVAIL_SPLITS,
        classes: list[str] | None = None,
        target_length: int = 16,
        frame_size: int = 224,
        logger: Logger = visualise_logger,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:

        if classes is None:
            classes = get_class_list()
        else:
            assert len(classes) != 0, 'No classes provided'

        self.logger = logger
        self.cls_idx = cls_idx
        self.set_name = set_name
        self.split_name = split_name
        self.classes = classes
        self.target_length = target_length
        self.frame_size = frame_size

        if transform is None:
            self.transform, _, _ = get_transform(
                temporal_aug=[OG_Sampler(target_length=target_length)],
                spatial_aug=[CentreCropConfig(frame_size=frame_size)],
                normalise_to_float=False,
                permute_time_channel=False,
            )
        else:
            self.transform = transform

        self.set_path_info = get_wlasl_info(split_name, set_name)
        self.data = all_sets[self.set_name][self.cls_idx]["instances"]
        self.tot_samples = len(self.data)

    def __getitem__(self, idx):
        self.logger.info(f"From: {self.split_name}S/{self.set_name}")
        self.logger.info(f'Example videos for class: "{self.classes[self.cls_idx]}"')
        self.logger.info(f"Instance: {idx + 1}/{self.tot_samples}")

        next_example = Instance.model_validate(self.data[idx])
        self.logger.info(
            f"Next example video path: {get_video_path(next_example.video_id, self.set_path_info['root'])}"
        )
        return self.transform(load_instance_frames(next_example, self.set_path_info["root"]))

    def __len__(self):
        return self.tot_samples


def _sample_frames(frames: Tensor, num: int) -> Tensor:
    """Evenly sample (at most) `num` frames from a (T, C, H, W) tensor."""
    if num < 1:
        raise ValueError("num must be >= 1")
    step = 1 if len(frames) <= num else len(frames) // num
    return frames[::step][:num]


def _to_display(frame: Tensor) -> np.ndarray:
    """(C, H, W) frame -> min-max normalised (H, W, C) array for `imshow`."""
    np_frame = frame.permute(1, 2, 0).cpu().numpy().astype(float)
    return (np_frame - np_frame.min()) / (np_frame.max() - np_frame.min())


def _draw_frame_grid(fig: FigureBase, sampled: Tensor, cols: int) -> np.ndarray:
    """Draw `sampled` frames onto a (rows x cols) grid of `fig` (a Figure or SubFigure).

    Unused cells in the last row are hidden. Returns the 2D axes array.
    """
    rows = math.ceil(len(sampled) / cols)
    axes = fig.subplots(rows, cols, squeeze=False)
    for i, frame in enumerate(sampled):
        ax = axes[i // cols][i % cols]
        ax.imshow(_to_display(frame))
        ax.axis("off")
    for j in range(len(sampled), rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    return axes


def plot_frame_grid(
    frames: Tensor,
    num: int,
    size: tuple[float, float] = (5.0, 5.0),
    adapt: bool = False,
    cols: int = 8,
    title: str | None = None,
):
    """
    Arrange an evenly-sampled subset of video frames into a grid, with
    consistent thesis styling. Intended for showing/saving example clips
    (e.g. FrameVisualiser, or per-gloss prediction/misprediction frame grids).

    frames: (T, C, H, W) tensor, RGB channel order.
    num: number of frames to show, evenly sampled across `frames`. Must be >= 1.
    size: (width, height) in inches per grid cell.
    adapt: if True, scale `size` from the frames' actual resolution instead of
        using `size` as given -- useful when frame_size differs from the
        256x256 this default was tuned against.
    cols: max frames per row; unused cells in the last row are hidden.

    Returns (fig, axes) -- axes is a 2D array (rows x cols), the same
    exception to the single-`ax` return convention as
    plot_dimension_distributions, since this is inherently a grid of
    subplots rather than one axes to hand back or accept.
    """
    sampled = _sample_frames(frames, num)

    if adapt:
        factor = 5 / 256
        h, w = frames.shape[2], frames.shape[3]
        size = (w * factor, h * factor)

    rows = math.ceil(len(sampled) / cols)
    fig = plt.figure(figsize=(size[0] * cols, size[1] * rows))
    axes = _draw_frame_grid(fig, sampled, cols)

    # All axes have axis("off") -- no tick/axis labels for tight_layout to
    # account for -- so subplots_adjust with explicit margins is used
    # directly instead, leaving room at the top only when there's a title. That
    # room is a fixed height in inches, as a fraction would overlap the title
    # with a short (e.g. single-row) grid.
    top = 0.995
    if title:
        fig.suptitle(title)
        top = 1 - 0.45 / fig.get_figheight()
    fig.subplots_adjust(left=0.005, right=0.995, top=top, bottom=0.005, wspace=0.02, hspace=0.02)

    return fig, axes


# ---------------------------------------------------------------------------
# Frames as a video
# ---------------------------------------------------------------------------

def _frame_panel_size(frames: Tensor, scale: float) -> tuple[float, float]:
    """(width, height) in inches at which `frames` show at `scale` x their own pixels."""
    dpi = plt.rcParams["figure.dpi"]
    return frames.shape[-1] * scale / dpi, frames.shape[-2] * scale / dpi


def _animate(fig: Figure, panels: Sequence[tuple[Axes, Tensor]], interval: int) -> FuncAnimation:
    """Play each (ax, (T, C, H, W) RGB frames) clip on its axes in step, one frame per
    `interval` ms, with the axes hidden. Shorter clips hold their last frame."""
    images: list[AxesImage] = []
    for ax, frames in panels:
        ax.axis("off")
        images.append(ax.imshow(_to_display(frames[0])))

    def _update(i: int) -> list[AxesImage]:
        for image, (_, frames) in zip(images, panels):
            image.set_data(_to_display(frames[min(i, len(frames) - 1)]))
        return images

    length = max(len(frames) for _, frames in panels)
    return FuncAnimation(fig, _update, frames=length, interval=interval, blit=True)


def animate_frames(
    frames: Tensor,
    scale: float = 1.0,
    interval: int = 40,
    title: str | None = None,
):
    """
    Play a clip frame by frame -- the video counterpart of plot_frame_grid.

    frames: (T, C, H, W) tensor, RGB channel order; every frame is played.
    scale: size relative to the frames' own resolution. At 1 (default), a
        256x256 clip plays at 256x256 pixels, with no border.
    interval: delay between frames in milliseconds. The default, 40, is
        WLASL's 25 fps; raise it to slow down a short, subsampled clip.
    title: drawn in a strip above the frames, which makes the figure taller.

    Returns (fig, anim). Show it in a notebook with `animation_html(fig, anim)`,
    which keeps the pixel size; save with `anim.save(path, dpi=fig.dpi)` (a
    ".gif" needs pillow, ".mp4" needs ffmpeg) -- save_fig only handles still
    figures.
    """
    frame_w, frame_h = _frame_panel_size(frames, scale)
    title_h = 0.3 if title else 0.0  # inches
    fig = plt.figure(figsize=(frame_w, frame_h + title_h), dpi=plt.rcParams["figure.dpi"])
    ax = fig.add_axes((0.0, 0.0, 1.0, frame_h / (frame_h + title_h)))
    if title:
        fig.suptitle(title, y=1 - 0.05 / (frame_h + title_h), va="top")
    return fig, _animate(fig, [(ax, frames)], interval)


def animation_html(fig: Figure, anim: FuncAnimation) -> HTML:
    """Render `anim` as an inline notebook player, closing `fig`.

    Frames are embedded as JPEGs (about 4x smaller than PNG for video) at the
    figure's own dpi rather than `savefig.dpi` (300 under set_thesis_style), so
    the player is the figure's on-screen size. Without closing `fig`, a notebook
    also shows its static first frame below the player. Display the result with
    `display(...)`, or as a cell's last expression.
    """
    plt.close(fig)
    with plt.rc_context({"savefig.dpi": "figure", "animation.frame_format": "jpeg"}):
        return HTML(anim.to_jshtml())


# ---------------------------------------------------------------------------
# Joining panels
# ---------------------------------------------------------------------------

PanelPosition = Literal["left", "right", "top", "bottom"]


def _is_side_by_side(position: PanelPosition) -> bool:
    return position in ("left", "right")


def _centre_in(slot: SubFigure, extent: float, full: float, vertical: bool) -> SubFigure:
    """The part of `slot` that is `extent` of its `full` inches long, centred, along
    the vertical (else horizontal) axis -- or `slot` itself if it isn't shorter."""
    if extent >= full:
        return slot
    pad = (full - extent) / 2
    ratios = [pad, extent, pad]
    if vertical:
        return slot.subfigures(3, 1, height_ratios=ratios)[1]
    return slot.subfigures(1, 3, width_ratios=ratios)[1]


def join_panels(
    first_size: tuple[float, float],
    second_size: tuple[float, float],
    position: PanelPosition,
) -> tuple[Figure, SubFigure, SubFigure]:
    """Lay out a figure of two panels, the second on the `position` side of the first.

    A building block for figures that pair two plots, e.g. a clip and its top-k chart,
    or two clips compared side by side. Draw into each panel with its `subplots()`.

    Args:
        first_size (tuple[float, float]): (width, height) in inches of the first panel.
        second_size (tuple[float, float]): (width, height) in inches of the second panel.
        position (PanelPosition): Side of the first panel the second sits on.

    Returns:
        tuple[Figure, SubFigure, SubFigure]: (fig, first, second). Where the panels'
            sizes differ across the joining direction (heights for "left"/"right",
            widths for "top"/"bottom"), the smaller one is centred, padded to the
            larger. The figure uses constrained layout, so tick labels get room without
            overlapping the other panel; don't call `tight_layout` on it.
    """
    side_by_side = _is_side_by_side(position)
    (first_w, first_h), (second_w, second_h) = first_size, second_size
    if side_by_side:
        figsize = (first_w + second_w, max(first_h, second_h))
        extents = [first_w, second_w]
    else:
        figsize = (max(first_w, second_w), first_h + second_h)
        extents = [first_h, second_h]
    second_first = position in ("left", "top")
    if second_first:
        extents.reverse()

    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)  # type: ignore[union-attr]
    if side_by_side:
        slots = fig.subfigures(1, 2, width_ratios=extents)
    else:
        slots = fig.subfigures(2, 1, height_ratios=extents)
    second_slot, first_slot = slots if second_first else slots[::-1]

    def _centred(slot: SubFigure, size: tuple[float, float]) -> SubFigure:
        if side_by_side:
            return _centre_in(slot, size[1], figsize[1], vertical=True)
        return _centre_in(slot, size[0], figsize[0], vertical=False)

    return fig, _centred(first_slot, first_size), _centred(second_slot, second_size)


# ---------------------------------------------------------------------------
# Frames joined with a top-k confidence bar chart
# ---------------------------------------------------------------------------

# Highlights the ground-truth class among the otherwise DEFAULT_ACCENT top-k bars.
TRUE_CLASS_COLOR = LINE_PALETTE[2]  # Okabe-Ito green

# Least inches a top-k chart takes along the joining direction, so its tick labels
# (gloss names, or the confidence axis) still leave room for the bars.
MIN_CHART_EXTENT = 2.5
# Least inches per bar across a top-k chart, so neighbouring labels don't overlap.
# Vertical bars need more, as their value labels sit side by side.
MIN_HBAR_PITCH = 0.2
MIN_VBAR_PITCH = 0.35


def _top_k(labels: Sequence[str], scores: ArrayLike, k: int) -> tuple[list[str], np.ndarray]:
    """The `k` highest-scoring (label, score) pairs, in descending score order."""
    scores_arr = np.asarray(scores, dtype=float)
    if len(labels) != len(scores_arr):
        raise ValueError(f"{len(labels)} labels but {len(scores_arr)} scores.")
    if not 1 <= k <= len(scores_arr):
        raise ValueError(f"k must be in [1, {len(scores_arr)}], got {k}.")
    order = np.argsort(scores_arr)[::-1][:k]
    return [labels[int(i)] for i in order], scores_arr[order]


def draw_topk_chart(
    ax: Axes,
    labels: Sequence[str],
    scores: ArrayLike,
    k: int = 5,
    true_label: str | None = None,
    horizontal: bool = True,
    shared_axis: bool = False,
) -> None:
    """Draw a bar chart of the `k` highest `scores` on `ax`, highest first.

    Args:
        ax (Axes): Axes to draw on.
        labels (Sequence[str]): Class names, aligned with `scores`. Need not be sorted or
            complete -- a stored top-N (`InstanceTopK`) works as long as k <= N.
        scores (ArrayLike): Confidences (e.g. softmax probabilities), aligned with `labels`.
        k (int, optional): Number of bars. Defaults to 5.
        true_label (str | None, optional): If given and among the top k, its bar is drawn
            in TRUE_CLASS_COLOR instead of DEFAULT_ACCENT. Defaults to None.
        horizontal (bool, optional): If True, horizontal bars reading top-down (long gloss
            labels read easily); else vertical bars reading left-right, with rotated
            labels. Defaults to True.
        shared_axis (bool, optional): If True, the confidence axis always spans [0, 1], so
            bar lengths can be compared between charts (e.g. stepping through a gloss's
            instances). If False, it is scaled to the top score, which keeps
            low-confidence bars readable. Defaults to False.

    Raises:
        ValueError: If `labels` and `scores` differ in length, or k is out of range.
    """
    top_labels, top_scores = _top_k(labels, scores, k)
    conf_max = (1.0 if shared_axis else top_scores.max()) * 1.15  # headroom for value labels
    palette = [TRUE_CLASS_COLOR if lab == true_label else DEFAULT_ACCENT for lab in top_labels]
    tick_labels = [lab.replace("_", " ") for lab in top_labels]
    positions = np.arange(len(top_labels))
    if horizontal:
        container = ax.barh(positions, top_scores, color=palette)
        ax.set_yticks(positions, tick_labels)
        ax.invert_yaxis()
        ax.set_xlim(0, conf_max)
        ax.set_xlabel("Confidence")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
    else:
        container = ax.bar(positions, top_scores, color=palette)
        ax.set_xticks(positions, tick_labels, rotation=45, ha="right")
        ax.set_ylim(0, conf_max)
        ax.set_ylabel("Confidence")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.bar_label(container, fmt=VALUE_FMT, padding=3, fontsize=plt.rcParams["xtick.labelsize"])


def _matched_chart_size(
    frames_size: tuple[float, float], position: PanelPosition, k: int
) -> tuple[float, float]:
    """The frames area's own size, so chart and frames each take half the figure.

    Raised where that's too small to read: to MIN_CHART_EXTENT along the joining
    direction, and across it (the side the `k` bars spread along) to k x
    MIN_HBAR_PITCH/MIN_VBAR_PITCH, in which case `join_panels` centres the frames.
    """
    width, height = frames_size
    if _is_side_by_side(position):
        return max(width, MIN_CHART_EXTENT), max(height, k * MIN_HBAR_PITCH)
    return max(width, k * MIN_VBAR_PITCH), max(height, MIN_CHART_EXTENT)


def _fitted_cell_size(
    frames: Tensor, cols: int, position: PanelPosition
) -> tuple[float, float]:
    """Frame-cell (width, height) in inches at which a `cols`-wide grid of `frames` plus
    its default (matched) top-k chart make a FIGSIZE-wide figure.

    Text is sized in points, so a figure only shows it at the same size as others
    (e.g. an `animate_frames_topk` video, or a thesis page) if it's drawn near the size
    it's displayed at, not shrunk to fit.
    """
    fig_w = FIGSIZE[0]
    # beside the grid the chart matches its width, but takes at least MIN_CHART_EXTENT
    grid_w = min(fig_w / 2, fig_w - MIN_CHART_EXTENT) if _is_side_by_side(position) else fig_w
    cell_w = grid_w / cols
    return cell_w, cell_w * frames.shape[-2] / frames.shape[-1]


def _join_frames_and_chart(
    frames_size: tuple[float, float],
    labels: Sequence[str],
    scores: ArrayLike,
    k: int,
    true_label: str | None,
    bar_position: PanelPosition,
    shared_axis: bool,
    chart_size: tuple[float, float] | None,
    title: str | None,
) -> tuple[Figure, SubFigure, Axes]:
    """`join_panels` of a frames area and a `draw_topk_chart` beside it (matched to the
    frames' size if `chart_size` is None). Returns (fig, frames_panel, bar_ax)."""
    if chart_size is None:
        chart_size = _matched_chart_size(frames_size, bar_position, k)
    fig, frames_panel, chart_panel = join_panels(frames_size, chart_size, bar_position)
    bar_ax = chart_panel.subplots()
    draw_topk_chart(
        bar_ax, labels, scores, k, true_label, _is_side_by_side(bar_position), shared_axis
    )
    if title:
        fig.suptitle(title)
    return fig, frames_panel, bar_ax


def plot_frame_grid_topk(
    frames: Tensor,
    labels: Sequence[str],
    scores: ArrayLike,
    k: int = 5,
    true_label: str | None = None,
    bar_position: PanelPosition = "right",
    shared_axis: bool = False,
    num: int = 16,
    cols: int = 8,
    size: tuple[float, float] | None = None,
    chart_size: tuple[float, float] | None = None,
    title: str | None = None,
):
    """A `plot_frame_grid` of a clip joined to a bar chart of the model's top-k class
    confidences for it. Intended for inspecting individual (mis)predictions, e.g. what a
    gloss was predicted as and how confidently.

    Args:
        frames (Tensor): The clip, as for `plot_frame_grid`.
        labels (Sequence[str]): Class names, aligned with `scores`, as for
            `draw_topk_chart`.
        scores (ArrayLike): Confidences, aligned with `labels`, as for `draw_topk_chart`.
        k (int, optional): Number of bars, highest score first. Defaults to 5.
        true_label (str | None, optional): If given and among the top k, its bar is drawn
            in TRUE_CLASS_COLOR instead of DEFAULT_ACCENT. Defaults to None.
        bar_position (PanelPosition, optional): Side of the frames the chart sits on.
            "left"/"right" use horizontal bars (long gloss labels read easily);
            "top"/"bottom" use vertical bars with rotated labels. Defaults to "right".
        shared_axis (bool, optional): Fix the confidence axis to [0, 1], as for
            `draw_topk_chart`. Defaults to False.
        num (int, optional): Number of frames sampled, as for `plot_frame_grid`.
            Defaults to 16.
        cols (int, optional): Frame-grid columns, as for `plot_frame_grid`. Defaults to 8.
        size (tuple[float, float] | None, optional): (width, height) in inches of each
            frame cell. Defaults to None: sized so that, with the default `chart_size`,
            the figure is FIGSIZE wide, which keeps its text the same displayed size as
            other figures' rather than shrunk with an oversized figure.
        chart_size (tuple[float, float] | None, optional): (width, height) in inches of
            the chart; centred against the grid if it's smaller across the joining
            direction. Defaults to None: the grid's own size, so each takes half the
            figure, but enlarged where k bars wouldn't fit (see `_matched_chart_size`).
        title (str | None, optional): Figure suptitle. Defaults to None.

    Returns:
        tuple[Figure, np.ndarray, Axes]: (fig, frame_axes, bar_ax). frame_axes is the 2D
            grid array, as for `plot_frame_grid`; there's no `ax` parameter for the same
            reason.
    """
    sampled = _sample_frames(frames, num)
    rows = math.ceil(len(sampled) / cols)
    if size is None:
        size = _fitted_cell_size(frames, cols, bar_position)
    fig, frames_panel, bar_ax = _join_frames_and_chart(
        (size[0] * cols, size[1] * rows),
        labels, scores, k, true_label, bar_position, shared_axis, chart_size, title,
    )
    return fig, _draw_frame_grid(frames_panel, sampled, cols), bar_ax


def animate_frames_topk(
    frames: Tensor,
    labels: Sequence[str],
    scores: ArrayLike,
    k: int = 5,
    true_label: str | None = None,
    bar_position: PanelPosition = "right",
    shared_axis: bool = False,
    scale: float = 1.0,
    chart_size: tuple[float, float] | None = None,
    interval: int = 100,
    title: str | None = None,
):
    """The video counterpart of `plot_frame_grid_topk`: plays the clip frame by frame
    beside the (static) top-k confidence bar chart.

    Args:
        frames (Tensor): (T, C, H, W) tensor, RGB channel order; every frame is played.
        labels (Sequence[str]): As for `plot_frame_grid_topk`.
        scores (ArrayLike): As for `plot_frame_grid_topk`.
        k (int, optional): As for `plot_frame_grid_topk`. Defaults to 5.
        true_label (str | None, optional): As for `plot_frame_grid_topk`. Defaults to None.
        bar_position (PanelPosition, optional): As for `plot_frame_grid_topk`.
            Defaults to "right".
        shared_axis (bool, optional): As for `plot_frame_grid_topk`. Defaults to False.
        scale (float, optional): Video size relative to the frames' own resolution, as
            for `animate_frames`. With the default `chart_size`, this sizes the chart
            too, so raise it if the chart's labels crowd. Defaults to 1.0.
        chart_size (tuple[float, float] | None, optional): As for
            `plot_frame_grid_topk`, matching the video's size by default.
        interval (int, optional): Delay between frames in milliseconds (e.g. 40 for
            25 fps). Defaults to 100.
        title (str | None, optional): Figure suptitle. Defaults to None.

    Returns:
        tuple[Figure, FuncAnimation, Axes]: (fig, anim, bar_ax), shown and saved as for
            `animate_frames`.
    """
    fig, video_panel, bar_ax = _join_frames_and_chart(
        _frame_panel_size(frames, scale),
        labels, scores, k, true_label, bar_position, shared_axis, chart_size, title,
    )
    return fig, _animate(fig, [(video_panel.subplots(), frames)], interval), bar_ax


class FrameVisualiser:
    def __init__(self, **kwargs: Unpack[MiniSetKwargs]):
        self.target_frames = kwargs.get("target_length", 16)
        self.frame_size = kwargs.get('frame_size', 224)
        self.iter_loader = iter(
            DataLoader(
                MiniSet(**kwargs),
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=False,
            )
        )

    def __call__(self):
        frames = next(self.iter_loader)[0]
        if len(frames.shape) == 5:
            frames = frames.squeeze(dim=0)
        if frames.shape[1] != 3:
            frames = frames.permute(1, 0, 2, 3)  # swap T and C

        plot_frame_grid(frames, self.target_frames)


class FrameFetcher:
    """Fetch one class's instances' frames, one instance per call, in set order.

    cycle: if True, wrap back to the first instance after the last, instead of
        raising StopIteration -- for notebook cells re-run to step through a
        class.

    `cur_idx` is the 1-based position of the most recently fetched instance
    (0 before the first call), and `current_instance` its metadata (e.g. its
    `video_id`, to look up per-instance predictions).
    """

    def __init__(self, cycle: bool = False, **kwargs: Unpack[MiniSetKwargs]):
        self.cycle = cycle
        self.cur_idx: int = 0
        self.dataset = MiniSet(**kwargs)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )
        self.iter_loader = iter(self.dataloader)
        self.len = len(self.dataloader)

    @property
    def current_instance(self) -> Instance:
        if self.cur_idx == 0:
            raise RuntimeError("No instance fetched yet")
        return Instance.model_validate(self.dataset.data[self.cur_idx - 1])

    def __call__(self) -> Tensor:
        if self.cycle and self.cur_idx == self.len:
            self.iter_loader = iter(self.dataloader)
            self.cur_idx = 0
        frames = next(self.iter_loader)[0]
        self.cur_idx += 1
        if len(frames.shape) == 5:
            frames = frames.squeeze(dim=0)
        if frames.shape[1] != 3:
            frames = frames.permute(1, 0, 2, 3)  # swap T and C

        return frames


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _load_trained(
    admin: MinInfo,
    set_name: AVAIL_SETS,
    split_name: AVAIL_SPLITS,
    check_name: str,
) -> tuple[torch.nn.Module, DataLoader[VideoDataset]]:
    """Build a run's model from its checkpoint, plus the test loader for one set.

    See `infer` for why the data comes from `data_info.json`.
    """
    save_path = Path(admin.save_path)
    data = load_test_sizes(save_path.parent)
    test_loader, num_classes, _, _ = setup_data(set_name, split_name, data)

    model = get_model(admin.model, num_classes, 0.0)
    checkpoint = torch.load(save_path / check_name)
    model.load_state_dict(checkpoint)
    return model, test_loader


def infer(
    admin: MinInfo,
    set_name: AVAIL_SETS,
    split_name: AVAIL_SPLITS,
    check_name: str = "best.pth",
) -> tuple[BaseRes, dict[str, dict[str, float]], list[int], list[int]]:
    """
    Load a trained model from its checkpoint and run it over one dataset set,
    in the spirit of FrameFetcher/FrameVisualiser: bundles the data/model/
    checkpoint plumbing behind one call instead of repeating it per notebook.

    admin: identifies which run's checkpoint directory to load
        (admin.save_path / check_name). Only `MinInfo` (not the full
        `AdminInfo`) is needed -- see below.
    set_name/split_name: which dataset set to build the test loader for --
        kept separate from admin.split since a checkpoint from one split can
        be evaluated against another (e.g. a subset derived from it).
    check_name: checkpoint filename within admin.save_path. Defaults to
        "best.pth", the convention testing.py writes to.

    Data is loaded via `load_test_sizes` from the `data_info.json` that
    `train_loop` writes next to every run's checkpoints (regular or sweep
    trial) -- NOT by re-parsing `admin.config_path`. For a sweep trial,
    `config_path` points at the shared, untuned base sweep config rather
    than that trial's resolved hyperparameters, so re-parsing it would
    silently build the wrong `DataInfo`; `data_info.json` always holds what
    the run actually trained with.

    Returns (topk_res, cls_report, all_targets, all_preds), matching
    test_topk_clsrep's return shape so existing downstream code (e.g.
    sorting cls_report by per-gloss f1-score) can be reused as-is.
    """
    model, test_loader = _load_trained(admin, set_name, split_name, check_name)
    return test_topk_clsrep(model=model, test_loader=test_loader)


def infer_instance_topk(
    admin: MinInfo,
    set_name: AVAIL_SETS,
    split_name: AVAIL_SPLITS,
    check_name: str = "best.pth",
    max_k: int = 20,
) -> list[InstanceTopK]:
    """
    Like `infer`, but returns each instance's `max_k` most probable classes
    (via `test_instance_topk`) instead of aggregate metrics -- the input for
    plot_frame_grid_topk/animate_frames_topk. `max_k` caps the `k` those can
    later plot from the stashed result.
    """
    model, test_loader = _load_trained(admin, set_name, split_name, check_name)
    return test_instance_topk(model=model, test_loader=test_loader, max_k=max_k)
