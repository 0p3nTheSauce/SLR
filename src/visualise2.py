"""
Shared plotting style utilities for the thesis (Chapter 5 and onward).

Usage:
    from thesis_plot_style import set_thesis_style, plot_bar_chart

    set_thesis_style()  # call once, at the top of a notebook/script
    fig, ax = plot_bar_chart(df["config_name"], df["test_loss"], xlabel="Config", ylabel="Test Loss")
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from logging import Logger
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from typing_extensions import TypedDict, Unpack

from src.configs import get_class_list
from src.preprocess import Instance

# locals
from src.run_types import AVAIL_SETS, AVAIL_SPLITS, CentreCropConfig, OG_Sampler
from src.utils import load_rgb_frames_from_video, plt_display_grid
from src.video_dataset import (
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
            ax.bar_label(container, fmt=value_fmt, padding=3)

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

    for i, name in enumerate(series_names):
        label = legend_labels.get(name, name) if legend_labels else name
        values = np.asarray(ys[name], dtype=float)
        container = ax.bar(x_pos, values, width, bottom=bottoms, label=label, color=palette[i])
        if show_values:
            for bar, val, bot in zip(container, values, bottoms):
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
        ex_path = get_video_path(next_example.video_id, self.set_path_info["root"])

        self.logger.info(f"Next example video path: {ex_path}")

        return self.transform(
            load_rgb_frames_from_video(
                ex_path, next_example.frame_start, next_example.frame_end
            )
        )

    def __len__(self):
        return self.tot_samples


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
        
        plt_display_grid(frames, self.target_frames)
        
class FrameFetcher:
    def __init__(self, **kwargs: Unpack[MiniSetKwargs]):
        self.frames: Tensor | None = None
        self.cur_idx : int = 0
        dataloader = DataLoader(
                        MiniSet(**kwargs),
                        batch_size=1,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=False,
                    )
        
        self.iter_loader = iter(
            dataloader
        )
        self.len = len(dataloader)

    def __call__(self) -> Tensor:
        frames = next(self.iter_loader)[0]
        self.cur_idx += 1
        if len(frames.shape) == 5:
            frames = frames.squeeze(dim=0)
        if frames.shape[1] != 3:
            frames = frames.permute(1, 0, 2, 3)  # swap T and C
        
        return frames
