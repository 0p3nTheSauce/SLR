"""Helpers for `all_misspredictions.ipynb`: read an instance's stored predictions, and
show clips (with their top-k chart, alone, or a gloss's every instance) as frame grids
or videos, per the notebook's `as_video` setting, and save what was shown."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, TypeAlias

from IPython.display import display
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from torch import Tensor

from src.preprocess import Instance
from src.results import get_asset_path
from src.run_types import InstanceTopK
from src.visualise2 import (
    PanelPosition,
    animate_clip_grid,
    animate_frames,
    animate_frames_topk,
    animation_html,
    plot_clip_grid,
    plot_frame_grid,
    plot_frame_grid_topk,
    save_animation,
    save_fig,
)


def top1_gloss(pred: InstanceTopK, classes: Sequence[str]) -> str:
    return classes[pred.topk_idxs[0]]


def true_rank(pred: InstanceTopK, max_k: int) -> str:
    """The true gloss's 1-based rank among the stored predictions, or `>max_k`."""
    if pred.target not in pred.topk_idxs:
        return f">{max_k}"
    return str(pred.topk_idxs.index(pred.target) + 1)


SortBy = Literal["none", "signer", "variation"]


def sort_instances(
    fetched: Sequence[tuple[Instance, Tensor]], sort_by: SortBy
) -> list[tuple[Instance, Tensor]]:
    """`fetched` ordered by ascending signer or variation id (set order among equal
    ids), or left in set order for "none"."""
    if sort_by == "none":
        return list(fetched)
    if sort_by == "signer":
        return sorted(fetched, key=lambda pair: pair[0].signer_id)
    return sorted(fetched, key=lambda pair: pair[0].variation_id)


def instance_label(inst: Instance, sep: str = ", ") -> str:
    return f"signer {inst.signer_id}{sep}variation {inst.variation_id}"


class VariationPick(NamedTuple):
    """One clip picked by `VariationStepper`: the `position`-th (1-based) of its
    variation's `total` instances."""

    instance: Instance
    frames: Tensor
    position: int
    total: int


class VariationStepper:
    """Step through a gloss's fetched clips (e.g. `FrameFetcher.fetch_all()`) one instance
    per variation at a time, so its different signs (`variation_id`) can be compared.

    Each call returns the next instance of every variation, in ascending variation id (set
    order within a variation). A variation with no instances left drops out until the one
    with the most has run out too; the next call then restarts every variation from its
    first instance.
    """

    def __init__(self, fetched: Sequence[tuple[Instance, Tensor]]) -> None:
        groups: dict[int, list[tuple[Instance, Tensor]]] = defaultdict(list)
        for inst, frames in fetched:
            groups[inst.variation_id].append((inst, frames))
        self.groups = dict(sorted(groups.items()))
        self.num_steps = max((len(group) for group in self.groups.values()), default=0)
        self.step = 0  # 1-based index of the most recent step, 0 before the first

    def counts(self) -> dict[int, int]:
        """Instances per variation id."""
        return {variation: len(group) for variation, group in self.groups.items()}

    def __call__(self) -> list[VariationPick]:
        if self.step == self.num_steps:
            self.step = 0
        self.step += 1
        return [
            VariationPick(*group[self.step - 1], position=self.step, total=len(group))
            for group in self.groups.values()
            if self.step <= len(group)
        ]


Shown: TypeAlias = Figure | tuple[Figure, FuncAnimation]
"""What a `ClipView` method showed: a still figure, or a video's (fig, anim)."""


def save_shown(shown: Shown, metric_descriptor: str, stub: str, project_name: str) -> Path:
    """Save a `ClipView` result under `get_asset_path`: a still as a PDF, a video as
    an MP4. Returns the saved path."""
    if isinstance(shown, Figure):
        return save_fig(shown, get_asset_path(metric_descriptor, stub, project_name))
    fig, anim = shown
    path = get_asset_path(metric_descriptor, stub, project_name, file_suffix=".mp4")
    return save_animation(fig, anim, path)


@dataclass(frozen=True)
class ClipView:
    """How the notebook shows clips: as videos if `as_video`, else frame grids.

    `num_frames` is how many frames a frame grid samples; `grid_cell_size` sizes a
    plain (chart-less) frame grid, the others size themselves. `video_scale`/
    `video_interval` are `animate_frames_topk`'s `scale`/`interval`. `grid_cols` is how
    many clips a row of `show_instances` holds.

    Each `show_*` method returns what it showed (`Shown`), for `save_shown`. A frame
    grid is left for the notebook to display; a video is displayed as a player.
    """

    classes: Sequence[str]
    k: int
    bar_position: PanelPosition
    shared_axis: bool
    num_frames: int
    grid_cell_size: tuple[float, float]
    as_video: bool
    video_scale: float
    video_interval: int
    grid_cols: int = 6

    def show_topk(self, clip: Tensor, pred: InstanceTopK, true_label: str) -> Shown:
        """`clip` with `pred`'s top-k chart."""
        labels = [self.classes[i] for i in pred.topk_idxs]
        if self.as_video:
            fig, anim, _ = animate_frames_topk(
                clip,
                labels=labels,
                scores=pred.topk_probs,
                k=self.k,
                true_label=true_label,
                bar_position=self.bar_position,
                shared_axis=self.shared_axis,
                scale=self.video_scale,
                interval=self.video_interval,
            )
            display(animation_html(fig, anim))
            return fig, anim
        fig, _, _ = plot_frame_grid_topk(
            clip,
            labels=labels,
            scores=pred.topk_probs,
            k=self.k,
            true_label=true_label,
            bar_position=self.bar_position,
            shared_axis=self.shared_axis,
            num=self.num_frames,
        )
        return fig

    def show_clip(self, clip: Tensor) -> Shown:
        """`clip` alone, for instances with no stored predictions to chart."""
        if self.as_video:
            fig, anim = animate_frames(clip, scale=self.video_scale, interval=self.video_interval)
            display(animation_html(fig, anim))
            return fig, anim
        fig, _ = plot_frame_grid(clip, num=self.num_frames, size=self.grid_cell_size)
        return fig

    def show_instances(
        self,
        fetched: Sequence[tuple[Instance, Tensor]],
        title: str,
        sort_by: SortBy = "none",
    ) -> Shown:
        """Every fetched clip (e.g. `FrameFetcher.fetch_all()`), one per cell, titled by
        signer and variation, ordered by `sort_by`. Stills show each clip's middle
        frame."""
        ordered = sort_instances(fetched, sort_by)
        clips = [frames for _, frames in ordered]
        titles = [instance_label(inst, sep="\n") for inst, _ in ordered]
        if self.as_video:
            fig, anim = animate_clip_grid(
                clips, self.grid_cols, interval=self.video_interval, titles=titles, title=title
            )
            display(animation_html(fig, anim))
            return fig, anim
        fig, _ = plot_clip_grid(clips, self.grid_cols, titles=titles, title=title)
        return fig
