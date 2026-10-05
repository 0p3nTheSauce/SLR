"""Helpers for `all_misspredictions.ipynb`: read an instance's stored predictions, and
show clips (with their top-k chart, alone, or a gloss's every instance) as frame grids
or videos, per the notebook's `as_video` setting."""

from collections.abc import Sequence
from dataclasses import dataclass

from IPython.display import display
from matplotlib.figure import Figure
from torch import Tensor

from src.preprocess import Instance
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
)


def top1_gloss(pred: InstanceTopK, classes: Sequence[str]) -> str:
    return classes[pred.topk_idxs[0]]


def true_rank(pred: InstanceTopK, max_k: int) -> str:
    """The true gloss's 1-based rank among the stored predictions, or `>max_k`."""
    if pred.target not in pred.topk_idxs:
        return f">{max_k}"
    return str(pred.topk_idxs.index(pred.target) + 1)


@dataclass(frozen=True)
class ClipView:
    """How the notebook shows clips: as videos if `as_video`, else frame grids.

    `num_frames` is how many frames a frame grid samples; `grid_cell_size` sizes a
    plain (chart-less) frame grid, the others size themselves. `video_scale`/
    `video_interval` are `animate_frames_topk`'s `scale`/`interval`.
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

    def show_topk(self, clip: Tensor, pred: InstanceTopK, true_label: str) -> Figure | None:
        """`clip` with `pred`'s top-k chart. Returns the frame grid's figure, to save, or
        None for a video (already displayed)."""
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
            return None
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

    def show_clip(self, clip: Tensor) -> None:
        """`clip` alone, for instances with no stored predictions to chart."""
        if self.as_video:
            fig, anim = animate_frames(clip, scale=self.video_scale, interval=self.video_interval)
            display(animation_html(fig, anim))
        else:
            plot_frame_grid(clip, num=self.num_frames, size=self.grid_cell_size)

    def show_instances(
        self, fetched: Sequence[tuple[Instance, Tensor]], title: str
    ) -> Figure | None:
        """Every fetched clip (e.g. `FrameFetcher.fetch_all()`), one per cell, titled by
        signer. Stills show each clip's middle frame. Returns the still figure, or None
        for a video (already displayed)."""
        clips = [frames for _, frames in fetched]
        titles = [f"signer {inst.signer_id}" for inst, _ in fetched]
        if self.as_video:
            fig, anim = animate_clip_grid(
                clips, interval=self.video_interval, titles=titles, title=title
            )
            display(animation_html(fig, anim))
            return None
        fig, _ = plot_clip_grid(clips, titles=titles, title=title)
        return fig
