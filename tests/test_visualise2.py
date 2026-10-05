import base64
import io
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.colors import to_hex
from matplotlib.container import BarContainer
from PIL import Image

matplotlib.use("Agg")

import src.visualise2 as v2
from src.preprocess import Instance
from src.run_types import MinInfo
from src.visualise2 import (
    CONTROL_COLORS,
    DEFAULT_ACCENT,
    SPLIT_NAME_MAP,
    TRUE_CLASS_COLOR,
    BarPosition,
    animate_frames,
    animate_frames_topk,
    animation_html,
    load_instance_frames,
    plot_bboxes_on_canvas,
    plot_dimension_distributions,
    plot_frame_grid_topk,
    plot_metric_correlation,
    save_fig,
    split_name_mapper,
    suggest_palette,
)


def _instance(bbox: list[int], label_name: str) -> Instance:
    return Instance(
        bbox=bbox,
        frame_end=10,
        frame_start=0,
        instance_id=0,
        signer_id=1,
        source="src",
        split="train",
        url="",
        variation_id=0,
        video_id="v1",
        label_num=0,
        label_name=label_name,
    )


class TestSplitNameMapper:
    def test_maps_known_split(self) -> None:
        assert split_name_mapper("asl100") == "WLASL-100"
        assert split_name_mapper("asl2000_cutoff_9") == "WLASL-2000"

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(KeyError):
            split_name_mapper("not_a_real_split")  # type: ignore[arg-type]


class TestSuggestPalette:
    def test_recognises_controls_case_insensitively(self) -> None:
        palette = suggest_palette(["Baseline", "no_aug", "spatial_crop"])
        assert palette == [
            CONTROL_COLORS["baseline"],
            CONTROL_COLORS["no_aug"],
            DEFAULT_ACCENT,
        ]

    def test_ignores_controls_when_disabled(self) -> None:
        palette = suggest_palette(["baseline", "spatial_crop"], recognise_controls=False)
        assert palette == [DEFAULT_ACCENT, DEFAULT_ACCENT]

    def test_empty_categories(self) -> None:
        assert suggest_palette([]) == []


class TestSaveFig:
    def test_creates_parent_dirs_and_writes_file(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        dest = tmp_path / "nested" / "does" / "not" / "exist" / "plot.pdf"
        returned = save_fig(fig, dest)

        assert returned == dest
        assert dest.exists()
        plt.close(fig)


class TestInfer:
    def test_wires_config_model_and_checkpoint_together(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # infer() needs a real trained checkpoint + dataset on disk to run for
        # real, so this checks the wiring (right args flow to the right
        # calls) with every collaborator mocked, rather than gating on
        # weights/dataset availability like a true smoke test would.
        admin = MinInfo(
            model="S3D",
            split="asl100",
            save_path=str(tmp_path),
        )
        fake_data = "data-info-sentinel"
        fake_loader = MagicMock()
        fake_model = MagicMock()
        fake_state_dict = {"sentinel": True}
        expected_result = ("topk_res", "cls_report", [1], [2])

        load_test_sizes_calls = []
        setup_data_calls = []
        get_model_calls = []

        monkeypatch.setattr(
            v2,
            "load_test_sizes",
            lambda save_dir: (load_test_sizes_calls.append(save_dir), fake_data)[1],
        )
        monkeypatch.setattr(
            v2,
            "setup_data",
            lambda set_name, split, data_info: (
                setup_data_calls.append((set_name, split, data_info)),
                (fake_loader, 10, None, None),
            )[1],
        )
        monkeypatch.setattr(
            v2,
            "get_model",
            lambda model_name, num_classes, drop_p: (
                get_model_calls.append((model_name, num_classes, drop_p)),
                fake_model,
            )[1],
        )
        monkeypatch.setattr(v2.torch, "load", lambda path: fake_state_dict)
        monkeypatch.setattr(
            v2, "test_topk_clsrep", lambda model, test_loader: expected_result
        )

        result = v2.infer(admin, "test", "asl100", check_name="best.pth")

        assert result == expected_result
        assert load_test_sizes_calls == [tmp_path.parent]
        assert setup_data_calls == [("test", "asl100", "data-info-sentinel")]
        assert get_model_calls == [("S3D", 10, 0.0)]
        fake_model.load_state_dict.assert_called_once_with(fake_state_dict)


class TestPlotBboxesOnCanvas:
    def test_averages_per_class_by_default(self) -> None:
        import matplotlib.pyplot as plt

        instances = [
            _instance([0, 0, 10, 10], "book"),
            _instance([10, 10, 30, 30], "book"),
            _instance([100, 100, 120, 120], "dog"),
        ]

        fig, ax = plot_bboxes_on_canvas(instances)

        # one rectangle patch per distinct class once averaged, not per instance
        assert len(ax.patches) == 2
        plt.close(fig)

    def test_average_false_draws_every_instance(self) -> None:
        import matplotlib.pyplot as plt

        instances = [
            _instance([0, 0, 10, 10], "book"),
            _instance([10, 10, 30, 30], "book"),
        ]

        fig, ax = plot_bboxes_on_canvas(instances, average=False)

        assert len(ax.patches) == 2
        plt.close(fig)


class TestPlotDimensionDistributions:
    def test_returns_two_axes_for_width_and_height(self) -> None:
        import matplotlib.pyplot as plt

        instances = [
            _instance([0, 0, 10, 20], "book"),
            _instance([0, 0, 15, 25], "book"),
        ]

        fig, axes = plot_dimension_distributions(instances)

        assert len(axes) == 2
        plt.close(fig)


def test_split_name_map_only_maps_known_avail_splits() -> None:
    # Every value should be a non-empty display name; guards against typos
    # silently mapping a split to an empty/placeholder string.
    assert all(isinstance(v, str) and v for v in SPLIT_NAME_MAP.values())


def test_plot_metric_correlation_log_x() -> None:
    x = [1e-5, 1e-4, 1e-3, 1e-2]
    y = [4.0, 3.0, 2.0, 1.0]
    _, ax, tau, _ = plot_metric_correlation(x, y, log_x=True)
    assert ax.get_xscale() == "log"
    assert tau == pytest.approx(-1.0)
    # linear in log10(x), so the fit line passes exactly through the points
    line_x, line_y = (np.asarray(d) for d in ax.get_lines()[0].get_data())
    assert line_x[0] == pytest.approx(1e-5) and line_x[-1] == pytest.approx(1e-2)
    assert line_y[0] == pytest.approx(4.0) and line_y[-1] == pytest.approx(1.0)


class TestFrameGridTopK:
    labels: ClassVar[list[str]] = ["a", "b", "c", "d"]
    scores: ClassVar[list[float]] = [0.1, 0.4, 0.2, 0.3]

    @staticmethod
    def _frames(t: int = 8) -> torch.Tensor:
        return torch.rand(t, 3, 16, 16)

    def test_bars_are_top_k_in_descending_order(self) -> None:
        _, _, bar_ax = plot_frame_grid_topk(self._frames(), self.labels, self.scores, k=3)
        assert [t.get_text() for t in bar_ax.get_yticklabels()] == ["b", "d", "c"]
        bars = bar_ax.containers[0]
        assert isinstance(bars, BarContainer)
        assert np.asarray(bars.datavalues) == pytest.approx([0.4, 0.3, 0.2])

    def test_true_label_highlighted(self) -> None:
        _, _, bar_ax = plot_frame_grid_topk(
            self._frames(), self.labels, self.scores, k=3, true_label="d"
        )
        colours = [to_hex(p.get_facecolor()) for p in bar_ax.patches]
        assert colours == [DEFAULT_ACCENT.lower(), TRUE_CLASS_COLOR.lower(), DEFAULT_ACCENT.lower()]

    @pytest.mark.parametrize(
        ("position", "horizontal"),
        [("left", True), ("right", True), ("top", False), ("bottom", False)],
    )
    def test_bar_position_sets_side_and_orientation(
        self, position: BarPosition, horizontal: bool
    ) -> None:
        fig, frame_axes, bar_ax = plot_frame_grid_topk(
            self._frames(), self.labels, self.scores, k=4, bar_position=position, num=8, cols=4
        )
        fig.canvas.draw()
        assert frame_axes.shape == (2, 4)
        # display coords -- get_position() is relative to each axes' own subfigure
        bar_box = bar_ax.get_window_extent()
        first_frame = frame_axes[0][0].get_window_extent()
        last_frame = frame_axes[-1][-1].get_window_extent()
        if position == "left":
            assert bar_box.x1 < first_frame.x0
        elif position == "right":
            assert bar_box.x0 > last_frame.x1
        elif position == "top":
            assert bar_box.y0 > first_frame.y1
        else:
            assert bar_box.y1 < last_frame.y0
        tick_labels = bar_ax.get_yticklabels() if horizontal else bar_ax.get_xticklabels()
        assert [t.get_text() for t in tick_labels] == ["b", "d", "c", "a"]

    @pytest.mark.parametrize(("shared_axis", "expected_max"), [(True, 1.15), (False, 0.4 * 1.15)])
    def test_shared_axis_fixes_confidence_range(
        self, shared_axis: bool, expected_max: float
    ) -> None:
        _, _, bar_ax = plot_frame_grid_topk(
            self._frames(), self.labels, self.scores, k=3, shared_axis=shared_axis
        )
        assert bar_ax.get_xlim() == pytest.approx((0, expected_max))

    @pytest.mark.parametrize("k", [0, 5])
    def test_k_out_of_range_raises(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be"):
            plot_frame_grid_topk(self._frames(), self.labels, self.scores, k=k)

    def test_misaligned_labels_raise(self) -> None:
        with pytest.raises(ValueError, match="labels"):
            plot_frame_grid_topk(self._frames(), self.labels[:3], self.scores)

    def test_animation_plays_every_frame(self) -> None:
        frames = self._frames(t=5)
        _, anim, bar_ax = animate_frames_topk(frames, self.labels, self.scores, k=2)
        assert len(bar_ax.patches) == 2
        html = anim.to_jshtml()
        assert html.count("data:image/png;base64") == len(frames)


class TestAnimateFrames:
    def test_plays_every_frame(self) -> None:
        frames = torch.rand(5, 3, 16, 16)
        _, anim = animate_frames(frames)
        assert anim.to_jshtml().count("data:image/png;base64") == len(frames)

    @pytest.mark.parametrize(("scale", "expected"), [(1.0, (16, 24)), (2.0, (32, 48))])
    def test_figure_matches_frame_pixels(self, scale: float, expected: tuple[int, int]) -> None:
        fig, _ = animate_frames(torch.rand(2, 3, 24, 16), scale=scale)
        assert tuple(np.round(fig.get_size_inches() * fig.dpi)) == expected
        assert fig.axes[0].get_position().bounds == (0.0, 0.0, 1.0, 1.0)

    def test_title_adds_strip_above_frames(self) -> None:
        fig, _ = animate_frames(torch.rand(2, 3, 24, 16), title="t")
        _, frame_bottom, _, frame_height = fig.axes[0].get_position().bounds
        assert frame_bottom == 0.0 and frame_height < 1.0
        assert round(fig.get_size_inches()[1] * fig.dpi * frame_height) == 24

    def test_html_renders_at_figure_dpi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(plt.rcParams, "savefig.dpi", 300)
        fig, anim = animate_frames(torch.rand(2, 3, 24, 16))
        html = animation_html(fig, anim).data
        assert isinstance(html, str)
        jpeg = base64.b64decode(html.split("data:image/jpeg;base64,")[1].split('"')[0].replace("\\n", ""))
        assert Image.open(io.BytesIO(jpeg)).size == (16, 24)

    def test_html_closes_figure(self) -> None:
        fig, anim = animate_frames(torch.rand(2, 3, 16, 16))
        html = animation_html(fig, anim)
        assert not plt.fignum_exists(fig.number)
        assert isinstance(html.data, str) and "data:image/jpeg;base64" in html.data


def test_load_instance_frames_uses_labelled_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(
        v2, "load_rgb_frames_from_video", lambda *args: (calls.append(args), torch.zeros(1))[1]
    )
    inst = _instance([0, 0, 1, 1], "a").model_copy(update={"frame_start": 3, "frame_end": 7})
    video = tmp_path / f"{inst.video_id}.mp4"
    video.touch()
    load_instance_frames(inst, tmp_path)
    assert calls == [(video, 3, 7)]


class TestInferInstanceTopK:
    def test_runs_instance_topk_on_loaded_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        admin = MinInfo(model="S3D", split="asl100", save_path="unused")
        fake_model, fake_loader = MagicMock(), MagicMock()
        calls = []
        monkeypatch.setattr(
            v2, "_load_trained", lambda *args: (calls.append(args), (fake_model, fake_loader))[1]
        )
        monkeypatch.setattr(
            v2,
            "test_instance_topk",
            lambda model, test_loader, max_k: (model, test_loader, max_k),
        )

        result = v2.infer_instance_topk(admin, "test", "asl100", max_k=7)

        assert result == (fake_model, fake_loader, 7)
        assert calls == [(admin, "test", "asl100", "best.pth")]
