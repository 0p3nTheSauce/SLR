from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
import pytest

matplotlib.use("Agg")

import src.visualise2 as v2
from src.preprocess import Instance
from src.run_types import MinInfo
from src.visualise2 import (
    CONTROL_COLORS,
    DEFAULT_ACCENT,
    SPLIT_NAME_MAP,
    plot_bboxes_on_canvas,
    plot_dimension_distributions,
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
    line_x, line_y = ax.get_lines()[0].get_data()
    assert line_x[0] == pytest.approx(1e-5) and line_x[-1] == pytest.approx(1e-2)
    assert line_y[0] == pytest.approx(4.0) and line_y[-1] == pytest.approx(1.0)
