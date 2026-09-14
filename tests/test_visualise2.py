from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from src.visualise2 import (
    CONTROL_COLORS,
    DEFAULT_ACCENT,
    SPLIT_NAME_MAP,
    save_fig,
    split_name_mapper,
    suggest_palette,
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


def test_split_name_map_only_maps_known_avail_splits() -> None:
    # Every value should be a non-empty display name; guards against typos
    # silently mapping a split to an empty/placeholder string.
    assert all(isinstance(v, str) and v for v in SPLIT_NAME_MAP.values())
