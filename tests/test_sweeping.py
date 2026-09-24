from pathlib import Path

import pytest

from src.sweeping import (
    SweepConfigError,
    apply_sweep_overrides,
    extract_sweep_values,
    validate_sweep_key_map,
)

KEY_MAP = {
    "lr": "optimizer.lr",
    "hflip_p": "augs.spatial.type:HFLIP.p",
    "frame_size": ["augs.spatial.type:CROP.size", "test.crop.size"],
}


def _skeleton() -> dict:
    return {
        "optimizer": {"lr": None},
        "augs": {"spatial": [{"type": "HFLIP", "p": None}, {"type": "CROP", "size": None}]},
        "test": {"crop": {"size": None}},
        "patience": None,
    }


class TestExtractSweepValues:
    def test_inverts_apply_sweep_overrides(self) -> None:
        values = {"lr": 1e-3, "hflip_p": 0.2, "frame_size": 224}
        config = apply_sweep_overrides(_skeleton(), values, KEY_MAP)
        assert extract_sweep_values(config, KEY_MAP, list(values)) == values

    def test_unmapped_name_read_as_dotted_path(self) -> None:
        config = apply_sweep_overrides(_skeleton(), {"patience": 15}, KEY_MAP)
        assert extract_sweep_values(config, KEY_MAP, ["patience"]) == {"patience": 15}

    def test_unresolvable_path_raises(self) -> None:
        with pytest.raises(KeyError):
            extract_sweep_values(_skeleton(), {"x": "augs.spatial.type:MISSING.p"}, ["x"])


class TestValidateSweepKeyMap:
    def _write_base(self, tmp_path: Path, key_map: dict) -> Path:
        base = tmp_path / "base.py"
        base.write_text(f"base_config = {_skeleton()!r}\nsweep_key_map = {key_map!r}\n")
        return base

    def test_valid_map_passes(self, tmp_path: Path) -> None:
        validate_sweep_key_map(self._write_base(tmp_path, KEY_MAP))

    def test_bad_target_in_list_raises(self, tmp_path: Path) -> None:
        bad = KEY_MAP | {"frame_size": ["augs.spatial.type:CROP.size", "test.nope.size"]}
        with pytest.raises(SweepConfigError, match="test.nope.size"):
            validate_sweep_key_map(self._write_base(tmp_path, bad))
