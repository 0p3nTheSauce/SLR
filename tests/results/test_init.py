from pathlib import Path

from pydantic import BaseModel

from src.results import (
    format_exp_label,
    get_asset_path,
    get_out_stub,
    match,
    same_augs,
    stash_split_name,
)


class _Aug(BaseModel):
    type: str
    frame_size: int


class TestMatch:
    def test_dict_subset_matches(self) -> None:
        obj = {"a": 1, "b": 2, "c": 3}
        assert match(obj, {"a": 1, "b": 2})

    def test_dict_mismatch_fails(self) -> None:
        obj = {"a": 1, "b": 2}
        assert not match(obj, {"a": 1, "b": 3})

    def test_both_none(self) -> None:
        assert match(None, None)

    def test_one_none_fails(self) -> None:
        assert not match(None, {"a": 1})
        assert not match({"a": 1}, None)

    def test_pydantic_model_dumped_before_comparison(self) -> None:
        obj = _Aug(type="Centre_crop", frame_size=224)
        assert match(obj, {"type": "Centre_crop"})
        assert not match(obj, {"type": "Random_crop"})


class TestSameAugs:
    def test_equal_lists_of_dicts(self) -> None:
        augs = [{"type": "Centre_crop", "frame_size": 224}]
        assert same_augs(augs, list(augs))

    def test_different_lengths(self) -> None:
        assert not same_augs([{"type": "a"}], [{"type": "a"}, {"type": "b"}])

    def test_different_order_fails(self) -> None:
        a = [{"type": "a"}, {"type": "b"}]
        b = [{"type": "b"}, {"type": "a"}]
        assert not same_augs(a, b)

    def test_pydantic_models_compared_by_value(self) -> None:
        a = [_Aug(type="Centre_crop", frame_size=224)]
        b = [_Aug(type="Centre_crop", frame_size=224)]
        assert same_augs(a, b)

    def test_empty_lists_are_equal(self) -> None:
        assert same_augs([], [])


class TestFormatExpLabel:
    def test_numeric_exp_no_gets_exp_prefix(self) -> None:
        assert format_exp_label("000") == "exp000"
        assert format_exp_label("12") == "exp12"

    def test_sweep_run_id_left_as_is(self) -> None:
        assert format_exp_label("3f8a1b2c") == "3f8a1b2c"


class TestStashSplitName:
    def test_cutoff_1_indexed_maps_back(self) -> None:
        assert stash_split_name("asl100_cutoff_9_1_indexed") == "asl100_cutoff_9"

    def test_others_unchanged(self) -> None:
        for split in ("asl100_cutoff_9", "asl100_1_indexed", "asl2000", "asl100_worst"):
            assert stash_split_name(split) == split


class TestGetOutStub:
    def test_plain_split(self) -> None:
        assert get_out_stub("asl100", "S3D", "exp001") == "asl100_S3D_exp001_"

    def test_cutoff_1_indexed_keeps_the_old_stub(self) -> None:
        """Existing stash files were named before the `*_cutoff_9` splits were renamed."""
        old = get_out_stub("asl2000_cutoff_9", "MViTv2_S", "exp002")
        assert get_out_stub("asl2000_cutoff_9_1_indexed", "MViTv2_S", "exp002") == old
        assert old == "asl2000_cutoff_9_MViTv2_S_exp002_"

    def test_plain_1_indexed_split_unchanged(self) -> None:
        assert get_out_stub("asl100_1_indexed", "S3D", "exp001") == "asl100_1_indexed_S3D_exp001_"


class TestGetAssetPath:
    def test_plain_descriptor_unchanged(self, tmp_path: Path) -> None:
        path = get_asset_path(
            "corr_signers", "asl100_S3D_exp000_", "proj", asset_dir=tmp_path
        )
        assert path == tmp_path / "proj" / "corr_signers_asl100_S3D_exp000_.pdf"

    def test_spaces_and_punctuation_replaced(self, tmp_path: Path) -> None:
        path = get_asset_path(
            "overachiever_don't want", "stub_", "proj", asset_dir=tmp_path
        )
        assert path.name == "overachiever_don_t_want_stub_.pdf"

    def test_dot_in_descriptor_keeps_full_name(self, tmp_path: Path) -> None:
        path = get_asset_path("lr_1e-3.5", "stub_", "proj", ".png", asset_dir=tmp_path)
        assert path.name == "lr_1e-3_5_stub_.png"
