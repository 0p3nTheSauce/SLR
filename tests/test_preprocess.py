import json
from pathlib import Path

import pytest

from src.preprocess import (
    CacheEntry,
    Instance,
    RawInstance,
    WLASLClass,
    load_instance_cache,
    preprocess_split,
)

PLACEHOLDER_BBOX = [137, 16, 492, 480]


def _raw_instance(video_id: str) -> RawInstance:
    return RawInstance(
        bbox=list(PLACEHOLDER_BBOX),
        frame_end=20,
        frame_start=0,
        instance_id=0,
        signer_id=0,
        source="test",
        split="train",
        url="",
        variation_id=0,
        video_id=video_id,
    )


def _write_split(split_path: Path, video_ids: list[str]) -> None:
    split = [
        WLASLClass(gloss="book", instances=[_raw_instance(vid) for vid in video_ids])
    ]
    split_path.write_text(json.dumps([c.model_dump() for c in split]))


def _fake_bbox_for(video_id: str) -> list[int]:
    """A distinct, non-placeholder "detected" bbox per video, so tests can tell whether
    real bbox-fixing ran versus the raw placeholder bbox passing straight through."""
    n = int(video_id)
    return [n, n, n + 1, n + 1]


@pytest.fixture
def stub_fixers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stubs out the video-dependent fixers so tests don't need real video files or a
    YOLO model. Returns the list of video_ids that fix_bad_bboxes was actually called
    with, across the whole test, so tests can assert on when (not) it ran."""
    bbox_fix_calls: list[str] = []

    def fake_fix_bad_frame_range(
        *, instances: list[Instance], **_: object
    ) -> list[Instance]:
        return instances

    def fake_fix_bad_bboxes(
        *, instances: list[Instance], **_: object
    ) -> list[Instance]:
        for inst in instances:
            bbox_fix_calls.append(inst.video_id)
            inst.bbox = _fake_bbox_for(inst.video_id)
        return instances

    monkeypatch.setattr("src.preprocess.fix_bad_frame_range", fake_fix_bad_frame_range)
    monkeypatch.setattr("src.preprocess.fix_bad_bboxes", fake_fix_bad_bboxes)
    return bbox_fix_calls


class TestPreprocessSplitCache:
    def test_do_bboxes_false_leaves_placeholder_bbox(
        self, tmp_path: Path, stub_fixers: list[str]
    ) -> None:
        split_path = tmp_path / "asl100.json"
        _write_split(split_path, ["00001", "00002"])
        raw_path = tmp_path / "raw"
        raw_path.mkdir()
        output_base = tmp_path / "out"
        output_base.mkdir()

        preprocess_split(
            split_path=split_path,
            raw_path=raw_path,
            output_base=output_base,
            do_bboxes=False,
            length_cuttoff=0,
        )

        assert stub_fixers == []  # bbox fixing never invoked
        out = json.loads(
            (output_base / "asl100" / "train_fixed_frange_bboxes.json").read_text()
        )
        assert [inst["bbox"] for inst in out] == [PLACEHOLDER_BBOX, PLACEHOLDER_BBOX]

        cache = load_instance_cache(output_base / "instance_cache.json")
        assert cache["00001"].bboxes_fixed is False
        assert cache["00001"].instance.bbox == PLACEHOLDER_BBOX

    def test_later_do_bboxes_true_run_fixes_previously_cached_instances(
        self, tmp_path: Path, stub_fixers: list[str]
    ) -> None:
        """Regression test: a cache built by a do_bboxes=False run must not permanently
        poison later do_bboxes=True runs with the raw placeholder bbox."""
        split_path = tmp_path / "asl100.json"
        _write_split(split_path, ["00001", "00002"])
        raw_path = tmp_path / "raw"
        raw_path.mkdir()
        output_base = tmp_path / "out"
        output_base.mkdir()

        preprocess_split(
            split_path=split_path,
            raw_path=raw_path,
            output_base=output_base,
            do_bboxes=False,
            length_cuttoff=0,
        )
        assert stub_fixers == []

        preprocess_split(
            split_path=split_path,
            raw_path=raw_path,
            output_base=output_base,
            do_bboxes=True,
            length_cuttoff=0,
        )

        assert sorted(stub_fixers) == ["00001", "00002"]
        out = json.loads(
            (output_base / "asl100" / "train_fixed_frange_bboxes.json").read_text()
        )
        bboxes = {inst["video_id"]: inst["bbox"] for inst in out}
        assert bboxes["00001"] == _fake_bbox_for("00001")
        assert bboxes["00002"] == _fake_bbox_for("00002")

        cache = load_instance_cache(output_base / "instance_cache.json")
        assert cache["00001"].bboxes_fixed is True
        assert cache["00001"].instance.bbox == _fake_bbox_for("00001")

    def test_already_bbox_fixed_cache_entries_are_not_recomputed(
        self, tmp_path: Path, stub_fixers: list[str]
    ) -> None:
        split_path = tmp_path / "asl100.json"
        _write_split(split_path, ["00001"])
        raw_path = tmp_path / "raw"
        raw_path.mkdir()
        output_base = tmp_path / "out"
        output_base.mkdir()

        preprocess_split(
            split_path=split_path,
            raw_path=raw_path,
            output_base=output_base,
            do_bboxes=True,
            length_cuttoff=0,
        )
        assert stub_fixers == ["00001"]

        preprocess_split(
            split_path=split_path,
            raw_path=raw_path,
            output_base=output_base,
            do_bboxes=True,
            length_cuttoff=0,
        )

        # no new calls: the already bbox-fixed cache entry was reused as-is
        assert stub_fixers == ["00001"]


class TestLoadInstanceCache:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_instance_cache(tmp_path / "missing.json") == {}

    def test_old_format_is_ignored_not_misread(self, tmp_path: Path) -> None:
        """Old caches were a flat list of Instance dicts (no bboxes_fixed marker). Loading
        one must not silently misinterpret it as new-format entries."""
        cache_path = tmp_path / "instance_cache.json"
        old_format = [
            _raw_instance("00001").model_dump() | {"label_num": 0, "label_name": "x"}
        ]
        cache_path.write_text(json.dumps(old_format))

        assert load_instance_cache(cache_path) == {}

    def test_round_trips_through_save(self, tmp_path: Path) -> None:
        from src.preprocess import save_instance_cache

        cache_path = tmp_path / "instance_cache.json"
        inst = Instance(
            **_raw_instance("00001").model_dump(), label_num=0, label_name="book"
        )
        save_instance_cache(
            cache_path, {"00001": CacheEntry(instance=inst, bboxes_fixed=True)}
        )

        loaded = load_instance_cache(cache_path)
        assert loaded["00001"].bboxes_fixed is True
        assert loaded["00001"].instance.video_id == "00001"
