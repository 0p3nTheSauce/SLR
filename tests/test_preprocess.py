import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.preprocess import (
    CACHE_VERSION,
    CacheEntry,
    Instance,
    PreprocessLog,
    RawInstance,
    RemovePolicy,
    WLASLClass,
    fix_bad_bboxes,
    fix_bad_frame_range,
    instance_to_processed,
    load_instance_cache,
    preprocess_split,
    remove_short_samples,
    save_instance_cache,
)

PLACEHOLDER_BBOX = [137, 16, 492, 480]


def _raw_instance(
    video_id: str, frame_start: int = 1, frame_end: int = 20
) -> RawInstance:
    """A WLASL-style instance: `frame_start` is 1-based."""
    return RawInstance(
        bbox=list(PLACEHOLDER_BBOX),
        frame_end=frame_end,
        frame_start=frame_start,
        instance_id=0,
        signer_id=0,
        source="test",
        split="train",
        url="",
        variation_id=0,
        video_id=video_id,
    )


def _instance(video_id: str, frame_start: int = 0, frame_end: int = 20) -> Instance:
    """A processed instance: frames are 0-based."""
    return Instance(
        **_raw_instance(video_id, frame_start, frame_end).model_dump(),
        label_num=0,
        label_name="book",
    )


def _entry(video_id: str, frame_start: int = 0, frame_end: int = 20) -> CacheEntry:
    return CacheEntry(
        instance=_instance(video_id, frame_start, frame_end), bboxes_fixed=False
    )


def _write_split(split_path: Path, instances: list[RawInstance]) -> None:
    split = [WLASLClass(gloss="book", instances=instances)]
    split_path.write_text(json.dumps([c.model_dump() for c in split]))


def _fake_bbox_for(video_id: str) -> list[int]:
    """A distinct, non-placeholder "detected" bbox per video, so tests can tell whether
    real bbox-fixing ran versus the raw placeholder bbox passing straight through."""
    n = int(video_id)
    return [n, n, n + 1, n + 1]


@pytest.fixture
def dirs(tmp_path: Path) -> SimpleNamespace:
    """split_path / raw_path / output_base under tmp_path, with the directories created."""
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    output_base = tmp_path / "out"
    output_base.mkdir()
    return SimpleNamespace(
        split_path=tmp_path / "asl100.json", raw_path=raw_path, output_base=output_base
    )


def _run(
    dirs: SimpleNamespace, split_path: Path | None = None, **kwargs: object
) -> None:
    preprocess_split(
        split_path=split_path or dirs.split_path,
        raw_path=dirs.raw_path,
        output_base=dirs.output_base,
        **kwargs,  # type: ignore[arg-type]
    )


def _cache_path(dirs: SimpleNamespace) -> Path:
    return dirs.output_base / f"instance_cache_v{CACHE_VERSION}.json"


def _read_log(dirs: SimpleNamespace, split: str = "asl100") -> PreprocessLog:
    return PreprocessLog.model_validate_json(
        (dirs.output_base / split / "preprocess_log.json").read_text()
    )


def _read_set(dirs: SimpleNamespace, split: str = "asl100") -> list[dict]:
    return json.loads(
        (dirs.output_base / split / "train_fixed_frange_bboxes.json").read_text()
    )


@pytest.fixture
def stub_fixers(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stubs out the video-dependent fixers so tests don't need real video files or a
    YOLO model. The frame-range stub resets any instance whose video_id is in
    `reset_ids`. Returns a namespace with `reset_ids` (for tests to fill) and
    `frame_calls`/`bbox_calls` (video_ids each fixer was called with, across the test)."""
    stubs = SimpleNamespace(reset_ids=set(), frame_calls=[], bbox_calls=[])

    def fake_fix_bad_frame_range(
        raw_path: Path, entries: list[CacheEntry], **_: object
    ) -> tuple[list[CacheEntry], list[CacheEntry]]:
        for entry in entries:
            stubs.frame_calls.append(entry.instance.video_id)
            if entry.instance.video_id in stubs.reset_ids:
                entry.record("frame_range", "reset", "stub reset")
        return entries, []

    def fake_fix_bad_bboxes(
        raw_path: Path, entries: list[CacheEntry], **_: object
    ) -> tuple[list[CacheEntry], list[CacheEntry]]:
        for entry in entries:
            stubs.bbox_calls.append(entry.instance.video_id)
            entry.instance.bbox = _fake_bbox_for(entry.instance.video_id)
            entry.bboxes_fixed = True
        return entries, []

    monkeypatch.setattr("src.preprocess.fix_bad_frame_range", fake_fix_bad_frame_range)
    monkeypatch.setattr("src.preprocess.fix_bad_bboxes", fake_fix_bad_bboxes)
    return stubs


class TestPreprocessSplitCache:
    def test_do_bboxes_false_leaves_placeholder_bbox(
        self, dirs: SimpleNamespace, stub_fixers: SimpleNamespace
    ) -> None:
        _write_split(dirs.split_path, [_raw_instance("00001"), _raw_instance("00002")])

        _run(dirs, do_bboxes=False, length_cuttoff=0)

        assert stub_fixers.bbox_calls == []  # bbox fixing never invoked
        assert [inst["bbox"] for inst in _read_set(dirs)] == [
            PLACEHOLDER_BBOX,
            PLACEHOLDER_BBOX,
        ]

        cache = load_instance_cache(_cache_path(dirs))
        assert cache["00001"].bboxes_fixed is False
        assert cache["00001"].instance.bbox == PLACEHOLDER_BBOX

    def test_later_do_bboxes_true_run_fixes_previously_cached_instances(
        self, dirs: SimpleNamespace, stub_fixers: SimpleNamespace
    ) -> None:
        """Regression test: a cache built by a do_bboxes=False run must not permanently
        poison later do_bboxes=True runs with the raw placeholder bbox."""
        _write_split(dirs.split_path, [_raw_instance("00001"), _raw_instance("00002")])

        _run(dirs, do_bboxes=False, length_cuttoff=0)
        assert stub_fixers.bbox_calls == []

        _run(dirs, do_bboxes=True, length_cuttoff=0)

        assert sorted(stub_fixers.bbox_calls) == ["00001", "00002"]
        bboxes = {inst["video_id"]: inst["bbox"] for inst in _read_set(dirs)}
        assert bboxes["00001"] == _fake_bbox_for("00001")
        assert bboxes["00002"] == _fake_bbox_for("00002")

        cache = load_instance_cache(_cache_path(dirs))
        assert cache["00001"].bboxes_fixed is True
        assert cache["00001"].instance.bbox == _fake_bbox_for("00001")

    def test_already_bbox_fixed_cache_entries_are_not_recomputed(
        self, dirs: SimpleNamespace, stub_fixers: SimpleNamespace
    ) -> None:
        _write_split(dirs.split_path, [_raw_instance("00001")])

        _run(dirs, do_bboxes=True, length_cuttoff=0)
        assert stub_fixers.bbox_calls == ["00001"]

        _run(dirs, do_bboxes=True, length_cuttoff=0)

        # no new calls: the already bbox-fixed cache entry was reused as-is
        assert stub_fixers.bbox_calls == ["00001"]

    def test_use_cache_false_reprocesses_but_keeps_other_entries(
        self, dirs: SimpleNamespace, stub_fixers: SimpleNamespace
    ) -> None:
        cache_path = _cache_path(dirs)
        save_instance_cache(cache_path, {"99999": _entry("99999")})
        _write_split(dirs.split_path, [_raw_instance("00001")])

        _run(dirs, do_bboxes=False, length_cuttoff=0)
        _run(dirs, do_bboxes=False, length_cuttoff=0, use_cache=False)

        assert stub_fixers.frame_calls == ["00001", "00001"]
        assert set(load_instance_cache(cache_path)) == {"00001", "99999"}


class TestPreprocessSplitLog:
    def test_log_is_written_when_nothing_changed(
        self, dirs: SimpleNamespace, stub_fixers: SimpleNamespace
    ) -> None:
        _write_split(dirs.split_path, [_raw_instance("00001")])

        _run(dirs, do_bboxes=False, length_cuttoff=0)

        train = _read_log(dirs).sets["train"]
        assert (train.num_raw, train.num_kept, train.num_removed) == (1, 1, 0)
        assert train.instances == []

    def test_log_accounts_for_every_raw_instance(
        self, dirs: SimpleNamespace, stub_fixers: SimpleNamespace
    ) -> None:
        stub_fixers.reset_ids = {"00002"}
        _write_split(
            dirs.split_path,
            [
                _raw_instance("00001"),
                _raw_instance("00002"),
                _raw_instance("00003", frame_end=5),
            ],
        )

        _run(dirs, do_bboxes=False, length_cuttoff=9)

        log = _read_log(dirs, "asl100_cutoff_9")
        train = log.sets["train"]
        assert (train.num_raw, train.num_kept, train.num_removed) == (3, 2, 1)
        assert train.counts == {"frame_range:reset": 1, "length:removed": 1}
        assert {inst.video_id for inst in train.instances} == {"00002", "00003"}
        assert [inst["video_id"] for inst in _read_set(dirs, "asl100_cutoff_9")] == [
            "00001",
            "00002",
        ]

    def test_cached_instances_keep_their_fix_records(
        self, dirs: SimpleNamespace, stub_fixers: SimpleNamespace
    ) -> None:
        """A split run after another that shares its instances must still log their fixes."""
        stub_fixers.reset_ids = {"00001"}
        _write_split(dirs.split_path, [_raw_instance("00001")])
        _run(dirs, do_bboxes=False, length_cuttoff=0)

        asl300_path = dirs.split_path.with_name("asl300.json")
        _write_split(asl300_path, [_raw_instance("00001")])
        _run(dirs, split_path=asl300_path, do_bboxes=False, length_cuttoff=0)

        assert stub_fixers.frame_calls == ["00001"]  # second split used the cache
        assert _read_log(dirs, "asl300").sets["train"].counts == {
            "frame_range:reset": 1
        }

    def test_cutoff_applies_to_cached_instances(
        self, dirs: SimpleNamespace, stub_fixers: SimpleNamespace
    ) -> None:
        """Regression test: a short sample cached by a cutoff-0 run must still be removed
        by a later cutoff-9 run."""
        _write_split(dirs.split_path, [_raw_instance("00001", frame_end=5)])

        _run(dirs, do_bboxes=False, length_cuttoff=0)
        _run(dirs, do_bboxes=False, length_cuttoff=9)

        assert stub_fixers.frame_calls == ["00001"]
        assert _read_set(dirs, "asl100_cutoff_9") == []
        assert _read_log(dirs, "asl100_cutoff_9").sets["train"].counts == {
            "length:removed": 1
        }


class TestZeroBasedFrames:
    def test_start_is_converted_and_end_kept(self) -> None:
        inst = instance_to_processed(_raw_instance("00001", 1, 20), 0, "book")

        assert (inst.frame_start, inst.frame_end) == (0, 20)

    def test_labels_are_written_zero_based(
        self, dirs: SimpleNamespace, stub_fixers: SimpleNamespace
    ) -> None:
        _write_split(dirs.split_path, [_raw_instance("00001", 1, 20)])

        _run(dirs, do_bboxes=False, length_cuttoff=0)

        ((start, end),) = [(i["frame_start"], i["frame_end"]) for i in _read_set(dirs)]
        assert (start, end) == (0, 20)

    def test_cutoff_counts_every_annotated_frame(
        self, dirs: SimpleNamespace, stub_fixers: SimpleNamespace
    ) -> None:
        """Regression test: 1-based starts used to make a 10-frame clip count as 9."""
        _write_split(
            dirs.split_path,
            [_raw_instance("00001", 1, 10), _raw_instance("00002", 1, 9)],
        )

        _run(dirs, do_bboxes=False, length_cuttoff=9)

        assert [i["video_id"] for i in _read_set(dirs, "asl100_cutoff_9")] == ["00001"]


class FakeCapture:
    """Stands in for cv2.VideoCapture: videos whose path contains "missing" don't open."""

    NUM_FRAMES = 100

    def __init__(self, path: str) -> None:
        self.opened = "missing" not in path

    def isOpened(self) -> bool:
        return self.opened

    def get(self, _prop: int) -> float:
        return self.NUM_FRAMES

    def release(self) -> None:
        pass


class TestFixBadFrameRange:
    @pytest.fixture(autouse=True)
    def fake_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.preprocess.cv2.VideoCapture", FakeCapture)

    @pytest.mark.parametrize("policy", ["strict", "reset"])
    def test_valid_range_is_untouched(
        self, tmp_path: Path, policy: RemovePolicy
    ) -> None:
        kept, removed = fix_bad_frame_range(tmp_path, [_entry("00001", 5, 50)], policy)

        assert removed == []
        assert (kept[0].instance.frame_start, kept[0].instance.frame_end) == (5, 50)
        assert kept[0].fixes == []

    def test_reset_resets_and_records(self, tmp_path: Path) -> None:
        kept, removed = fix_bad_frame_range(
            tmp_path, [_entry("00001", 3732, 3852)], "reset"
        )

        assert removed == []
        (entry,) = kept
        assert (entry.instance.frame_start, entry.instance.frame_end) == (0, 100)
        assert [(f.stage, f.action) for f in entry.fixes or []] == [
            ("frame_range", "reset"),
            ("frame_range", "reset"),
        ]

    def test_end_of_minus_one_means_last_frame(self, tmp_path: Path) -> None:
        kept, _ = fix_bad_frame_range(tmp_path, [_entry("00001", 0, -1)], "strict")

        (entry,) = kept
        assert entry.instance.frame_end == FakeCapture.NUM_FRAMES
        assert entry.fixes == []

    def test_end_past_clip_resets_to_clip_length(self, tmp_path: Path) -> None:
        """Regression test: with a valid start, an end past the clip used to be accepted
        up to start + num_frames, and reset to start + num_frames beyond that."""
        kept, _ = fix_bad_frame_range(tmp_path, [_entry("00001", 5, 102)], "reset")

        (entry,) = kept
        assert (entry.instance.frame_start, entry.instance.frame_end) == (5, 100)
        assert [(f.stage, f.action) for f in entry.fixes or []] == [
            ("frame_range", "reset")
        ]

    def test_strict_removes_and_records(self, tmp_path: Path) -> None:
        kept, removed = fix_bad_frame_range(
            tmp_path, [_entry("00001", 3732, 3852)], "strict"
        )

        assert kept == []
        (entry,) = removed
        assert [(f.stage, f.action) for f in entry.fixes or []] == [
            ("frame_range", "removed")
        ]

    def test_unreadable_video_is_removed_even_with_reset(self, tmp_path: Path) -> None:
        kept, removed = fix_bad_frame_range(
            tmp_path / "missing", [_entry("00001")], "reset"
        )

        assert kept == []
        assert [(f.stage, f.action) for f in removed[0].fixes or []] == [
            ("unreadable", "removed")
        ]


class TestFixBadBboxes:
    """YOLO and frame loading are stubbed: the fake model detects no person."""

    FRAMES_SHAPE = (4, 3, 64, 80)  # (T, C, H, W)

    @pytest.fixture(autouse=True)
    def no_person_detected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        no_boxes = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=torch.zeros((0, 4)), cls=torch.zeros(0))
        )
        monkeypatch.setattr(
            "src.preprocess.YOLO", lambda _weights: lambda *_a, **_k: [no_boxes]
        )
        monkeypatch.setattr(
            "src.preprocess.load_rgb_frames_from_video",
            lambda *_a: torch.zeros(self.FRAMES_SHAPE, dtype=torch.uint8),
        )

    def test_reset_uses_whole_frame(self, tmp_path: Path) -> None:
        """Regression test: "reset" used to raise ValueError here."""
        kept, removed = fix_bad_bboxes(
            tmp_path, [_entry("00001")], "reset", num_workers=0
        )

        assert removed == []
        (entry,) = kept
        assert entry.instance.bbox == [0, 0, 80, 64]
        assert entry.bboxes_fixed is True
        assert [(f.stage, f.action) for f in entry.fixes or []] == [("bbox", "reset")]

    def test_strict_removes(self, tmp_path: Path) -> None:
        kept, removed = fix_bad_bboxes(
            tmp_path, [_entry("00001")], "strict", num_workers=0
        )

        assert kept == []
        assert [(f.stage, f.action) for f in removed[0].fixes or []] == [
            ("bbox", "removed")
        ]


class TestRemoveShortSamples:
    def test_removes_at_or_below_cutoff(self) -> None:
        kept, removed = remove_short_samples(
            [_entry("00001", 1, 10), _entry("00002", 0, 10)], cutoff=9
        )

        assert [e.instance.video_id for e in kept] == ["00002"]
        assert [e.instance.video_id for e in removed] == ["00001"]
        assert [(f.stage, f.action) for f in removed[0].fixes or []] == [
            ("length", "removed")
        ]


class TestLoadInstanceCache:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_instance_cache(tmp_path / "missing.json") == {}

    def test_old_format_is_ignored_not_misread(self, tmp_path: Path) -> None:
        """The oldest caches were a flat list of Instance dicts (no bboxes_fixed marker).
        Loading one must not silently misinterpret it as new-format entries."""
        cache_path = tmp_path / "instance_cache.json"
        old_format = [_instance("00001").model_dump()]
        cache_path.write_text(json.dumps(old_format))

        assert load_instance_cache(cache_path) == {}

    def test_other_version_is_ignored(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "instance_cache.json"
        entry = CacheEntry(instance=_instance("00001"), bboxes_fixed=True)
        cache_path.write_text(
            json.dumps({"version": CACHE_VERSION - 1, "entries": [entry.model_dump()]})
        )

        assert load_instance_cache(cache_path) == {}

    def test_version_1_list_format_is_ignored(self, tmp_path: Path) -> None:
        """Version-1 caches were a bare list of entries, with 1-based frames."""
        cache_path = tmp_path / "instance_cache.json"
        entry = CacheEntry(instance=_instance("00001"), bboxes_fixed=True)
        cache_path.write_text(json.dumps([entry.model_dump()]))

        assert load_instance_cache(cache_path) == {}

    def test_round_trips_through_save(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "instance_cache.json"
        entry = CacheEntry(instance=_instance("00001"), bboxes_fixed=True)
        entry.record("frame_range", "reset", "x")
        save_instance_cache(cache_path, {"00001": entry})

        loaded = load_instance_cache(cache_path)
        assert loaded["00001"] == entry
