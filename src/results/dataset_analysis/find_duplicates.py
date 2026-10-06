"""Find duplicated videos in WLASL-2000.

Every instance in the asl2000 split file is fingerprinted from its decoded frames, and three
kinds of duplicate group are reported:

- ``exact``: the SHA-1 of all decoded frames is identical (the same file, or the same frames).
- ``near``: the same decoded frame count, and small greyscale thumbnails of evenly spaced frames
  differ by less than ``--threshold`` (mean absolute difference, 0-255). Catches re-encoded
  copies, which decode to slightly different pixels. Copies trimmed to a different length are
  not caught.
- ``same_url_and_range``: the same source URL and annotated frame range, from the annotations
  alone (no decoding).

Each group lists its members' gloss, set, signer and variation, and is flagged when it spans
several glosses or several sets (a train/test leak). Background, including the first known pairs
(05741/41452 and 05743/41454, before/past): ``src/info/WLASL_info.md``.

Fingerprints are cached, so a rerun only decodes videos it hasn't seen. Run from the repo root:

    python src/results/dataset_analysis/find_duplicates.py

It is run as a script rather than with ``-m`` so that ``src/results/__init__.py`` (which imports
the Que) isn't loaded. It doesn't import ``src.preprocess`` either, because that loads YOLO.
"""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, TypeAdapter
from tqdm import tqdm

from src.run_types import RAW_DIR, RESULTS_DIR, SPLIT_DIR, WLASL_ROOT
from src.utils import cv_load

THUMB_SIZE = 16
NUM_THUMBS = 4
DEFAULT_THRESHOLD = 2.0
CACHE_VERSION = 1
DEFAULT_CACHE = WLASL_ROOT / f"preprocessed/duplicate_fingerprints_v{CACHE_VERSION}.npz"
DEFAULT_REPORT = RESULTS_DIR / "dataset_analysis/duplicates_asl2000.json"
KNOWN_DUPLICATES: list[set[str]] = [{"05741", "41452"}, {"05743", "41454"}]

DuplicateKind: TypeAlias = Literal["exact", "near", "same_url_and_range"]
Thumbs: TypeAlias = npt.NDArray[np.uint8]


class SplitInstance(BaseModel):
    """The fields of a raw WLASL instance this script uses (see `preprocess.RawInstance`)."""

    video_id: str
    split: str
    signer_id: int
    variation_id: int
    source: str
    url: str
    frame_start: int
    frame_end: int


class SplitGloss(BaseModel):
    gloss: str
    instances: list[SplitInstance]


@dataclass(frozen=True)
class Fingerprint:
    """A decoded video's frame count, the SHA-1 of its frames and `NUM_THUMBS` greyscale thumbnails."""

    video_id: str
    num_frames: int
    sha1: str
    thumbs: Thumbs


class Member(BaseModel):
    video_id: str
    gloss: str
    set: str
    signer_id: int
    variation_id: int
    source: str


class DuplicateGroup(BaseModel):
    kind: DuplicateKind
    members: list[Member]
    cross_gloss: bool
    cross_set: bool


class Report(BaseModel):
    split_file: str
    num_instances: int
    near_threshold: float
    failed: dict[str, str]
    groups: list[DuplicateGroup]


def load_split(split_file: Path) -> dict[str, Member]:
    """Map each video id in a WLASL split file to its gloss and instance details."""
    glosses = TypeAdapter(list[SplitGloss]).validate_json(split_file.read_bytes())
    members: dict[str, Member] = {}
    for gloss in glosses:
        for inst in gloss.instances:
            members[inst.video_id] = Member(
                video_id=inst.video_id,
                gloss=gloss.gloss,
                set=inst.split,
                signer_id=inst.signer_id,
                variation_id=inst.variation_id,
                source=inst.source,
            )
    return members


def fingerprint_video(video_path: Path) -> Fingerprint:
    """Decode every frame of a video and fingerprint it."""
    frames = cv_load(video_path, 0, 0, all=True)
    sha = hashlib.sha1()
    for frame in frames:
        sha.update(frame.tobytes())
    idx = np.linspace(0, len(frames) - 1, NUM_THUMBS).round().astype(int)
    thumbs = np.stack(
        [
            cv2.resize(
                cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY),
                (THUMB_SIZE, THUMB_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            for i in idx
        ]
    ).astype(np.uint8)
    return Fingerprint(
        video_id=video_path.stem,
        num_frames=len(frames),
        sha1=sha.hexdigest(),
        thumbs=thumbs,
    )


def _fingerprint_or_error(video_path: Path) -> Fingerprint | str:
    try:
        return fingerprint_video(video_path)
    except (FileNotFoundError, ValueError) as e:
        return str(e)


def load_cache(cache_path: Path) -> dict[str, Fingerprint]:
    if not cache_path.exists():
        return {}
    with np.load(cache_path) as data:
        return {
            str(vid): Fingerprint(
                video_id=str(vid), num_frames=int(n), sha1=str(sha), thumbs=thumbs
            )
            for vid, n, sha, thumbs in zip(
                data["video_ids"],
                data["num_frames"],
                data["sha1"],
                data["thumbs"],
                strict=True,
            )
        }


def save_cache(cache_path: Path, fingerprints: Iterable[Fingerprint]) -> None:
    fps = list(fingerprints)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        video_ids=np.array([f.video_id for f in fps]),
        num_frames=np.array([f.num_frames for f in fps]),
        sha1=np.array([f.sha1 for f in fps]),
        thumbs=np.stack([f.thumbs for f in fps]),
    )


def fingerprint_all(
    video_ids: Sequence[str], raw_dir: Path, cache_path: Path, workers: int
) -> tuple[dict[str, Fingerprint], dict[str, str]]:
    """Fingerprint every video, reusing and updating the cache. Returns (fingerprints, failures)."""
    cached = load_cache(cache_path)
    todo = [v for v in video_ids if v not in cached]
    failed: dict[str, str] = {}
    if todo:
        paths = [raw_dir / f"{v}.mp4" for v in todo]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_fingerprint_or_error, paths, chunksize=16)
            for vid, res in tqdm(
                zip(todo, results, strict=True), total=len(todo), desc="Decoding"
            ):
                if isinstance(res, str):
                    failed[vid] = res
                else:
                    cached[vid] = res
        save_cache(cache_path, cached.values())
    return {v: cached[v] for v in video_ids if v in cached}, failed


def _union_find_groups(
    ids: Sequence[str], pairs: Iterable[tuple[str, str]]
) -> list[list[str]]:
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def exact_groups(fingerprints: dict[str, Fingerprint]) -> list[list[str]]:
    by_sha: dict[str, list[str]] = defaultdict(list)
    for fp in fingerprints.values():
        by_sha[fp.sha1].append(fp.video_id)
    return [sorted(g) for g in by_sha.values() if len(g) > 1]


def near_pairs(
    fingerprints: dict[str, Fingerprint], threshold: float
) -> list[tuple[str, str]]:
    """Pairs of videos with the same frame count whose thumbnails differ by less than threshold."""
    by_len: dict[int, list[Fingerprint]] = defaultdict(list)
    for fp in fingerprints.values():
        by_len[fp.num_frames].append(fp)
    pairs: list[tuple[str, str]] = []
    for bucket in by_len.values():
        if len(bucket) < 2:
            continue
        x = np.stack([fp.thumbs.reshape(-1) for fp in bucket]).astype(np.float32)
        for i in range(len(bucket) - 1):
            diffs = np.abs(x[i + 1 :] - x[i]).mean(axis=1)
            pairs += [
                (bucket[i].video_id, bucket[i + 1 + j].video_id)
                for j in np.flatnonzero(diffs < threshold)
            ]
    return pairs


def url_groups(split_file: Path) -> list[list[str]]:
    glosses = TypeAdapter(list[SplitGloss]).validate_json(split_file.read_bytes())
    by_key: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for gloss in glosses:
        for inst in gloss.instances:
            by_key[(inst.url, inst.frame_start, inst.frame_end)].append(inst.video_id)
    return [sorted(g) for g in by_key.values() if len(g) > 1]


def make_group(
    kind: DuplicateKind, ids: Sequence[str], members: dict[str, Member]
) -> DuplicateGroup:
    ms = [members[i] for i in ids]
    return DuplicateGroup(
        kind=kind,
        members=ms,
        cross_gloss=len({m.gloss for m in ms}) > 1,
        cross_set=len({m.set for m in ms}) > 1,
    )


def find_duplicates(
    split_file: Path, raw_dir: Path, cache_path: Path, threshold: float, workers: int
) -> Report:
    members = load_split(split_file)
    ids = sorted(members)
    fingerprints, failed = fingerprint_all(ids, raw_dir, cache_path, workers)

    exact = exact_groups(fingerprints)
    in_exact = {frozenset(g) for g in exact}
    near = [
        g
        for g in _union_find_groups(
            list(fingerprints), near_pairs(fingerprints, threshold)
        )
        if frozenset(g) not in in_exact
    ]
    groups = (
        [make_group("exact", g, members) for g in exact]
        + [make_group("near", g, members) for g in near]
        + [make_group("same_url_and_range", g, members) for g in url_groups(split_file)]
    )
    return Report(
        split_file=str(split_file),
        num_instances=len(members),
        near_threshold=threshold,
        failed=failed,
        groups=groups,
    )


def print_summary(report: Report) -> None:
    print(f"{report.num_instances} instances, {len(report.failed)} failed to decode")
    for kind in ("exact", "near", "same_url_and_range"):
        gs = [g for g in report.groups if g.kind == kind]
        n_videos = sum(len(g.members) for g in gs)
        n_gloss = sum(g.cross_gloss for g in gs)
        n_set = sum(g.cross_set for g in gs)
        print(
            f"{kind}: {len(gs)} groups ({n_videos} videos), {n_gloss} cross-gloss, {n_set} cross-set"
        )
    found = {
        frozenset(m.video_id for m in g.members)
        for g in report.groups
        if g.kind == "exact"
    }
    for known in KNOWN_DUPLICATES:
        status = "found" if any(known <= f for f in found) else "NOT FOUND"
        print(f"sanity check {sorted(known)}: {status}")
    for g in report.groups:
        if g.cross_gloss or g.cross_set:
            desc = ", ".join(
                f"{m.video_id} {m.gloss}/{m.set}/s{m.signer_id}" for m in g.members
            )
            print(f"  [{g.kind}] {desc}")


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--split-file", type=Path, default=SPLIT_DIR / "asl2000.json")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    report = find_duplicates(
        args.split_file, args.raw_dir, args.cache, args.threshold, args.workers
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.model_dump(), indent=2))
    print_summary(report)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
