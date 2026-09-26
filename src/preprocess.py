import json
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeGuard

# NOTE: Running this script will mess up the environment you are using, becuase of this stupid YOLO thing
# it will give a '3D conv not implemented yada yada' error message
# The solution is to delete and recreate the environment
import cv2
import torch
import tqdm
from pydantic import BaseModel, TypeAdapter, ValidationError
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO  # type: ignore

from src.configs import LABELS_PATH
from src.run_types import AVAIL_SETS, RAW_DIR, SPLIT_DIR, WLASL_ROOT

# local imports
from src.utils import load_rgb_frames_from_video

"""Naming convention:
- set: one of train, test and val
- split: one of asl100, asl300, asl1000, asl2000"""


class RawInstance(BaseModel):
    """Represents a single raw instance of a gloss in the dataset."""

    bbox: list[int]  # [x_min, y_min, x_max, y_max]
    frame_end: int
    frame_start: int
    instance_id: int
    signer_id: int
    source: str
    split: str
    url: str
    variation_id: int
    video_id: str


class Instance(RawInstance):
    """Represents a single instance of a gloss in the dataset, with the label_num and label_name added.

    Unlike `RawInstance`, frames are 0-based and end-exclusive: the clip is
    `frames[frame_start:frame_end]`. Label files generated before cache version 2 kept
    WLASL's 1-based `frame_start` instead; see `src/info/WLASL_info.md`.
    """

    label_num: int
    label_name: str


class WLASLClass(BaseModel):
    """Represents a single gloss and its associated raw instances."""

    gloss: str
    instances: list[RawInstance]


RemovePolicy: TypeAlias = Literal["strict", "reset"]
"""How a fixer handles a bad instance: "strict" removes it, "reset" falls back to the
whole video (frame range) or whole frame (bbox). Both are logged."""

FixStage: TypeAlias = Literal["unreadable", "frame_range", "bbox", "length"]
FixAction: TypeAlias = Literal["removed", "reset"]


class FixRecord(BaseModel):
    """One change preprocessing made to an instance."""

    stage: FixStage
    action: FixAction
    reason: str


class CacheEntry(BaseModel):
    """An instance plus its preprocessing history. This is both the unit the fixers work
    on and the instance cache's on-disk format.

    `bboxes_fixed` is needed because `do_bboxes` can differ between runs that otherwise
    share a cache (e.g. a `--no_bbox` run followed by a normal one): without this marker,
    an instance cached with its raw/unfixed placeholder bbox would be indistinguishable
    from one that was actually bbox-fixed, and would be reused as-is forever.

    """

    instance: Instance
    bboxes_fixed: bool
    fixes: list[FixRecord] = []

    @property
    def removed(self) -> bool:
        return any(fix.action == "removed" for fix in self.fixes)

    def record(self, stage: FixStage, action: FixAction, reason: str) -> None:
        self.fixes.append(FixRecord(stage=stage, action=action, reason=reason))


CACHE_VERSION = 2
"""Bump when cached instances stop being valid for the current code. Version 2 switched
frames to 0-based, so version-1 entries have shifted frame ranges and bboxes."""


class InstanceCache(BaseModel):
    """On-disk format of the instance cache."""

    version: int
    entries: list[CacheEntry]


class LoggedInstance(Instance):
    """An instance that preprocessing changed or removed, as written to the log."""

    fixes: list[FixRecord]


class SetLog(BaseModel):
    """What preprocessing did to one set. `num_raw == num_kept + num_removed` always holds."""

    num_raw: int
    num_kept: int
    num_removed: int
    counts: dict[str, int]  # "<stage>:<action>" -> number of instances
    instances: list[LoggedInstance]


class PreprocessLog(BaseModel):
    """Log of one split's preprocessing run, written to `<split output dir>/preprocess_log.json`."""

    split: str
    length_cutoff: int
    strictness: tuple[RemovePolicy, RemovePolicy]
    do_bboxes: bool
    sets: dict[str, SetLog]


def is_processed_instance(obj: Any) -> TypeGuard[Instance]:
    """Type guard to check if an object is a valid Instance dict/object."""
    try:
        Instance.model_validate(obj)
        return True
    except ValidationError:
        return False


def instance_to_processed(d: RawInstance, label_num: int, label_name: str) -> Instance:
    """Convert a RawInstance to an Instance: add labels, and convert WLASL's 1-based
    `frame_start` to a 0-based index.

    WLASL's inclusive, 1-based `frame_end` already equals the 0-based exclusive end, so it
    is kept as-is, including its -1 ("ends at the last frame") sentinel, which
    `fix_bad_frame_range` resolves against the video.
    """
    return Instance(
        **d.model_dump() | {"frame_start": d.frame_start - 1},
        label_num=label_num,
        label_name=label_name,
    )


def get_set(
    lst_wlasl_class_dicts: list[WLASLClass], set_name: AVAIL_SETS
) -> list[Instance]:
    """Filters list of WLASLClass based on whether the instances are from the provided set_name."""
    mod_instances = []
    for i, gloss_d in enumerate(lst_wlasl_class_dicts):
        for inst in gloss_d.instances:
            if inst.split == set_name:
                mod_instances.append(instance_to_processed(inst, i, gloss_d.gloss))
    return mod_instances


def _partition(entries: list[CacheEntry]) -> tuple[list[CacheEntry], list[CacheEntry]]:
    """Split entries into (kept, removed)."""
    kept = [entry for entry in entries if not entry.removed]
    removed = [entry for entry in entries if entry.removed]
    return kept, removed


def fix_bad_frame_range(
    raw_path: Path,
    entries: list[CacheEntry],
    remove_policy: RemovePolicy = "strict",
) -> tuple[list[CacheEntry], list[CacheEntry]]:
    """Check each entry's frame range against its video, recording every change on the entry.

    Frames are 0-based and end-exclusive (see `instance_to_processed`). A `frame_end` of -1
    means the clip's last frame, and is set to the frame count without being logged.

    A video that cannot be opened is always removed, whatever the policy, since there is
    nothing to reset to. An impossible start or end frame is removed ("strict") or reset
    to the start/end of the video ("reset").

    Returns:
        (kept, removed) entries.
    """
    for entry in tqdm.tqdm(entries, desc="fixing frame ranges"):
        instance = entry.instance
        vid_path = raw_path / f"{instance.video_id}.mp4"

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            entry.record("unreadable", "removed", f"Could not open video {vid_path}.")
            continue
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        start = instance.frame_start
        end = num_frames if instance.frame_end == -1 else instance.frame_end

        if start < 0 or start >= num_frames:
            message = f"Invalid start frame {start} for video length {num_frames}."
            if remove_policy == "strict":
                entry.record("frame_range", "removed", message)
                continue
            entry.record("frame_range", "reset", message + " Setting to 0.")
            start = 0

        if end <= start or end > num_frames:
            message = f"Invalid end frame {end} for video length {num_frames} and start frame {start}."
            if remove_policy == "strict":
                entry.record("frame_range", "removed", message)
                continue
            entry.record("frame_range", "reset", message + " Setting to num_frames.")
            end = num_frames

        instance.frame_start = start
        instance.frame_end = end

    return _partition(entries)


def get_largest_bbox(bboxes: list[list[float]]) -> list[float] | None:
    """Given a list of bounding boxes, returns the largest bounding box that encompasses all of them, if one exists."""
    if not bboxes:
        return None
    x_min, y_min, x_max, y_max = bboxes[0]
    for box in bboxes:
        x1, y1, x2, y2 = box
        x_min = min(x_min, x1)
        y_min = min(y_min, y1)
        x_max = max(x_max, x2)
        y_max = max(y_max, y2)
    return [x_min, y_min, x_max, y_max]


class VideoFrameDataset(Dataset):
    """Loads each instance's frames, yielding `(frames, index into instances)`."""

    def __init__(self, raw_path: Path, instances: list[Instance]):
        self.raw_path = raw_path
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        instance = self.instances[idx]
        vid_path = self.raw_path / f"{instance.video_id}.mp4"
        frames = load_rgb_frames_from_video(
            str(vid_path), instance.frame_start, instance.frame_end
        )
        # stays uint8 here — 4x smaller in the prefetch queue than float32
        return frames, idx


def fix_bad_bboxes(
    raw_path: Path,
    entries: list[CacheEntry],
    remove_policy: RemovePolicy = "strict",
    num_workers: int = 8,
) -> tuple[list[CacheEntry], list[CacheEntry]]:
    """Replace each entry's bbox with the smallest box enclosing every person YOLOv8 detects
    across its frames, recording every change on the entry.

    If no person is detected, the entry is removed ("strict") or given a whole-frame bbox
    ("reset"). Kept entries are marked `bboxes_fixed`.

    Returns:
        (kept, removed) entries.
    """
    model = YOLO("yolov8n.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = VideoFrameDataset(raw_path, [entry.instance for entry in entries])
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,  # tune to your CPU core count
        collate_fn=lambda batch: batch[0],  # unwrap the single (frames, idx) pair
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    for frames, idx in tqdm.tqdm(
        loader, total=len(entries), desc="Fixing bounding boxes"
    ):
        entry = entries[idx]
        # normalize after transfer, not before
        frames = frames.to(device).float() / 255.0
        results = model(frames, device=device, verbose=False)
        bboxes = []
        for result in results:
            person_bboxes = result.boxes.xyxy[result.boxes.cls == 0]
            if len(person_bboxes) > 0:
                bboxes.extend(person_bboxes.tolist())

        largest_bbox = get_largest_bbox(bboxes)
        if largest_bbox is None:
            message = "No person detected by YOLO."
            if remove_policy == "strict":
                entry.record("bbox", "removed", message)
                continue
            entry.record("bbox", "reset", message + " Using whole frame.")
            _, _, height, width = frames.shape
            largest_bbox = [0, 0, width, height]

        entry.instance.bbox = [round(coord) for coord in largest_bbox]
        entry.bboxes_fixed = True

    return _partition(entries)


def remove_short_samples(
    entries: list[CacheEntry], cutoff: int
) -> tuple[list[CacheEntry], list[CacheEntry]]:
    """Remove entries with `cutoff` or fewer frames, recording the removal on the entry.

    Returns:
        (kept, removed) entries.
    """
    for entry in entries:
        num_frames = entry.instance.frame_end - entry.instance.frame_start
        if num_frames <= cutoff:
            entry.record(
                "length",
                "removed",
                f"{num_frames} frames, at or below the cutoff of {cutoff}.",
            )
    return _partition(entries)


def print_v(s: str, y: bool) -> None:
    if y:
        print(s)


def check_paths(
    split_path: Path, raw_path: Path, output_path: Path, verbose: bool
) -> bool:
    """Checks if the provided paths exist and are of the correct type."""
    if split_path.exists() and split_path.is_file():
        print_v(f"split path: {split_path}, found", verbose)
    else:
        print(f"split path: {split_path}, not found")
        return False
    if raw_path.exists() and raw_path.is_dir():
        print_v(f"raw path: {raw_path}, found", verbose)
    else:
        print(f"raw path: {raw_path}, not found")
        return False
    if output_path.exists() and output_path.is_dir():
        print_v(f"output path: {output_path}, found", verbose)
    else:
        print(f"output path: {output_path}, not found")
        return False
    return True


def load_instance_cache(cache_path: Path) -> dict[str, CacheEntry]:
    """Load the instance cache, keyed by video_id.

    A missing cache, or one that is unreadable or from a different `CACHE_VERSION`, loads as
    empty, so every instance is reprocessed.
    """
    if not cache_path.exists():
        return {}
    try:
        cache = InstanceCache.model_validate_json(cache_path.read_text())
    except ValidationError:
        cache = None
    if cache is None or cache.version != CACHE_VERSION:
        print(
            f"Cache at {cache_path} is from an older version or unreadable; ignoring it "
            "and reprocessing from scratch."
        )
        return {}
    return {entry.instance.video_id: entry for entry in cache.entries}


def save_instance_cache(cache_path: Path, cache: dict[str, CacheEntry]) -> None:
    cache_path.write_text(
        InstanceCache(
            version=CACHE_VERSION, entries=list(cache.values())
        ).model_dump_json(indent=2)
    )


def _fix_uncached(
    raw_path: Path,
    entries: list[CacheEntry],
    do_bboxes: bool,
    strictness: tuple[RemovePolicy, RemovePolicy],
    verbose: bool,
) -> tuple[list[CacheEntry], list[CacheEntry]]:
    """Run the video-dependent fixes (frame range, then bboxes) on fresh entries.

    Returns:
        (kept, removed) entries.
    """
    if not entries:
        return [], []
    print_v("Fixing frame ranges", verbose)
    kept, removed = fix_bad_frame_range(raw_path, entries, remove_policy=strictness[0])
    if do_bboxes:
        print_v("Fixing bounding boxes", verbose)
        kept, bbox_removed = fix_bad_bboxes(raw_path, kept, remove_policy=strictness[1])
        removed += bbox_removed
    return kept, removed


def _set_log(num_raw: int, kept: list[CacheEntry], removed: list[CacheEntry]) -> SetLog:
    """Summarise one set's preprocessing, checking that every raw instance is accounted for."""
    if num_raw != len(kept) + len(removed):
        raise RuntimeError(
            f"{num_raw} raw instances but {len(kept)} kept + {len(removed)} removed"
        )
    changed = [entry for entry in kept + removed if entry.fixes]
    counts = Counter(
        f"{fix.stage}:{fix.action}" for entry in changed for fix in entry.fixes
    )
    return SetLog(
        num_raw=num_raw,
        num_kept=len(kept),
        num_removed=len(removed),
        counts=dict(sorted(counts.items())),
        instances=[
            LoggedInstance(**entry.instance.model_dump(), fixes=entry.fixes)
            for entry in changed
        ],
    )


def preprocess_split(
    split_path: Path,
    raw_path: Path,
    output_base: Path,
    verbose: bool = False,
    file_extension: str = "fixed_frange_bboxes.json",
    strictness: tuple[RemovePolicy, RemovePolicy] = ("strict", "strict"),
    do_bboxes: bool = True,
    length_cuttoff: int = 9,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> None:
    """Preprocesses a split of the WLASL dataset, reusing fixes for
    instances already processed in a previous split (e.g. asl100 -> asl300).

    Writes each set's kept instances, plus one `preprocess_log.json` for the whole split
    recording every instance that was changed or removed (see `PreprocessLog`). The log
    covers cached instances too, so it is complete whatever order the splits are run in.

    The cache holds instances after the frame-range and bbox fixes, but before the length
    cutoff, which is re-applied to every instance on every run. Removed instances are not
    cached.

    use_cache: if True, trusts that the fix parameters (strictness etc.) are unchanged
    since the cache was built and reuses cached instances as-is. If False, reprocesses
    every instance from scratch (still writing results back to the cache for later runs).

    A cached instance is only reused as-is if it matches this run's `do_bboxes`
    requirement: an instance cached from a `do_bboxes=False` run (raw/unfixed bbox)
    is bbox-fixed (not reprocessed from scratch) before being reused by a
    `do_bboxes=True` run.
    """

    if not check_paths(split_path, raw_path, output_base, verbose):
        return

    with open(split_path, "r") as f:
        raw_json_data = json.load(f)

    if not raw_json_data:
        print(f"no data found in {split_path}")
        return

    wlasl_adapter = TypeAdapter(list[WLASLClass])
    asl_num = wlasl_adapter.validate_python(raw_json_data)

    base_name = split_path.name.replace(".json", "")
    base_name = (
        f"{base_name}_cutoff_{length_cuttoff}" if length_cuttoff > 0 else base_name
    )
    output_dir = output_base / base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_path or (output_base / f"instance_cache_v{CACHE_VERSION}.json")
    cache = load_instance_cache(cache_path)
    set_logs: dict[str, SetLog] = {}

    print_v(f"Processing {base_name}", verbose)
    subsets: list[AVAIL_SETS] = ["train", "test", "val"]
    for subset in subsets:
        print_v(f"For split: {subset}", verbose)
        instances = get_set(asl_num, subset)

        reused: list[CacheEntry] = []
        needs_bboxes: list[CacheEntry] = []
        fresh: list[CacheEntry] = []
        for inst in instances:
            cached = cache.get(inst.video_id) if use_cache else None
            if cached is None:
                fresh.append(CacheEntry(instance=inst.model_copy(), bboxes_fixed=False))
                continue
            entry = cached.model_copy(deep=True)
            if do_bboxes and not entry.bboxes_fixed:
                needs_bboxes.append(entry)
            else:
                reused.append(entry)
        print_v(
            f"Reusing {len(reused)} cached / re-fixing bboxes for "
            f"{len(needs_bboxes)} cached / fixing {len(fresh)} new",
            verbose,
        )

        bbox_kept, bbox_removed = (
            fix_bad_bboxes(raw_path, needs_bboxes, remove_policy=strictness[1])
            if needs_bboxes
            else ([], [])
        )
        fresh_kept, fresh_removed = _fix_uncached(
            raw_path, fresh, do_bboxes, strictness, verbose
        )

        fixed = reused + bbox_kept + fresh_kept
        removed = bbox_removed + fresh_removed
        for entry in fixed:
            cache[entry.instance.video_id] = entry.model_copy(deep=True)

        if length_cuttoff > 0:
            print_v("Removing small samples", verbose)
            kept, short = remove_short_samples(fixed, length_cuttoff)
            removed += short
        else:
            kept = fixed

        set_log = _set_log(len(instances), kept, removed)
        set_logs[subset] = set_log
        print(
            f"{base_name}/{subset}: {set_log.num_raw} raw -> {set_log.num_kept} kept, "
            f"{set_log.num_removed} removed {set_log.counts}"
        )

        inst_path = output_dir / f"{subset}_{file_extension}"
        with open(inst_path, "w") as f:
            json.dump([entry.instance.model_dump() for entry in kept], f, indent=2)

    log = PreprocessLog(
        split=base_name,
        length_cutoff=length_cuttoff,
        strictness=strictness,
        do_bboxes=do_bboxes,
        sets=set_logs,
    )
    (output_dir / "preprocess_log.json").write_text(log.model_dump_json(indent=2))

    save_instance_cache(cache_path, cache)
    print("\n------------------------- finished preprocessing ---------------\n")


if __name__ == "__main__":
    avail_splits = ["asl100", "asl300", "asl1000", "asl2000"]

    parser = ArgumentParser(description="preprocess.py")
    parser.add_argument(
        "asl_split",
        type=str,
        choices=avail_splits + ["all"],
        help="Which WLASL split to preprocess",
    )
    parser.add_argument(
        "-rt",
        "--root",
        type=str,
        help=f"WLASL root if not {WLASL_ROOT}",
        default=WLASL_ROOT,
    )
    parser.add_argument(
        "-sd",
        "--split_dir",
        type=str,
        help=f"Split directory if not {SPLIT_DIR}",
        default=SPLIT_DIR,
    )
    parser.add_argument(
        "-rd",
        "--raw_dir",
        type=str,
        help=f"Video directory if not {RAW_DIR}",
        default=RAW_DIR,
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        type=str,
        help=f"Output directory if not {LABELS_PATH}",
        default=LABELS_PATH,
    )
    parser.add_argument("-ve", "--verbose", action="store_true", help="verbose output")
    parser.add_argument(
        "-ss",
        "--strictness",
        nargs=2,
        choices=["strict", "reset"],
        default=["reset", "reset"],
        help="The strictness levels for frame range, and bounding boxes respectively. Reset takes the full video/frame. Strict disgards. Both log.",
    )
    parser.add_argument(
        "-nb", "--no_bbox", action="store_true", help="Skip intense bbox step"
    )
    parser.add_argument(
        "-nc",
        "--no_cache",
        action="store_true",
        help="Reprocess every instance from scratch instead of reusing the instance cache "
        "(the cache is still updated). Slow: reruns YOLO on every video.",
    )
    parser.add_argument(
        "-lc",
        "--length_cutoff",
        type=int,
        default=0,
        help="Remove samples with this many frames or fewer; 0 keeps all. (default: %(default)s)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.asl_split == "all":
        todo_splits = avail_splits
    else:
        todo_splits = [args.asl_split]

    for split in todo_splits:
        split_path = Path(args.split_dir) / f"{split}.json"
        preprocess_split(
            split_path=split_path,
            raw_path=raw_dir,
            output_base=output_dir,
            verbose=args.verbose,
            strictness=tuple(args.strictness),
            do_bboxes=(not args.no_bbox),
            length_cuttoff=args.length_cutoff,
            use_cache=not args.no_cache,
        )
