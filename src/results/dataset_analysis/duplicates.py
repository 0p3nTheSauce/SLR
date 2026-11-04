"""Find WLASL clips that are copies of each other, and how they fall across glosses and sets: for
`duplicate_clips.ipynb` (the dataset), `error_analysis/duplicate_accuracy.ipynb` and
`error_analysis/signer_shortcut.py` (their effect on a model).

The supplied clips are cut per instance (see `src/info/WLASL_info.md`), so a duplicate is one
video file filed under two `video_id`s. WLASL files many signs under several glosses, and
sometimes under the same gloss in two sets, so a copy is a label conflict (different glosses) or
a train/test leak (same gloss, different sets).

Exact copies are found by MD5 (`MD5_STASH`), which a re-encode of the same video escapes. The
near-duplicate search at the end (per-frame perceptual hashes, `find_near_duplicates`) is
unfinished: it recovers every exact copy but can't yet tell a trimmed copy from another sign by
the same signer in the same session. Findings: `near_duplicates.md`.

Run from the repo root, `python -m src.results.dataset_analysis.duplicates` runs
`find_near_duplicates` on the stashed hashes.
"""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.fft import dctn
from tqdm import tqdm

from src.results import load_json, stash_json
from src.run_types import RAW_DIR, SPLIT_DIR
from src.utils import cv_load

TWIN_COLUMNS = ["has_twin", "twin_other_gloss", "twin_other_set", "twin_same_gloss_other_set"]


def clip_md5(video_path: Path) -> str:
    """MD5 of a whole clip (clips are small, so it is read in one go)."""
    return hashlib.md5(video_path.read_bytes()).hexdigest()


def _clip_paths(raw_dir: Path, video_ids: list[str]) -> list[Path]:
    """Each video's file in `raw_dir` (`<video_id>.<ext>`)."""
    return [next(raw_dir.glob(f"{video_id}.*")) for video_id in video_ids]


def hash_clips(raw_dir: Path, video_ids: Iterable[str], workers: int = 8) -> dict[str, str]:
    """MD5 of each video file in `raw_dir` (`<video_id>.<ext>`), keyed by `video_id`."""
    ids = list(video_ids)
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(ids, pool.map(clip_md5, _clip_paths(raw_dir, ids))))


STASH_DIR = Path(__file__).parent / "stashed_results"
MD5_STASH = STASH_DIR / "clip_md5.json"
"""`hash_clips` of every WLASL-2000 clip, keyed by `video_id`."""


def load_instances(split: str) -> pd.DataFrame:
    """Every instance of a split's annotation file (`SPLIT_DIR/<split>.json`, e.g. `asl2000`),
    one row each, with its clip's MD5 from `MD5_STASH`.

    Columns: `video_id`, `gloss`, `set` (train/val/test), `source`, `signer_id`, `variation_id`,
    `md5`.
    """
    glosses = json.loads((SPLIT_DIR / f"{split}.json").read_text())
    md5s: dict[str, str] = json.loads(MD5_STASH.read_text())
    return pd.DataFrame(
        [
            {
                "video_id": i["video_id"],
                "gloss": g["gloss"],
                "set": i["split"],
                "source": i["source"],
                "signer_id": i["signer_id"],
                "variation_id": i["variation_id"],
                "md5": md5s[i["video_id"]],
            }
            for g in glosses
            for i in g["instances"]
        ]
    )


def load_flagged(split: str) -> pd.DataFrame:
    """`load_instances` with the `flag_twins` and `flag_twins_by_set` columns, twins looked for
    within the split."""
    return flag_twins_by_set(flag_twins(load_instances(split)))


def _per_group(
    frame: pd.DataFrame, by: list[str], column: str, how: Literal["size", "nunique"]
) -> pd.Series:
    """`how` of `column` within each `by` group, broadcast back to `frame`'s rows."""
    return cast(pd.Series, frame.groupby(by)[column].transform(how))


def flag_twins(instances: pd.DataFrame) -> pd.DataFrame:
    """Add, per instance, whether another instance has an identical clip (`md5`).

    `instances` needs `md5`, `gloss` and `set` columns. Added boolean columns:

    - `has_twin`: some other instance has the same clip.
    - `twin_other_gloss`: one of its twins has a different gloss (a label conflict).
    - `twin_other_set`: one of its twins is in a different set.
    - `twin_same_gloss_other_set`: one of its twins has the same gloss in a different set (a
      leak of a correctly-labelled clip).

    Twins are only looked for within `instances`, so pass a split's own instances to ask the
    question for that split.
    """
    out = instances.copy()
    out["has_twin"] = _per_group(instances, ["md5"], "gloss", "size") > 1
    out["twin_other_gloss"] = _per_group(instances, ["md5"], "gloss", "nunique") > 1
    out["twin_other_set"] = _per_group(instances, ["md5"], "set", "nunique") > 1
    out["twin_same_gloss_other_set"] = _per_group(instances, ["md5", "gloss"], "set", "nunique") > 1
    return out


SETS = ("train", "val", "test")
TwinRelation = Literal["same_gloss", "other_gloss"]
TWIN_KINDS = {
    "repeat": "same gloss, same set",
    "leak": "same gloss, other set",
    "conflict_same_set": "other gloss, same set",
    "conflict_other_set": "other gloss, other set",
}


def twin_set_column(relation: TwinRelation, set_name: str) -> str:
    """Name of `flag_twins_by_set`'s column for a twin with `relation` in `set_name`."""
    return f"{relation}_in_{set_name}"


def twin_pairs(instances: pd.DataFrame) -> pd.DataFrame:
    """Every ordered pair of distinct instances with the same clip, one row each.

    `instances` needs `video_id` (unique), `md5`, `gloss` and `set` columns. Each pair appears in
    both orders: columns `video_id`, `gloss`, `set`, the same with a `_twin` suffix, `md5`, and
    `same_gloss`.
    """
    cols = ["video_id", "md5", "gloss", "set"]
    pairs = instances[cols].merge(instances[cols], on="md5", suffixes=("", "_twin"))
    pairs = cast(pd.DataFrame, pairs[pairs.video_id != pairs.video_id_twin]).reset_index(drop=True)
    pairs["same_gloss"] = pairs.gloss == pairs.gloss_twin
    return pairs


TWIN_GLOSSES = ("other gloss", "same gloss", "both")
"""`twin_set_counts`' columns: an instance's twins in a set are all under other glosses, all
under its own gloss, or some of each."""


def twin_set_counts(flagged: pd.DataFrame) -> pd.DataFrame:
    """Instances with a twin in each set, by the instance's own set and the twin's.

    `flagged` is `flag_twins_by_set` output. Rows are a (`set`, `twin set`) MultiIndex in `SETS`
    order, columns `TWIN_GLOSSES`. The columns are exclusive, so a row sums to the instances of
    `set` with any twin in `twin set`; an instance with twins in several sets is in several rows.
    """
    rows: dict[tuple[str, str], dict[str, int]] = {}
    for own in SETS:
        in_set = cast(pd.DataFrame, flagged[flagged.set == own])
        for twin in SETS:
            same = in_set[twin_set_column("same_gloss", twin)].to_numpy()
            other = in_set[twin_set_column("other_gloss", twin)].to_numpy()
            rows[own, twin] = {
                "other gloss": int((other & ~same).sum()),
                "same gloss": int((same & ~other).sum()),
                "both": int((same & other).sum()),
            }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["set", "twin set"])


def flag_twins_by_set(instances: pd.DataFrame) -> pd.DataFrame:
    """Add, per instance, which sets its twins are in, and the kind of each twin.

    `instances` needs `video_id` (unique), `md5`, `gloss` and `set` columns. Added boolean
    columns:

    - `same_gloss_in_<set>`, `other_gloss_in_<set>` for each of train/val/test: some other
      instance with the same clip, and the same / a different gloss, is in that set.
    - One per `TWIN_KINDS` key, relative to the instance's own set: `repeat` (same gloss, same
      set), `leak` (same gloss, other set), `conflict_same_set` and `conflict_other_set` (a
      different gloss). An instance of a clip with 3+ copies can have several.

    As for `flag_twins`, twins are only looked for within `instances`.
    """
    pairs = twin_pairs(instances)
    relation = pairs.same_gloss.map({True: "same_gloss", False: "other_gloss"})
    relations: tuple[TwinRelation, ...] = ("same_gloss", "other_gloss")
    keys: list[tuple[TwinRelation, str]] = [(r, s) for r in relations for s in SETS]
    found = (pd.crosstab(pairs.video_id, [relation, pairs.set_twin]) > 0).reindex(
        index=instances.video_id, columns=pd.MultiIndex.from_tuples(keys), fill_value=False
    )

    out = instances.copy()
    for r, s in keys:
        out[twin_set_column(r, s)] = found[r, s].to_numpy()

    own_set = out.set.to_numpy()

    def twin_in(relation: TwinRelation, own: bool) -> np.ndarray:
        """Whether a twin with `relation` is in the instance's own set (`own`) or another."""
        return np.logical_or.reduce(
            [((own_set == s) == own) & out[twin_set_column(relation, s)].to_numpy() for s in SETS]
        )

    out["repeat"] = twin_in("same_gloss", own=True)
    out["leak"] = twin_in("same_gloss", own=False)
    out["conflict_same_set"] = twin_in("other_gloss", own=True)
    out["conflict_other_set"] = twin_in("other_gloss", own=False)
    return out


Node = tuple[str, int]  # a (gloss, variation_id) pair: WLASL's finest class


T = TypeVar("T", str, Node)


def group_pairs(pairs: Iterable[tuple[T, T]]) -> list[list[T]]:
    """Connected components (union-find) of the pair graph, each sorted, largest first.

    A `(x, x)` pair adds `x` on its own, so a node with no partner still gets a component.
    """
    parent: dict[T, T] = {}

    def find(x: T) -> T:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups: dict[T, list[T]] = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g))


def duplicated_clips(flagged: pd.DataFrame) -> pd.DataFrame:
    """One row per clip with more than one copy, indexed by `md5`.

    `flagged` is `flag_twins` output. Columns: `copies`, `glosses` (distinct glosses among the
    copies), `gloss` (those glosses, ` / `-joined), `video_ids` (`, `-joined) and the copies in
    each of `SETS`.
    """
    twins = flagged[flagged.has_twin]
    clips = cast(
        pd.DataFrame,
        twins.groupby("md5").agg(
            copies=("video_id", "size"),
            glosses=("gloss", "nunique"),
            gloss=("gloss", lambda g: " / ".join(sorted(set(g)))),
            video_ids=("video_id", lambda v: ", ".join(sorted(v))),
        ),
    )
    per_set = pd.crosstab(twins.md5, twins.set).reindex(columns=list(SETS), fill_value=0)
    return clips.join(per_set)


def add_gloss_clusters(clips: pd.DataFrame) -> pd.DataFrame:
    """`duplicated_clips` output with each clip's gloss cluster.

    Glosses are in one cluster when any clip is filed under both, so a cluster is a connected
    component and can chain: `a / b` on one clip and `b / c` on another make `a / b / c`, though
    no clip has all three. Added columns: `cluster` (its glosses, ` / `-joined), `cluster_glosses`
    (how many) and `whole_cluster` (the clip is filed under every gloss of its cluster).
    """
    glosses = [g.split(" / ") for g in clips.gloss]
    clusters = group_pairs((g[0], other) for g in glosses for other in g)
    cluster_of = {gloss: c for c in clusters for gloss in c}
    out = clips.copy()
    out["cluster"] = [" / ".join(cluster_of[g[0]]) for g in glosses]
    out["cluster_glosses"] = [len(cluster_of[g[0]]) for g in glosses]
    out["whole_cluster"] = out.glosses == out.cluster_glosses
    return out


def cluster_table(clustered: pd.DataFrame) -> pd.DataFrame:
    """One row per gloss cluster of `add_gloss_clusters` output, most glosses (then clips) first.

    Columns: `glosses`, `clips`, `copies` (instances), `whole cluster` (clips filed under every
    gloss of the cluster), the copies in each of `SETS`, and `video_ids` (`, ` between the copies
    of a clip, ` | ` between clips).
    """
    table = cast(
        pd.DataFrame,
        clustered.groupby("cluster").agg(
            glosses=("cluster_glosses", "first"),
            clips=("copies", "size"),
            copies=("copies", "sum"),
            **{"whole cluster": ("whole_cluster", "sum")},
            **{s: (s, "sum") for s in SETS},
            video_ids=("video_ids", lambda v: " | ".join(sorted(v))),
        ),
    )
    return table.sort_values(["glosses", "clips", "cluster"], ascending=[False, False, True])


GLOSS_MEASURES = {
    "twin_any": "share of all its instances with a twin of any kind",
    "conflict_any": "share of all its instances with a twin under another gloss, in any set",
    "conflict_train_train": (
        "share of its train instances with a twin under another gloss in train: contradictory "
        "training targets"
    ),
    "conflict_test_train": (
        "share of its test instances with a twin under another gloss in train: the model has "
        "been taught that clip as the other gloss"
    ),
    "conflict_test_val": "share of its test instances with a twin under another gloss in val",
    "conflict_test_test": (
        "share of its test instances with a twin under another gloss in test: one input, two "
        "targets, so at most one is right"
    ),
    "leak_test": "share of its test instances with a twin under the same gloss in train or val",
    "lends_to_test": (
        "number of other glosses' test instances whose twin in train has this gloss: each is a "
        "likely false positive for it (a count, not a share)"
    ),
}
"""Per-gloss duplicate measures from `gloss_measures`, with what each counts. "Twin" means
another instance with the identical clip; the per-set conditions are inclusive (a test instance
with twins in train and test counts towards both)."""


def gloss_measures(flagged: pd.DataFrame) -> pd.DataFrame:
    """Per gloss, every `GLOSS_MEASURES` column (0 where the gloss has no instance in the set).

    `flagged` needs `flag_twins` and `flag_twins_by_set` columns.
    """

    def share(set_name: str | None, column: str) -> pd.Series:
        sub = flagged if set_name is None else flagged[flagged.set == set_name]
        return cast(pd.Series, sub.groupby("gloss")[column].mean())

    pairs = twin_pairs(flagged)
    lent = pairs[(pairs.set == "test") & (pairs.set_twin == "train") & ~pairs.same_gloss]
    glosses = pd.Index(sorted(flagged.gloss.unique()), name="gloss")
    table = pd.DataFrame(
        {
            "twin_any": share(None, "has_twin"),
            "conflict_any": share(None, "twin_other_gloss"),
            "conflict_train_train": share("train", twin_set_column("other_gloss", "train")),
            "conflict_test_train": share("test", twin_set_column("other_gloss", "train")),
            "conflict_test_val": share("test", twin_set_column("other_gloss", "val")),
            "conflict_test_test": share("test", twin_set_column("other_gloss", "test")),
            "leak_test": share("test", "leak"),
            "lends_to_test": lent.groupby("gloss_twin").video_id.nunique(),
        },
        index=glosses,
    ).fillna(0.0)
    assert list(table.columns) == list(GLOSS_MEASURES)
    return table


def shared_clip_counts(instances: pd.DataFrame) -> Counter[tuple[Node, Node]]:
    """For each pair of (gloss, variation) nodes, the number of distinct clips filed under both.

    `instances` needs `md5`, `gloss` and `variation_id` columns. Pairs are sorted tuples.
    """
    counts: Counter[tuple[Node, Node]] = Counter()
    nodes = list(zip(instances.gloss, instances.variation_id))
    for clip_nodes in pd.Series(nodes).groupby(instances.md5.to_numpy()):
        for pair in combinations(sorted(set(clip_nodes[1])), 2):
            counts[pair] += 1
    return counts


def merge_signs(instances: pd.DataFrame, min_shared: int) -> pd.Series:
    """Label each instance with its merged sign: (gloss, variation) nodes joined when they share
    at least `min_shared` clips, as connected components.

    A single shared clip joins unrelated signs into long chains (one component spans `across`,
    `over`, `repeat`, `again`, `after`, `before`, `last`, `past`), so `min_shared` should be 2 or
    more. The sign id is the smallest `gloss:variation` string in its component.
    """
    nodes = list(zip(instances.gloss, instances.variation_id))
    linked = [pair for pair, shared in shared_clip_counts(instances).items() if shared >= min_shared]
    components = group_pairs([(n, n) for n in nodes] + linked)
    names = [[f"{gloss}:{variation}" for gloss, variation in c] for c in components]
    sign_of = {node: min(n) for c, n in zip(components, names) for node in c}
    return pd.Series(
        [sign_of[node] for node in nodes],
        index=instances.index,
        name="sign",
    )


# Near duplicates: per-frame perceptual hashes. See `near_duplicates.md` for the method, results
# and why the verifier isn't good enough yet.

PHASH_STASH = STASH_DIR / "asl2000_frame_phash_results.json"
"""`hash_videos` of every WLASL-2000 clip, keyed by `video_id` (gitignored: about 27 MB and 36
min to rebuild with `stash_phashes`)."""

Hashes = npt.NDArray[np.uint64]
HashIndex = dict[int, list[str]]  # frame hash -> video_ids containing it


def _area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) matrix; row i holds the fractional coverage of each input pixel by output bin
    i, normalised to sum to 1."""
    edges = np.linspace(0, n_in, n_out + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    px = np.arange(n_in)[None, :]
    overlap = np.clip(np.minimum(hi, px + 1) - np.maximum(lo, px), 0, None)
    return (overlap / (n_in / n_out)).astype(np.float32)


def _resize_area(grey: np.ndarray, size: int) -> np.ndarray:
    """Batched area resize of T x H x W to T x `size` x `size` (float32)."""
    _, h, w = grey.shape
    return _area_matrix(h, size) @ grey @ _area_matrix(w, size).T


def _prep_frames(frames: np.ndarray, size: int = 64, canon: int = 256) -> np.ndarray:
    """T x H x W x C frames to T x `size` x `size` greyscale: area-resized to `canon` first, so
    every clip goes through the same shape, then block-averaged (`canon % size` must be 0)."""
    grey = _resize_area(frames.mean(axis=-1, dtype=np.float32), canon)
    q = canon // size
    small = grey.reshape(grey.shape[0], size, q, size, q).mean(axis=(2, 4))
    return np.rint(small).astype(np.uint8)


def frame_phashes(frames: np.ndarray, hash_size: int = 8) -> Hashes:
    """A 64-bit pHash of each frame of a T x H x W x C clip: the `hash_size` x `hash_size`
    low-frequency DCT corner of the 64 px greyscale frame, thresholded at its median."""
    dcts = cast(np.ndarray, dctn(_prep_frames(frames).astype(np.float32), axes=(1, 2)))
    low = dcts[:, :hash_size, :hash_size]
    bits = (low > np.median(low, axis=(1, 2), keepdims=True)).reshape(len(low), -1)
    return np.packbits(bits, axis=-1).view(np.uint64).squeeze(axis=-1)


def _clip_phashes(video_path: Path) -> Hashes:
    return frame_phashes(cv_load(video_path, 0, 0, all=True))


def hash_videos(
    raw_dir: Path, video_ids: Iterable[str], workers: int | None = None
) -> dict[str, Hashes]:
    """`frame_phashes` of each video file in `raw_dir`, keyed by `video_id` (decoded in parallel)."""
    ids = list(video_ids)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        hashes = pool.map(_clip_phashes, _clip_paths(raw_dir, ids), chunksize=16)
        return dict(tqdm(zip(ids, hashes, strict=True), total=len(ids), desc="Hashing"))


def stash_phashes(raw_dir: Path = RAW_DIR) -> Path:
    """Hash every WLASL-2000 clip into `PHASH_STASH`."""
    hashes = hash_videos(raw_dir, load_instances("asl2000").video_id)
    return stash_json({v: list(map(int, h)) for v, h in hashes.items()}, PHASH_STASH)


def load_phashes() -> dict[str, Hashes]:
    """`PHASH_STASH`, as arrays."""
    raw: dict[str, list[int]] = load_json(PHASH_STASH)
    return {v: np.asarray(h, dtype=np.uint64) for v, h in raw.items()}


def align_hashes(a: Hashes, b: Hashes, min_overlap: int | None = None) -> tuple[float, int]:
    """Mean per-frame Hamming distance between two clips' hashes, minimised over temporal offsets.

    Offset k lines frame j of `b` up with frame j + k of `a`: k > 0 means `b` starts k frames into
    `a`, k < 0 that `b` has -k extra frames first. Offsets overlapping fewer than `min_overlap`
    frames (default half the shorter clip) are skipped.

    Returns:
        (best mean distance, its offset k)
    """
    n1, n2 = len(a), len(b)
    if min_overlap is None:
        min_overlap = max(1, min(n1, n2) // 2)
    best_dist, best_k = float("inf"), 0
    for k in range(min_overlap - n2, n1 - min_overlap + 1):
        a0, b0 = max(k, 0), max(-k, 0)
        overlap = min(n1 - a0, n2 - b0)
        dist = float(np.bitwise_count(a[a0 : a0 + overlap] ^ b[b0 : b0 + overlap]).mean())
        if dist < best_dist:
            best_dist, best_k = dist, k
    return best_dist, best_k


def build_hash_index(hash_sets: dict[str, set[int]]) -> HashIndex:
    """Index each video once under every distinct frame hash it contains."""
    index: HashIndex = defaultdict(list)
    for vid, hashes in hash_sets.items():
        for h in hashes:
            index[h].append(vid)
    return index


def stoplist(index: HashIndex, max_videos: int) -> tuple[HashIndex, HashIndex]:
    """Split into (kept, dropped): dropped buckets hold more than `max_videos` videos.

    Single-video buckets can't form a pair, so they are discarded and appear in neither.
    """
    kept: HashIndex = {}
    dropped: HashIndex = {}
    for h, vids in index.items():
        if len(vids) > max_videos:
            dropped[h] = vids
        elif len(vids) > 1:
            kept[h] = vids
    return kept, dropped


def count_shared(index: HashIndex) -> Counter[tuple[str, str]]:
    """(a, b) with a < b -> number of distinct hashes the two videos share."""
    shared: Counter[tuple[str, str]] = Counter()
    for vids in index.values():
        shared.update(combinations(sorted(vids), 2))
    return shared


def candidates(
    shared: Counter[tuple[str, str]], hash_sets: dict[str, set[int]], min_frac: float
) -> dict[tuple[str, str], float]:
    """Pairs sharing at least `min_frac` of the smaller video's distinct hashes, with that fraction.

    `hash_sets` must be every video's full set of distinct hashes: taking the denominator from the
    stop-listed index would shrink it and inflate the fraction.
    """
    fracs = {
        (a, b): n / min(len(hash_sets[a]), len(hash_sets[b])) for (a, b), n in shared.items()
    }
    return {pair: f for pair, f in fracs.items() if f >= min_frac}


def verify(
    pairs: Iterable[tuple[str, str]], hashes: dict[str, Hashes], max_dist: float
) -> dict[tuple[str, str], tuple[float, int]]:
    """Pairs whose `align_hashes` distance is at most `max_dist`, so a pair is judged on all its
    overlapping frames, not just the hashes it shares.

    Returns:
        (a, b) -> (mean distance, offset k: frame j of b lines up with frame j + k of a)
    """
    out: dict[tuple[str, str], tuple[float, int]] = {}
    for a, b in pairs:
        dist, k = align_hashes(hashes[a], hashes[b])
        if dist <= max_dist:
            out[a, b] = (dist, k)
    return out


def find_near_duplicates(
    hashes: dict[str, Hashes], max_vids: int = 50, min_frac: float = 0.5, max_dist: float = 2.0
) -> dict[tuple[str, str], tuple[float, int]]:
    """Verified pairs (`verify`) among the candidates (`candidates` on the `stoplist`-ed index),
    printing the count at each stage and the recall on the exact copies.

    Every candidate found so far is a different sign by the same signer: see `near_duplicates.md`.
    """
    hash_sets = {v: set(map(int, h)) for v, h in hashes.items()}
    kept, dropped = stoplist(build_hash_index(hash_sets), max_vids)
    print(
        f"Videos: {len(hashes)}; kept buckets: {len(kept)}; dropped (> {max_vids} videos): "
        f"{len(dropped)}, sizes {sorted((len(v) for v in dropped.values()), reverse=True)[:10]}"
    )
    shared = count_shared(kept)
    cands = candidates(shared, hash_sets, min_frac)
    print(f"Pairs sharing a hash: {len(shared)}; candidates (frac >= {min_frac}): {len(cands)}")

    verified = verify(cands, hashes, max_dist)
    groups = group_pairs(verified)
    print(
        f"Verified pairs (mean dist <= {max_dist}): {len(verified)}; groups: {len(groups)}, "
        f"sizes {Counter(len(g) for g in groups).most_common()}"
    )
    pairs = twin_pairs(load_instances("asl2000"))
    exact = {(a, b) for a, b in zip(pairs.video_id, pairs.video_id_twin) if a < b}
    group_of = {v: i for i, g in enumerate(groups) for v in g}
    found = sum(group_of.get(a, -1) == group_of.get(b, -2) for a, b in exact)
    print(f"Recall on exact pairs: {found}/{len(exact)}")
    print(f"Verified pairs that aren't exact copies: {len(set(verified) - exact)}")
    return verified


if __name__ == "__main__":
    find_near_duplicates(load_phashes())
