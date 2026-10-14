"""Find WLASL clips that are byte-identical copies of each other, for `duplicate_clips.ipynb`.

The supplied clips are cut per instance (see `src/info/WLASL_info.md`), so a duplicate is one
video file filed under two `video_id`s. WLASL files many signs under several glosses, and
sometimes under the same gloss in two sets, so a copy is a label conflict (different glosses) or
a train/test leak (same gloss, different sets). Only exact copies are found: a re-encode of the
same video hashes differently.
"""

import hashlib
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import pandas as pd

TWIN_COLUMNS = ["has_twin", "twin_other_gloss", "twin_other_set", "twin_same_gloss_other_set"]


def clip_md5(video_path: Path) -> str:
    """MD5 of a whole clip (clips are small, so it is read in one go)."""
    return hashlib.md5(video_path.read_bytes()).hexdigest()


def hash_clips(raw_dir: Path, video_ids: Iterable[str], workers: int = 8) -> dict[str, str]:
    """MD5 of each video file in `raw_dir` (`<video_id>.<ext>`), keyed by `video_id`."""
    ids = list(video_ids)
    paths = [next(raw_dir.glob(f"{video_id}.*")) for video_id in ids]
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(ids, pool.map(clip_md5, paths)))


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
    by_clip = instances.groupby("md5")
    by_clip_gloss = instances.groupby(["md5", "gloss"])
    out = instances.copy()
    out["has_twin"] = by_clip["gloss"].transform("size") > 1
    out["twin_other_gloss"] = by_clip["gloss"].transform("nunique") > 1
    out["twin_other_set"] = by_clip["set"].transform("nunique") > 1
    out["twin_same_gloss_other_set"] = by_clip_gloss["set"].transform("nunique") > 1
    return out


def class_duplicate_table(flagged: pd.DataFrame) -> pd.DataFrame:
    """Per gloss: how much of it is affected by duplicates, for correlating with per-class F1.

    Columns (all fractions of the gloss's instances in the named set, or all sets):

    - `conflict_flag`: the gloss is in any label conflict (any set); the binary measure.
    - `conflict_all`, `twin_all`: instances with a twin under a different gloss / any twin.
    - `conflict_train`, `conflict_test`: the same conflict share within one set.
    - `leak_test`: test instances whose same-gloss twin is in train or val.
    """

    def share(frame: pd.DataFrame, column: str, set_name: str | None = None) -> pd.Series:
        sub = frame if set_name is None else frame[frame.set == set_name]
        return sub.groupby("gloss")[column].mean()

    glosses = pd.Index(sorted(flagged.gloss.unique()), name="gloss")
    table = pd.DataFrame(
        {
            "conflict_all": share(flagged, "twin_other_gloss"),
            "twin_all": share(flagged, "has_twin"),
            "conflict_train": share(flagged, "twin_other_gloss", "train"),
            "conflict_test": share(flagged, "twin_other_gloss", "test"),
            "leak_test": share(flagged, "twin_same_gloss_other_set", "test"),
        },
        index=glosses,
    ).fillna(0.0)
    table["conflict_flag"] = table.conflict_all > 0
    return table


Node = tuple[str, int]  # a (gloss, variation_id) pair: WLASL's finest class


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
    parent: dict[Node, Node] = {}

    def find(node: Node) -> Node:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for node in zip(instances.gloss, instances.variation_id):
        find(node)
    for (a, b), shared in shared_clip_counts(instances).items():
        if shared >= min_shared:
            parent[find(a)] = find(b)

    members: dict[Node, list[str]] = {}
    for node in list(parent):
        members.setdefault(find(node), []).append(f"{node[0]}:{node[1]}")
    sign_of = {root: min(names) for root, names in members.items()}
    return pd.Series(
        [sign_of[find(node)] for node in zip(instances.gloss, instances.variation_id)],
        index=instances.index,
        name="sign",
    )
