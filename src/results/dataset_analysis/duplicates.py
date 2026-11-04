"""Find WLASL clips that are byte-identical copies of each other, and how they fall across
glosses and sets: for `duplicate_clips.ipynb` (the dataset) and
`error_analysis/duplicate_accuracy.ipynb` (their effect on a model).

The supplied clips are cut per instance (see `src/info/WLASL_info.md`), so a duplicate is one
video file filed under two `video_id`s. WLASL files many signs under several glosses, and
sometimes under the same gloss in two sets, so a copy is a label conflict (different glosses) or
a train/test leak (same gloss, different sets). Only exact copies are found: a re-encode of the
same video hashes differently.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Literal, TypeVar, cast

import numpy as np
import pandas as pd

from src.run_types import SPLIT_DIR

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


MD5_STASH = Path(__file__).parent / "stashed_results" / "clip_md5.json"
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
      different gloss). An instance in a group of 3+ can have several.

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


def copy_groups(flagged: pd.DataFrame) -> pd.DataFrame:
    """One row per clip with more than one copy (a group), indexed by `md5`.

    `flagged` is `flag_twins` output. Columns: `copies`, `glosses` (distinct glosses among the
    copies), `gloss` (those glosses, ` / `-joined), `video_ids` (`, `-joined) and the copies in
    each of `SETS`.
    """
    twins = flagged[flagged.has_twin]
    groups = cast(
        pd.DataFrame,
        twins.groupby("md5").agg(
            copies=("video_id", "size"),
            glosses=("gloss", "nunique"),
            gloss=("gloss", lambda g: " / ".join(sorted(set(g)))),
            video_ids=("video_id", lambda v: ", ".join(sorted(v))),
        ),
    )
    per_set = pd.crosstab(twins.md5, twins.set).reindex(columns=list(SETS), fill_value=0)
    return groups.join(per_set)


def add_gloss_clusters(groups: pd.DataFrame) -> pd.DataFrame:
    """`copy_groups` output with each group's gloss cluster.

    Glosses are in one cluster when any clip is filed under both, so a cluster is a connected
    component and can chain: `a / b` on one clip and `b / c` on another make `a / b / c`, though
    no clip has all three. Added columns: `cluster` (its glosses, ` / `-joined), `cluster_glosses`
    (how many) and `whole_cluster` (the clip is filed under every gloss of its cluster).
    """
    glosses = [g.split(" / ") for g in groups.gloss]
    clusters = group_pairs((g[0], other) for g in glosses for other in g)
    cluster_of = {gloss: c for c in clusters for gloss in c}
    out = groups.copy()
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
