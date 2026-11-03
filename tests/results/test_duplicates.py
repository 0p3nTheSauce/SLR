from pathlib import Path

import pandas as pd

from src.results.dataset_analysis.duplicates import (
    GLOSS_MEASURES,
    TWIN_KINDS,
    conflict_clips,
    flag_twins,
    flag_twins_by_set,
    gloss_collisions,
    gloss_measures,
    hash_clips,
    merge_signs,
    set_pair_counts,
    shared_clip_counts,
    twin_set_column,
)


def instances(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(["video_id", "gloss", "set", "md5"]))


class TestFlagTwins:
    def test_unique_clip_has_no_twin(self) -> None:
        flagged = flag_twins(instances([("1", "a", "train", "x"), ("2", "b", "test", "y")]))
        assert not flagged[["has_twin", "twin_other_gloss", "twin_other_set"]].to_numpy().any()

    def test_same_clip_under_two_glosses_in_one_set_is_a_label_conflict(self) -> None:
        flagged = flag_twins(instances([("1", "a", "train", "x"), ("2", "b", "train", "x")]))
        assert flagged.has_twin.all() and flagged.twin_other_gloss.all()
        assert not flagged.twin_other_set.any()
        assert not flagged.twin_same_gloss_other_set.any()

    def test_same_gloss_in_two_sets_is_a_leak(self) -> None:
        flagged = flag_twins(instances([("1", "a", "train", "x"), ("2", "a", "test", "x")]))
        assert flagged.twin_same_gloss_other_set.all() and flagged.twin_other_set.all()
        assert not flagged.twin_other_gloss.any()

    def test_three_copies_flag_each_member_by_its_own_twins(self) -> None:
        flagged = flag_twins(
            instances([("1", "a", "train", "x"), ("2", "a", "train", "x"), ("3", "b", "test", "x")])
        )
        assert flagged.twin_other_gloss.all()
        assert not flagged.twin_same_gloss_other_set.any()  # a's twin of the same gloss is in train too


class TestFlagTwinsBySet:
    def test_each_kind_is_relative_to_the_instances_own_set(self) -> None:
        flagged = flag_twins_by_set(
            instances(
                [
                    ("1", "a", "train", "x"),
                    ("2", "a", "train", "x"),
                    ("3", "a", "test", "x"),
                    ("4", "b", "test", "x"),
                ]
            )
        ).set_index("video_id")
        assert flagged.loc["1", "repeat"] and flagged.loc["1", "leak"]
        assert flagged.loc["1", "conflict_other_set"] and not flagged.loc["1", "conflict_same_set"]
        assert flagged.loc["3", "conflict_same_set"] and not flagged.loc["3", "repeat"]
        assert flagged.loc["4", twin_set_column("other_gloss", "train")]
        assert not flagged.loc["4", twin_set_column("same_gloss", "train")]

    def test_unique_clip_has_no_flags(self) -> None:
        flagged = flag_twins_by_set(instances([("1", "a", "train", "x"), ("2", "a", "test", "y")]))
        assert not flagged[list(TWIN_KINDS)].to_numpy().any()


class TestSetPairCounts:
    def test_unordered_pairs_by_sets_and_gloss(self) -> None:
        counts = set_pair_counts(
            instances([("1", "a", "test", "x"), ("2", "b", "train", "x"), ("3", "a", "train", "x")])
        )
        assert counts.loc["train-test", "other gloss"] == 1  # 1-2
        assert counts.loc["train-test", "same gloss"] == 1  # 1-3
        assert counts.loc["train-train", "other gloss"] == 1  # 2-3
        assert counts.to_numpy().sum() == 3


class TestGlossCollisions:
    def test_clips_grouped_by_gloss_set_with_placement(self) -> None:
        flagged = flag_twins(
            instances(
                [
                    ("1", "doctor", "train", "x"),
                    ("2", "physician", "train", "x"),
                    ("3", "doctor", "train", "y"),
                    ("4", "physician", "test", "y"),
                    ("5", "past", "val", "z"),
                    ("6", "before", "test", "z"),
                ]
            )
        )
        collisions = gloss_collisions(flagged)
        assert collisions.loc["doctor / physician", "clips"] == 2
        assert collisions.loc["doctor / physician", "video_ids"] == "1, 2 | 3, 4"
        assert collisions.loc["doctor / physician", "in one set"] == 1  # x: both train
        assert collisions.loc["doctor / physician", "across sets"] == 1  # y: train and test
        assert collisions.loc["before / past", "across sets"] == 1
        assert collisions.index[0] == "doctor / physician"  # most clips first

        clips = conflict_clips(flagged)
        assert clips.loc["x", "in_one_set"] and not clips.loc["y", "in_one_set"]
        assert clips.loc["x", "gloss_set_clips"] == 2 and clips.loc["z", "gloss_set_clips"] == 1


class TestGlossMeasures:
    def test_shares_and_lent_count(self) -> None:
        flagged = flag_twins_by_set(
            flag_twins(
                instances(
                    [
                        ("1", "a", "train", "x"),
                        ("2", "b", "test", "x"),
                        ("3", "a", "train", "y"),
                        ("4", "a", "test", "z"),
                        ("5", "a", "train", "z"),
                    ]
                )
            )
        )
        table = gloss_measures(flagged)
        assert list(table.columns) == list(GLOSS_MEASURES)
        assert table.loc["a", "conflict_any"] == 1 / 4
        assert table.loc["a", "leak_test"] == 1.0
        assert table.loc["a", "conflict_train_train"] == 0.0  # 1's twin is in test
        assert table.loc["b", "conflict_test_train"] == 1.0
        assert table.loc["a", "lends_to_test"] == 1  # b's test clip is a in train
        assert table.loc["b", "lends_to_test"] == 0


class TestHashClips:
    def test_identical_files_share_a_hash(self, tmp_path: Path) -> None:
        (tmp_path / "1.mp4").write_bytes(b"same")
        (tmp_path / "2.mp4").write_bytes(b"same")
        (tmp_path / "3.mp4").write_bytes(b"other")
        hashes = hash_clips(tmp_path, ["1", "2", "3"], workers=2)
        assert hashes["1"] == hashes["2"] != hashes["3"]


def with_variations(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(["video_id", "gloss", "variation_id", "md5"]))


SIGN_ROWS = [
    ("1", "doctor", 0, "x"),
    ("2", "physician", 0, "x"),
    ("3", "doctor", 0, "y"),
    ("4", "physician", 0, "y"),
    ("5", "past", 0, "z"),
    ("6", "before", 0, "z"),
    ("7", "doctor", 1, "w"),
]


class TestMergeSigns:
    def test_shared_clips_are_counted_per_node_pair(self) -> None:
        counts = shared_clip_counts(with_variations(SIGN_ROWS))
        assert counts[(("doctor", 0), ("physician", 0))] == 2
        assert counts[(("before", 0), ("past", 0))] == 1

    def test_min_shared_keeps_a_one_off_pair_apart(self) -> None:
        signs = merge_signs(with_variations(SIGN_ROWS), min_shared=2)
        assert signs[0] == signs[1] == signs[2] == signs[3]  # doctor:0 and physician:0 share 2 clips
        assert signs[4] != signs[5]  # before and past share only one
        assert signs[6] != signs[0]  # a different variation of doctor stays its own sign

    def test_min_shared_one_merges_everything_connected(self) -> None:
        signs = merge_signs(with_variations(SIGN_ROWS), min_shared=1)
        assert signs[4] == signs[5]
