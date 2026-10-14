from pathlib import Path

import pandas as pd

from src.results.dataset_analysis.duplicates import (
    class_duplicate_table,
    flag_twins,
    hash_clips,
    merge_signs,
    shared_clip_counts,
)


def instances(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["video_id", "gloss", "set", "md5"])


class TestFlagTwins:
    def test_unique_clip_has_no_twin(self) -> None:
        flagged = flag_twins(instances([("1", "a", "train", "x"), ("2", "b", "test", "y")]))
        assert not flagged[["has_twin", "twin_other_gloss", "twin_other_set"]].any().any()

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


class TestClassDuplicateTable:
    def test_fractions_and_flag(self) -> None:
        flagged = flag_twins(
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
        table = class_duplicate_table(flagged)
        assert table.loc["a", "conflict_all"] == 1 / 4
        assert table.loc["a", "leak_test"] == 1.0
        assert table.loc["a", "conflict_train"] == 1 / 3
        assert table.loc["b", "conflict_test"] == 1.0
        assert table.loc["b", "conflict_train"] == 0.0
        assert table.conflict_flag.all()


class TestHashClips:
    def test_identical_files_share_a_hash(self, tmp_path: Path) -> None:
        (tmp_path / "1.mp4").write_bytes(b"same")
        (tmp_path / "2.mp4").write_bytes(b"same")
        (tmp_path / "3.mp4").write_bytes(b"other")
        hashes = hash_clips(tmp_path, ["1", "2", "3"], workers=2)
        assert hashes["1"] == hashes["2"] != hashes["3"]


def with_variations(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["video_id", "gloss", "variation_id", "md5"])


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
