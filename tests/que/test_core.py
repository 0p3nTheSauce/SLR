import logging
import pickle
from pathlib import Path
from typing import Any

import pytest

from src.que.core import Que, QueIdxOOR

# Plain dicts stand in for runs: get_nested/get_nested_or_none index dicts and
# pydantic models alike, so the list-manipulation helpers don't need real configs.
RUNS: list[Any] = [
    {"model": "S3D", "acc": 0.5},
    {"model": "R3D", "acc": 0.9},
    {"model": "S3D", "acc": 0.7},
    {"model": "MVIT", "acc": 0.1},
    {"model": "S3D", "acc": 0.2},
]
IS_S3D: dict[str, Any] = {
    "filter_keys": [["model"]],
    "criterions": [lambda x: x == "S3D"],
}


class TestIndexedListManipulation:
    def test_no_manipulation_is_identity(self) -> None:
        assert Que.indexed_list_manipulation(RUNS) == ([0, 1, 2, 3, 4], RUNS)

    def test_filter_keeps_original_indexes(self) -> None:
        idxs, _ = Que.indexed_list_manipulation(RUNS, **IS_S3D)
        assert idxs == [0, 2, 4]

    def test_filters_are_anded(self) -> None:
        idxs, _ = Que.indexed_list_manipulation(
            RUNS,
            filter_keys=[["model"], ["acc"]],
            criterions=[lambda x: x == "S3D", lambda x: x > 0.3],
        )
        assert idxs == [0, 2]

    def test_sort(self) -> None:
        idxs, _ = Que.indexed_list_manipulation(RUNS, sort_keys=[["acc"]])
        assert idxs == [3, 4, 0, 2, 1]

    def test_sort_reversed(self) -> None:
        idxs, _ = Que.indexed_list_manipulation(RUNS, sort_keys=[["acc"]], reverse=True)
        assert idxs == [1, 2, 0, 4, 3]

    def test_reverse_without_sort(self) -> None:
        idxs, _ = Que.indexed_list_manipulation(RUNS, reverse=True)
        assert idxs == [4, 3, 2, 1, 0]

    def test_indexes_stay_paired_with_runs(self) -> None:
        idxs, runs = Que.indexed_list_manipulation(RUNS, sort_keys=[["acc"]], **IS_S3D)
        assert idxs == [4, 0, 2]
        assert [RUNS[i] for i in idxs] == list(runs)

    def test_missing_key_passes_none_to_criterion(self) -> None:
        idxs, _ = Que.indexed_list_manipulation(
            RUNS, filter_keys=[["missing"]], criterions=[lambda x: x is None]
        )
        assert idxs == [0, 1, 2, 3, 4]

    def test_filtering_to_empty(self) -> None:
        result = Que.indexed_list_manipulation(
            RUNS,
            filter_keys=[["model"], ["acc"]],
            criterions=[lambda _: False, lambda _: True],
        )
        assert result == ([], [])

    def test_unpaired_filters_raise(self) -> None:
        with pytest.raises(ValueError, match="equal in length"):
            Que.indexed_list_manipulation(RUNS, filter_keys=[["model"]])

    def test_list_manipulation_returns_only_runs(self) -> None:
        runs = Que.list_manipulation(RUNS, sort_keys=[["acc"]])
        assert list(runs) == [RUNS[i] for i in [3, 4, 0, 2, 1]]


class TestSelectIndexes:
    def test_maps_view_indexes_to_original(self) -> None:
        idxs = Que.select_indexes(
            "to_run", RUNS, [0, -1], sort_keys=[["acc"]], **IS_S3D
        )
        assert idxs == [4, 2]

    def test_most_negative_index_is_valid(self) -> None:
        assert Que.select_indexes("to_run", RUNS, [-3], **IS_S3D) == [0]

    @pytest.mark.parametrize("idx", [3, -4])
    def test_out_of_filtered_range_raises(self, idx: int) -> None:
        with pytest.raises(QueIdxOOR) as exc_info:
            Que.select_indexes("to_run", RUNS, [idx], **IS_S3D)
        assert str(exc_info.value) == (
            f"Index {idx} is out of range for to_run after filtering (length: 3)"
        )

    def test_out_of_range_unfiltered_message(self) -> None:
        with pytest.raises(QueIdxOOR) as exc_info:
            Que.select_indexes("old_runs", RUNS, [5], sort_keys=[["acc"]])
        assert str(exc_info.value) == (
            "Index 5 is out of range for old_runs (length: 5)"
        )


class TestSelectRuns:
    def test_selects_from_manipulated_view(self, tmp_path: Path) -> None:
        # Que's default logger writes to the real src/que/Server.log via the root
        # logger, so use one that doesn't propagate.
        logger = logging.getLogger("test_select_runs")
        logger.propagate = False
        que = Que(logger=logger, runs_path=tmp_path / "Runs.json")
        que.to_run = list(RUNS)
        runs = que.select_runs("to_run", [0, 1], sort_keys=[["acc"]], **IS_S3D)
        assert runs == [RUNS[4], RUNS[0]]


class TestQueIdxOOR:
    def test_pickle_round_trip_keeps_filtered(self) -> None:
        # Que exceptions cross the manager connection, so must survive pickling.
        err = pickle.loads(pickle.dumps(QueIdxOOR("fail_runs", 2, 1, filtered=True)))
        assert err.filtered
        assert str(err) == (
            "Index 2 is out of range for fail_runs after filtering (length: 1)"
        )
