import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from factories import MakeQue, comp_run, exp_run, failed_run

from src.que import core as core_module
from src.que.core import (
    QUE_LOCATIONS,
    Que,
    QueBusy,
    QueIdxOOR,
    QueLocation,
    WorkerStateDict,
    clear_worker_process,
)
from src.run_types import WandbInfo

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


def test_clear_worker_process_keeps_exception() -> None:
    state = WorkerStateDict(
        task="training", current_run_id="abc", working_pid=123, exception="boom"
    )
    clear_worker_process(state)
    assert state == WorkerStateDict(
        task="inactive", current_run_id=None, working_pid=None, exception="boom"
    )


def locations(que: Que) -> dict[QueLocation, list[dict[str, Any]]]:
    """Every location's runs as plain dicts, for comparing a Que to its reloaded copy."""
    return {loc: [r.model_dump() for r in que.list_runs(loc)] for loc in QUE_LOCATIONS}  # type: ignore[arg-type]


def exp_nos(que: Que) -> list[str]:
    """Every run's exp_no across all locations, sorted: a run lost or duplicated mid-move shows here."""
    return sorted(r["admin"]["exp_no"] for runs in locations(que).values() for r in runs)


def _with_cur(run: Any, then: Callable[[Que], object]) -> Callable[[Que], object]:
    """Fill cur_run in memory only (not saved), then run the mutation that needs it."""

    def mutate(q: Que) -> object:
        q.cur_run = [run]
        return then(q)

    return mutate


SWEEP_WANDB = WandbInfo(entity="e", project="p", run_id="new", sweep_id="abc")

# Every Que mutation reachable from the Worker, Daemon or Shell (create_run/add_run need real
# config files and test results, so aren't covered here).
MUTATIONS: dict[str, Callable[[Que], object]] = {
    "stash_next_run": lambda q: q.stash_next_run(),
    "set_cur_run": lambda q: q.set_cur_run(exp_run("new")),
    "add_new_run": lambda q: q.add_new_run(exp_run("new"), SWEEP_WANDB, loc="cur_run"),
    "replace_cur_run": _with_cur(exp_run("new"), lambda q: q.replace_cur_run(exp_run("new2"))),
    "store_fin_run": _with_cur(exp_run("new"), lambda q: q.store_fin_run(comp_run("new"))),
    "stash_failed_run": _with_cur(exp_run("new"), lambda q: q.stash_failed_run("boom")),
    "recover_run": lambda q: q.recover_run(to_loc="to_run", from_loc="fail_runs"),
    "remove_run": lambda q: q.remove_run("old_runs", 0),
    "shuffle": lambda q: q.shuffle("to_run", 0, 1),
    "move": lambda q: q.move("to_run", "cur_run", 0),
    "clear_runs": lambda q: q.clear_runs("to_run"),
    "edit_run": lambda q: q.edit_run("to_run", 0, ["training", "max_epoch"], 5),
    "place_runs": lambda q: q.place_runs("to_run", [exp_run("new")]),
    "copy_runs": lambda q: q.copy_runs("old_runs", [0], "to_run", clean_slate=True, enum_chck=False),
    "update_runs": lambda q: q.update_runs(["training", "max_epoch"], lambda e: e + 1),
}


class TestPersistence:
    """Every Que mutation must reach disk before it returns: the server can die (e.g. a power
    outage) at any moment, and whatever only lived in memory is then lost. On 2026-10-02 this
    left sweep runs in memory-only state; see src/que/todo."""

    @pytest.fixture
    def saved_que(self, que: Que) -> Que:
        """A Que with a run in every location but cur_run, as last saved to disk."""
        que.to_run = [exp_run("t0"), exp_run("t1")]
        que.old_runs = [comp_run("o0"), comp_run("o1")]
        que.fail_runs = [failed_run("f0")]
        que.save_state()
        return que

    @pytest.mark.parametrize("mutation", MUTATIONS.values(), ids=MUTATIONS.keys())
    def test_mutation_is_on_disk_when_it_returns(
        self, saved_que: Que, make_que: MakeQue, mutation: Callable[[Que], object]
    ) -> None:
        mutation(saved_que)
        assert locations(make_que()) == locations(saved_que)

    @pytest.mark.parametrize(
        "name", ["stash_next_run", "replace_cur_run", "stash_failed_run", "recover_run", "move"]
    )
    def test_multi_step_mutation_never_saves_a_partial_state(
        self, saved_que: Que, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """A run moved between locations is popped then inserted: a save in between would put a
        state on disk with the run missing, so dying right after it would lose the run."""
        saved: list[list[str]] = []
        original_save = Que.save_state

        def recording_save(self: Que, *args: Any, **kwargs: Any) -> None:
            saved.append(exp_nos(self))
            original_save(self, *args, **kwargs)

        monkeypatch.setattr(Que, "save_state", recording_save)
        MUTATIONS[name](saved_que)
        assert saved, "the mutation was never saved"
        # these only move runs between locations, so every save should hold the final set of runs
        assert all(s == exp_nos(saved_que) for s in saved)

    def test_failed_mutation_still_saves_its_partial_changes(
        self, saved_que: Que, make_que: MakeQue
    ) -> None:
        """Disk mirrors memory even when a mutation raises partway (here after moving one run)."""
        with pytest.raises(QueBusy):
            saved_que.move("to_run", "cur_run", 0, 1)
        assert saved_que.len_loc("cur_run") == 1
        assert locations(make_que()) == locations(saved_que)

    def test_nested_mutation_saves_once(
        self, saved_que: Que, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saves: list[None] = []
        monkeypatch.setattr(Que, "save_state", lambda self: saves.append(None))
        saved_que.cur_run = [exp_run("new")]
        saved_que.replace_cur_run(exp_run("new2"))  # pop_cur_run + set_cur_run inside
        assert len(saves) == 1

    def test_auto_save_off_never_writes(self, runs_path: Path) -> None:
        que = Que(logger=logging.getLogger("test_que"), runs_path=runs_path, auto_save=False)
        que.add_new_run(exp_run("new"), SWEEP_WANDB)
        assert not runs_path.exists()

    def test_crash_mid_save_keeps_previous_file(
        self, saved_que: Que, make_que: MakeQue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A death mid-write must not leave a truncated, unloadable Runs.json."""
        before = locations(saved_que)

        def dump_then_die(obj: Any, f: Any, **kwargs: Any) -> None:
            f.write('{"to_run": [')
            raise KeyboardInterrupt

        monkeypatch.setattr(core_module.json, "dump", dump_then_die)
        with pytest.raises(KeyboardInterrupt):
            saved_que.remove_run("old_runs", 0)
        monkeypatch.undo()
        assert locations(make_que()) == before
        assert [p.name for p in saved_que.runs_path.parent.iterdir()] == ["Runs.json"]

    def test_pickled_copy_still_saves(self, saved_que: Que, make_que: MakeQue) -> None:
        """The Daemon and Worker, each holding the Que, are pickled into spawned processes."""
        copy = pickle.loads(pickle.dumps(saved_que))
        copy.remove_run("old_runs", 0)
        assert make_que().len_loc("old_runs") == 1


class TestRunTypes:
    """Each location is loaded back as one run type, so a run of another type there would make
    the saved Que fail to load -- and with it, the server fail to start."""

    def test_to_run_rejects_completed_run(self, que: Que) -> None:
        que.old_runs = [comp_run("o0")]
        with pytest.raises(TypeError, match="clean_slate"):
            que.move("old_runs", "to_run", 0)
        assert que.len_loc("old_runs") == 1

    def test_cur_run_rejects_failed_run(self, que: Que) -> None:
        with pytest.raises(TypeError):
            que.set_cur_run(failed_run("f0"))

    def test_store_fin_run_requires_results(self, que: Que) -> None:
        que.cur_run = [exp_run("r0")]
        with pytest.raises(TypeError):
            que.store_fin_run(exp_run("r0"))  # type: ignore[arg-type]
        assert que.len_loc("cur_run") == 1

    def test_store_fin_run_replaces_cur_run(self, que: Que) -> None:
        que.cur_run = [exp_run("r0")]
        que.store_fin_run(comp_run("r0"))
        assert que.len_loc("cur_run") == 0
        assert que.peak_run("old_runs", 0) == comp_run("r0")


def test_len_sweep_runs_counts_only_that_sweeps_runs_in_loc(que: Que) -> None:
    que.old_runs = [comp_run("a0", "abc"), comp_run("a1", "abc"), comp_run("x", "xyz"), comp_run("n")]
    que.fail_runs = [failed_run("a2", "abc")]
    assert que.len_sweep_runs("abc") == 2
    assert que.len_sweep_runs("abc", "fail_runs") == 1
    assert que.len_sweep_runs("missing") == 0
