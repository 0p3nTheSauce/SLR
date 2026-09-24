import io
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from src.que import shell as shell_module
from src.que.core import ServerState
from src.que.shell import QueShell, get_filters_drop_keys, unpack_filters


class TestUnpackFilters:
    def test_flattens_single_level(self) -> None:
        is_pos = lambda x: x > 0
        key_sets, criterions = unpack_filters({"a": is_pos})
        assert key_sets == [["a"]]
        assert criterions == [is_pos]

    def test_flattens_nested_dict(self) -> None:
        is_a = lambda x: x == "a"
        is_b = lambda x: x == "b"
        key_sets, criterions = unpack_filters({"outer": {"inner": is_a, "other": is_b}})
        assert key_sets == [["outer", "inner"], ["outer", "other"]]
        assert criterions == [is_a, is_b]

    def test_rejects_non_dict_non_callable_leaf(self) -> None:
        with pytest.raises(TypeError):
            unpack_filters({"a": 5})  # type: ignore[dict-item]

    def test_empty_filters(self) -> None:
        assert unpack_filters({}) == ([], [])


class TestGetFiltersDropKeys:
    def test_loads_filters_and_drop_keys(self, tmp_path: Path) -> None:
        filters_file = tmp_path / "filters.py"
        filters_file.write_text(
            "filters = {'admin': {'model': lambda x: x == 'S3D'}}\n"
            "drop_keys = [['results', 'check_name']]\n"
        )

        filters, drop_keys = get_filters_drop_keys(filters_file)

        assert drop_keys == [["results", "check_name"]]
        assert filters["admin"]["model"]("S3D") is True
        assert filters["admin"]["model"]("MViTv2_S") is False

    def test_missing_required_attribute_raises(self, tmp_path: Path) -> None:
        filters_file = tmp_path / "filters.py"
        filters_file.write_text("filters = {}\n")  # missing drop_keys

        with pytest.raises(AttributeError):
            get_filters_drop_keys(filters_file)


# Plain dicts stand in for runs (see tests/que/test_core.py).
RUNS: list[Any] = [
    {"model": "S3D", "acc": 0.5},
    {"model": "R3D", "acc": 0.9},
    {"model": "S3D", "acc": 0.7},
    {"model": "MVIT", "acc": 0.1},
    {"model": "S3D", "acc": 0.2},
]
S3D_BY_ACC = "-s acc -f model -c x == S3D"  # filtered+sorted view: orig [4, 0, 2]


class FakeQue:
    """Records the mutating calls the shell sends to the Que proxy."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def list_runs(self, loc: str) -> list[Any]:
        return list(RUNS)

    def edit_run(self, *args: Any) -> None:
        self.calls.append(("edit", *args))

    def place_runs(self, loc: str, runs: list[Any], index: int = 0) -> None:
        self.calls.append(("place", loc, runs, index))


class FakeServer:
    def __init__(self) -> None:
        self.que = FakeQue()

    def get_que(self) -> FakeQue:
        return self.que

    def get_daemon(self) -> None: ...
    def get_worker(self) -> None: ...
    def get_server_context(self) -> None: ...


class ShellHarness:
    def __init__(self, shell: QueShell, que: FakeQue, out: io.StringIO) -> None:
        self.shell = shell
        self.que = que
        self.out = out

    def run(self, cmd: str, arg: str) -> str:
        """Run `do_<cmd>(arg)` and return the console output."""
        self.que.calls.clear()
        self.out.seek(0)
        self.out.truncate()
        getattr(self.shell, f"do_{cmd}")(arg)
        return self.out.getvalue()


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> ShellHarness:
    # Skip the constructor's side effects: banner, the real ~/.que_shell_history
    # (plus its atexit save hook) and tmux.
    monkeypatch.setattr(QueShell, "_show_banner", lambda self: None)
    monkeypatch.setattr(QueShell, "_setup_history", lambda self: None)
    monkeypatch.setattr(shell_module, "tmux_manager", lambda: None)
    server = FakeServer()
    shell = QueShell(server)  # type: ignore[arg-type]
    out = io.StringIO()
    shell.console = Console(file=out, width=300)
    return ShellHarness(shell, server.que, out)


@pytest.fixture
def s3d_filters(tmp_path: Path) -> Path:
    filters_file = tmp_path / "filters.py"
    filters_file.write_text(
        "filters = {'model': lambda x: x == 'S3D'}\ndrop_keys = []\n"
    )
    return filters_file


class TestIndirectIndexing:
    """edit/display/copy/to_config index into the filtered/sorted view, but must
    act on the matching run in the underlying location."""

    def test_edit_maps_to_original_index(self, harness: ShellHarness) -> None:
        harness.run("edit", f"to_run 1 0.99 -de -ek acc {S3D_BY_ACC}")
        assert harness.que.calls == [("edit", "to_run", 0, ["acc"], "0.99", True)]

    def test_edit_negative_index_with_reverse(self, harness: ShellHarness) -> None:
        harness.run("edit", "to_run -1 0.99 -r -ek acc")
        assert [c[2] for c in harness.que.calls] == [0]

    def test_edit_merges_file_and_cli_filters(
        self, harness: ShellHarness, s3d_filters: Path
    ) -> None:
        harness.run("edit", f"to_run 0 1 -ek acc -ip {s3d_filters} -f acc -c x > 0.6")
        assert [c[2] for c in harness.que.calls] == [2]

    def test_edit_out_of_range(self, harness: ShellHarness) -> None:
        out = harness.run("edit", "to_run 3 0.99 -ek acc -f model -c x == S3D")
        assert harness.que.calls == []
        assert "Index 3 is out of range for to_run after filtering (length: 3)" in out

    def test_display_picks_from_view(self, harness: ShellHarness) -> None:
        out = harness.run("display", f"to_run 2 {S3D_BY_ACC}")
        assert '"acc": 0.7' in out

    def test_display_with_filter_file(
        self, harness: ShellHarness, s3d_filters: Path
    ) -> None:
        out = harness.run("display", f"to_run -1 -ip {s3d_filters} -s acc")
        assert '"acc": 0.7' in out

    def test_display_out_of_range(self, harness: ShellHarness) -> None:
        out = harness.run("display", "to_run 7")
        assert "Index 7 is out of range for to_run (length: 5)" in out

    def test_copy_places_runs_from_view(self, harness: ShellHarness) -> None:
        harness.run("copy", f"to_run old_runs -i 0 2 {S3D_BY_ACC}")
        assert harness.que.calls == [("place", "old_runs", [RUNS[4], RUNS[2]], 0)]

    def test_copy_with_filter_file(
        self, harness: ShellHarness, s3d_filters: Path
    ) -> None:
        harness.run("copy", f"to_run old_runs -i 0 -ip {s3d_filters} -r")
        assert harness.que.calls == [("place", "old_runs", [RUNS[4]], 0)]

    def test_copy_out_of_range_places_nothing(self, harness: ShellHarness) -> None:
        out = harness.run("copy", "to_run old_runs -i 0 5 -f model -c x == S3D")
        assert harness.que.calls == []
        assert "Index 5 is out of range" in out

    def test_to_config_writes_run_from_view(
        self, harness: ShellHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written: list[Any] = []
        fake_module = types.SimpleNamespace(
            write_config_file=lambda run, output=None: written.append(run)
        )
        monkeypatch.setitem(sys.modules, "src.que.runs_to_configs", fake_module)

        harness.run("to_config", f"to_run 0 {S3D_BY_ACC}")
        assert written == [RUNS[4]]

        out = harness.run("to_config", "to_run 3 -f model -c x == S3D")
        assert written == [RUNS[4]]
        assert "after filtering" in out


class FakeServerContext:
    """Mimics ServerContext.set_sweep_max_runs/get_state for the set_max_runs command."""

    def __init__(self, base_config: Path) -> None:
        self.sweep: dict[str, Any] = {
            "sweep_id": "abc",
            "sweep_project": "p",
            "sweep_entity": "e",
            "model": "S3D",
            "dataset": "WLASL",
            "split": "asl100",
            "base_config": str(base_config),
            "max_runs": 50,
        }
        self.completed = 48

    def set_sweep_max_runs(self, max_runs: int | None) -> int | None:
        previous = self.sweep["max_runs"]
        self.sweep["max_runs"] = max_runs
        return previous

    def get_state(self) -> ServerState:
        return ServerState(sweep=self.sweep, sweep_progress={"completed_runs": self.completed})  # type: ignore[arg-type]


class TestSetMaxRuns:
    @pytest.fixture
    def context(
        self, harness: ShellHarness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> FakeServerContext:
        # _get_daemon_parser does a bare `from configs import ...`, which only resolves with
        # src/ on the path (as setup.sh's `python -m que.shell` from src/ provides).
        monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "src"))
        context = FakeServerContext(tmp_path / "base.py")
        harness.shell.server_context = context  # type: ignore[assignment]
        return context

    def test_sets_cap_and_records_metadata(
        self, harness: ShellHarness, context: FakeServerContext, tmp_path: Path
    ) -> None:
        out = harness.run("daemon", "set_max_runs 60")
        assert context.sweep["max_runs"] == 60
        assert "50 → 60 (progress: 48/60)" in out
        [entry] = json.loads((tmp_path / "sweep_meta.json").read_text())
        assert entry | {"recorded": None} == {
            "recorded": None,
            "event": "set_max_runs",
            "sweep_id": "abc",
            "previous_max_runs": 50,
            "max_runs": 60,
            "completed_runs": 48,
        }

    def test_unlimited(self, harness: ShellHarness, context: FakeServerContext) -> None:
        out = harness.run("daemon", "set_max_runs -u")
        assert context.sweep["max_runs"] is None
        assert "50 → unlimited" in out

    def test_cap_at_completed_reports_complete(
        self, harness: ShellHarness, context: FakeServerContext
    ) -> None:
        out = harness.run("daemon", "set_max_runs 48")
        assert context.sweep["max_runs"] == 48
        assert "(progress: 48/48 (complete))" in out

    @pytest.mark.parametrize("arg", ["", "60 -u", "0", "-3"])
    def test_rejects_missing_conflicting_or_non_positive(
        self, harness: ShellHarness, context: FakeServerContext, arg: str
    ) -> None:
        harness.run("daemon", f"set_max_runs {arg}")
        assert context.sweep["max_runs"] == 50
