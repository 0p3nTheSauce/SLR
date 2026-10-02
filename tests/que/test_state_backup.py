import json
import subprocess
from pathlib import Path

import pytest

from src.que import state_backup
from src.que.state_backup import (
    commit_if_changed,
    init_repo,
    push,
    snapshot,
    summarise_runs,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _write_runs(state_dir: Path, old_runs: int = 0) -> None:
    runs = {"to_run": [{}], "cur_run": [], "old_runs": [{}] * old_runs, "fail_runs": []}
    (state_dir / "Runs.json").write_text(json.dumps(runs, indent=4))


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """A state dir like src/que/state/: the two tracked files, plus a snapshot and an archive."""
    state = tmp_path / "state"
    state.mkdir()
    _write_runs(state)
    (state / "Server.json").write_text("{}")
    (state / "Runs_2026-10-02_19:03:07.json").write_text("{}")
    (state / "old_ques").mkdir()
    (state / "old_ques" / "Runs_old.json").write_text("{}")
    init_repo(state)
    return state


class TestInit:
    def test_creates_repo_with_whitelist_gitignore(self, state_dir: Path) -> None:
        assert (state_dir / ".git").is_dir()
        assert _git(state_dir, "branch", "--show-current") == "main"

    def test_rerun_keeps_history(self, state_dir: Path) -> None:
        commit_if_changed(state_dir)
        init_repo(state_dir)
        assert _git(state_dir, "rev-list", "--count", "HEAD") == "1"

    def test_sets_and_updates_remote(self, state_dir: Path) -> None:
        init_repo(state_dir, remote="first")
        init_repo(state_dir, remote="second")
        assert _git(state_dir, "remote", "get-url", "origin") == "second"


class TestCommit:
    def test_tracks_only_runs_and_server_state(self, state_dir: Path) -> None:
        assert commit_if_changed(state_dir)
        tracked = _git(state_dir, "ls-files").splitlines()
        assert sorted(tracked) == [".gitignore", "Runs.json", "Server.json"]

    def test_message_summarises_locations(self, state_dir: Path) -> None:
        commit_if_changed(state_dir)
        assert _git(state_dir, "log", "-1", "--format=%s") == (
            "Que state: to_run 1, cur_run 0, old_runs 0, fail_runs 0"
        )

    def test_commits_only_when_changed(self, state_dir: Path) -> None:
        assert commit_if_changed(state_dir)
        assert not commit_if_changed(state_dir)
        (state_dir / "Runs_snapshot.json").write_text("untracked, so not a change")
        assert not commit_if_changed(state_dir)
        _write_runs(state_dir, old_runs=3)
        assert commit_if_changed(state_dir)
        assert _git(state_dir, "rev-list", "--count", "HEAD") == "2"

    def test_unreadable_runs_still_committed(self, state_dir: Path) -> None:
        (state_dir / "Runs.json").write_text("{not json")
        assert summarise_runs(state_dir / "Runs.json") == "Runs.json unreadable (JSONDecodeError)"
        assert commit_if_changed(state_dir)


class TestPush:
    @pytest.fixture
    def remote(self, tmp_path: Path, state_dir: Path) -> Path:
        bare = tmp_path / "remote.git"
        subprocess.run(["git", "init", "--bare", "--quiet", str(bare)], check=True)
        init_repo(state_dir, remote=str(bare))
        return bare

    def test_no_remote(self, state_dir: Path) -> None:
        commit_if_changed(state_dir)
        assert push(state_dir) is None

    def test_pushes_commits(self, state_dir: Path, remote: Path) -> None:
        assert snapshot(state_dir) == 0
        assert _git(remote, "log", "-1", "--format=%s", "main").startswith("Que state:")

    def test_failed_push_keeps_commit_and_retries(
        self, state_dir: Path, remote: Path, tmp_path: Path
    ) -> None:
        init_repo(state_dir, remote=str(tmp_path / "missing.git"))
        assert snapshot(state_dir) == 1
        assert _git(state_dir, "rev-list", "--count", "HEAD") == "1"

        init_repo(state_dir, remote=str(remote))
        assert snapshot(state_dir) == 0  # nothing new to commit, but the old commit is pushed
        assert _git(remote, "rev-list", "--count", "main") == "1"


def test_snapshot_without_repo(tmp_path: Path) -> None:
    assert snapshot(tmp_path) == 1


def test_main_init_then_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = tmp_path / "state"
    monkeypatch.setattr(state_backup, "STATE_DIR", state)
    assert state_backup.main(["init"]) == 0
    _write_runs(state)
    assert state_backup.main(["snapshot"]) == 0
    assert _git(state, "ls-files").splitlines() == [".gitignore", "Runs.json"]
