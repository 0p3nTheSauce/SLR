"""Version the Que's state in a git repo of its own.

`STATE_DIR` (`src/que/state/`, gitignored by the project) holds its own git repo, which tracks
only `Runs.json` and `Server.json`: the archives in `old_ques/` and `save -t` snapshots stay
untracked. A systemd timer installed by `setup.sh` runs `snapshot` every 15 minutes, which
commits the files if they changed and pushes to `origin` if one is set. This keeps every
version of the Que out of the project's history while still backing it up off the machine.

    python -m src.que.state_backup init [--remote URL]   # create the repo (idempotent)
    python -m src.que.state_backup snapshot              # commit if changed, then push

Run from the repo root. The server writes both files atomically, so a snapshot never reads a
half-written file, but the two files may be from moments apart.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.que.core import QUE_LOCATIONS, RUN_PATH, SERVER_STATE_PATH, STATE_DIR

TRACKED = (RUN_PATH.name, SERVER_STATE_PATH.name)
# Ignore everything, then allow back only the tracked files.
GITIGNORE = "*\n!.gitignore\n" + "".join(f"!{name}\n" for name in TRACKED)
REMOTE = "origin"
BRANCH = "main"
COMMITTER = ("Que state backup", "que-state-backup@localhost")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git in `repo`, raising with git's own error message if `check` and it fails.
    Prompts are disabled (no terminal under systemd), so a push that needs a password or an
    unknown host key fails instead of hanging."""
    env = os.environ | {"GIT_TERMINAL_PROMPT": "0", "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"}
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False, env=env
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result


def init_repo(state_dir: Path = STATE_DIR, remote: str | None = None) -> None:
    """Create the state repo if it doesn't exist, (re)write its .gitignore, and set `origin`
    to `remote` if one is given. Safe to run again on an existing repo."""
    state_dir.mkdir(parents=True, exist_ok=True)
    if not (state_dir / ".git").exists():
        _git(state_dir, "init", "--initial-branch", BRANCH)
        _git(state_dir, "config", "user.name", COMMITTER[0])
        _git(state_dir, "config", "user.email", COMMITTER[1])
        _git(state_dir, "config", "commit.gpgsign", "false")  # unattended: no key prompts
    (state_dir / ".gitignore").write_text(GITIGNORE)
    if remote is not None:
        has_remote = _git(state_dir, "remote", "get-url", REMOTE, check=False).returncode == 0
        _git(state_dir, "remote", "set-url" if has_remote else "add", REMOTE, remote)


def summarise_runs(runs_path: Path) -> str:
    """How many runs each Que location holds, e.g. `to_run 2, cur_run 1, old_runs 412,
    fail_runs 9`, for the commit message."""
    try:
        runs = json.loads(runs_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return f"{runs_path.name} unreadable ({type(e).__name__})"
    return ", ".join(f"{loc} {len(runs.get(loc, []))}" for loc in QUE_LOCATIONS)


def commit_if_changed(state_dir: Path = STATE_DIR) -> bool:
    """Commit the tracked files if they changed since the last commit. Returns whether a
    commit was made."""
    _git(state_dir, "add", "--all", ".")
    if _git(state_dir, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return False
    _git(state_dir, "commit", "--quiet", "-m", f"Que state: {summarise_runs(state_dir / RUN_PATH.name)}")
    return True


def push(state_dir: Path = STATE_DIR) -> bool | None:
    """Push to `origin`, including commits an earlier failed push left behind. Returns None if
    there's no remote, else whether the push succeeded."""
    if _git(state_dir, "remote", "get-url", REMOTE, check=False).returncode != 0:
        return None
    if _git(state_dir, "rev-parse", "--verify", "--quiet", "HEAD", check=False).returncode != 0:
        return True  # nothing committed yet
    result = _git(state_dir, "push", "--quiet", "--set-upstream", REMOTE, BRANCH, check=False)
    if result.returncode != 0:
        print(f"Push to {REMOTE} failed: {result.stderr.strip()}", file=sys.stderr)
    return result.returncode == 0


def snapshot(state_dir: Path = STATE_DIR) -> int:
    """Commit the state if it changed, then push. Returns an exit status: 1 if the repo is
    missing or the push failed (the commit is kept, and the next snapshot pushes it)."""
    if not (state_dir / ".git").exists():
        print(f"No state repo in {state_dir}: run `init` first", file=sys.stderr)
        return 1
    if commit_if_changed(state_dir):
        print(_git(state_dir, "log", "-1", "--format=%h %s").stdout.strip())
    return 1 if push(state_dir) is False else 0


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Version the Que's state in its own git repo")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create the state repo (safe to rerun)")
    init.add_argument("--remote", help=f"URL to push to, set as `{REMOTE}`")
    commands.add_parser("snapshot", help="Commit the state if it changed, then push")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = get_parser().parse_args(argv)
    if args.command == "init":
        init_repo(STATE_DIR, remote=args.remote)
        print(f"State repo ready in {STATE_DIR}")
        return 0
    return snapshot(STATE_DIR)


if __name__ == "__main__":
    sys.exit(main())
