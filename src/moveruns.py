import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

# adjust to wherever these actually live
from run_types import RUNS_PATH, CompExpInfo
from src.que.core import Que  # adjust import to wherever Que lives

logger = logging.getLogger(__name__)
console = Console()


def run_label(run: CompExpInfo) -> str:
    """Human-readable identifier for logging/tables."""
    return f"{run.admin.model}/{run.admin.split}/exp{run.admin.exp_no}"


@dataclass
class RunMove:
    run: CompExpInfo
    old_run_dir: Path
    new_run_dir: Path

    @property
    def new_save_path(self) -> Path:
        return self.new_run_dir / "checkpoints"


def plan_moves(runs: list[CompExpInfo], runs_path: Path) -> list[RunMove]:
    """Find runs whose save_path split segment doesn't match admin.split."""
    moves: list[RunMove] = []

    for run in runs:
        admin = run.admin
        save_path = Path(admin.save_path)
        correct_split = admin.split

        if save_path.name != "checkpoints":
            logger.warning("%s: save_path doesn't end in 'checkpoints' (%s) - skipping",
                            run_label(run), save_path)
            continue

        run_dir = save_path.parent
        try:
            rel_parts = run_dir.relative_to(runs_path).parts
        except ValueError:
            logger.warning("%s: %s not under runs_path %s - skipping",
                            run_label(run), save_path, runs_path)
            continue

        current_split = rel_parts[0]
        if current_split == correct_split:
            continue  # already correctly placed (covers asl100_worst/bottom too)

        new_run_dir = runs_path / correct_split / Path(*rel_parts[1:])
        moves.append(RunMove(run=run, old_run_dir=run_dir, new_run_dir=new_run_dir))

    return moves


def validate_moves(moves: list[RunMove]) -> None:
    errors: list[str] = []
    seen_dsts: dict[Path, str] = {}

    for m in moves:
        label = run_label(m.run)
        if not m.old_run_dir.exists():
            errors.append(f"[{label}] source missing: {m.old_run_dir}")
        elif not m.old_run_dir.is_dir():
            errors.append(f"[{label}] source not a directory: {m.old_run_dir}")
        if m.new_run_dir.exists():
            errors.append(f"[{label}] destination already exists: {m.new_run_dir}")
        if m.new_run_dir in seen_dsts:
            errors.append(f"[{label}] destination collides with {seen_dsts[m.new_run_dir]}: {m.new_run_dir}")
        else:
            seen_dsts[m.new_run_dir] = label

    if errors:
        raise ValueError("Move validation failed:\n" + "\n".join(errors))


def show_plan(moves: list[RunMove]) -> None:
    table = Table(title="Planned run relocations")
    table.add_column("run")
    table.add_column("old dir")
    table.add_column("new dir")
    for m in moves:
        table.add_row(run_label(m.run), str(m.old_run_dir), str(m.new_run_dir))
    console.print(table)


def execute_moves(moves: list[RunMove], *, dry_run: bool = False) -> None:
    for m in moves:
        if dry_run:
            console.print(f"[dry-run] {m.old_run_dir} -> {m.new_run_dir}")
            continue
        m.new_run_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(m.old_run_dir), str(m.new_run_dir))
        m.run.admin.save_path = str(m.new_save_path)  # AdminInfo isn't frozen, so this is fine
        logger.info("moved %s -> %s", m.old_run_dir, m.new_run_dir)


def backup_runs_file(runs_path: Path) -> Path:
    """Copy the existing state file aside before mutating anything."""
    runs_path = Path(runs_path)
    if not runs_path.exists():
        raise FileNotFoundError(f"Expected existing state file at {runs_path}, found none")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = runs_path.with_name(f"{runs_path.stem}.backup_{stamp}{runs_path.suffix}")
    shutil.copy2(runs_path, backup_path)
    console.print(f"[cyan]Backed up {runs_path} -> {backup_path}[/cyan]")
    return backup_path


def main(q: Que, runs_path: Path = RUNS_PATH, *, dry_run: bool = False) -> list[RunMove]:
    moves = plan_moves(q.old_runs, runs_path)
    if not moves:
        console.print("Nothing to move - all save_paths already match their split.")
        return moves

    validate_moves(moves)
    show_plan(moves)

    if not dry_run:
        backup_runs_file(Path(q.runs_path))

    execute_moves(moves, dry_run=dry_run)

    if not dry_run:
        q.save_state()  # now defaults to overwriting q.runs_path in place, no archive

    return moves


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    q = Que()
    main(q, dry_run=args.dry_run)