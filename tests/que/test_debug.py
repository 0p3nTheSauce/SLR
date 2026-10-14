from pathlib import Path

from src.que.debug import to_one_indexed_cutoff_split, to_one_indexed_split
from src.run_types import RUNS_PATH


def admin(split: str, weight_path: str | None = None) -> dict[str, str | None]:
    return {
        "split": split,
        "save_path": str(RUNS_PATH / split / "S3D" / "exp001"),
        "weight_path": weight_path,
    }


class TestToOneIndexedCutoffSplit:
    def test_moves_split_and_save_path(self) -> None:
        moved = to_one_indexed_cutoff_split(admin("asl100_cutoff_9"))
        assert moved["split"] == "asl100_cutoff_9_1_indexed"
        assert moved["save_path"] == str(RUNS_PATH / "asl100_cutoff_9_1_indexed" / "S3D" / "exp001")

    def test_other_splits_unchanged(self) -> None:
        run = admin("asl100_1_indexed")
        assert to_one_indexed_cutoff_split(run) == run

    def test_weight_path_into_another_moved_split_follows(self) -> None:
        """A transfer run's checkpoint lives in a different split's run dir, absolute or relative."""
        ckpt = RUNS_PATH / "asl100_cutoff_9" / "S3D" / "exp088" / "checkpoints" / "best.pth"
        moved = to_one_indexed_cutoff_split(admin("asl300_cutoff_9", str(ckpt)))
        assert moved["split"] == "asl300_cutoff_9_1_indexed"
        assert moved["weight_path"] == str(
            Path(str(ckpt).replace("asl100_cutoff_9", "asl100_cutoff_9_1_indexed"))
        )
        relative = to_one_indexed_cutoff_split(
            admin("asl300_cutoff_9", "runs/asl100_cutoff_9/S3D/exp088/checkpoints/best.pth")
        )
        assert relative["weight_path"] == "runs/asl100_cutoff_9_1_indexed/S3D/exp088/checkpoints/best.pth"

    def test_weight_path_outside_moved_splits_unchanged(self) -> None:
        run = admin("asl100_cutoff_9", "/elsewhere/best.pth")
        assert to_one_indexed_cutoff_split(run)["weight_path"] == "/elsewhere/best.pth"


class TestToOneIndexedSplit:
    def test_defaults_to_plain_splits(self) -> None:
        assert to_one_indexed_split(admin("asl100"))["split"] == "asl100_1_indexed"
        assert to_one_indexed_split(admin("asl100_cutoff_9"))["split"] == "asl100_cutoff_9"
