from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader

from src import testing
from src.run_types import RUNS_PATH
from src.testing import _resolve_save_path, get_test_parser
from src.video_dataset import VideoDataset


def _args(**overrides) -> Namespace:
    base = {
        "split": "asl100",
        "model": "S3D",
        "exp_no": None,
        "sweep": None,
        "run_id": None,
        "weight_path": None,
        "checkpoint_dir_no": None,
    }
    base.update(overrides)
    return Namespace(**base)


class TestResolveSavePath:
    def test_regular_exp_no(self) -> None:
        save_path = _resolve_save_path(_args(exp_no=2))
        assert save_path == RUNS_PATH / "asl100/S3D/exp002/checkpoints"

    def test_sweep_trial(self) -> None:
        save_path = _resolve_save_path(_args(sweep="sw123", run_id="abcd"))
        assert save_path == RUNS_PATH / "asl100/S3D/sweep_sw123/abcd/checkpoints"

    def test_weight_path_override_used_directly(self) -> None:
        save_path = _resolve_save_path(_args(weight_path="/some/manual/checkpoints"))
        assert save_path == Path("/some/manual/checkpoints")

    def test_none_given_raises(self) -> None:
        with pytest.raises(ValueError):
            _resolve_save_path(_args())

    def test_exp_no_and_sweep_raises(self) -> None:
        with pytest.raises(ValueError):
            _resolve_save_path(_args(exp_no=0, sweep="sw", run_id="r"))

    def test_sweep_without_run_id_raises(self) -> None:
        with pytest.raises(ValueError):
            _resolve_save_path(_args(sweep="sw"))

    def test_weight_path_and_exp_no_raises(self) -> None:
        with pytest.raises(ValueError):
            _resolve_save_path(_args(weight_path="/x", exp_no=0))

    def test_weight_path_and_checkpoint_dir_no_raises(self) -> None:
        with pytest.raises(ValueError):
            _resolve_save_path(_args(weight_path="/x", checkpoint_dir_no=1))


class TestGetTestParser:
    def test_partial_regular_exp_no(self) -> None:
        args = get_test_parser().parse_args(["partial", "S3D", "asl100", "0", "test"])
        assert (args.exp_no, args.set_name, args.sweep) == (0, "test", None)

    def test_partial_sweep_addressing_without_exp_no(self) -> None:
        args = get_test_parser().parse_args(
            ["partial", "S3D", "asl100", "test", "--sweep", "sw123", "--run_id", "abcd"]
        )
        assert (args.exp_no, args.sweep, args.run_id) == (None, "sw123", "abcd")

    def test_manual_override_flags_parsed(self) -> None:
        args = get_test_parser().parse_args(
            [
                "partial",
                "S3D",
                "asl100",
                "test",
                "-wp",
                "/tmp/ckpt",
                "-dp",
                "/tmp/data_info.json",
                "-op",
                "/tmp/out",
            ]
        )
        assert (args.weight_path, args.data_path, args.out_path) == (
            "/tmp/ckpt",
            "/tmp/data_info.json",
            "/tmp/out",
        )

    def test_num_frames_and_frame_size_no_longer_exist(self) -> None:
        # -nf/-fs were dead code (always raised "not fixed") and have been removed.
        with pytest.raises(SystemExit):
            get_test_parser().parse_args(
                ["partial", "S3D", "asl100", "0", "test", "-nf", "16"]
            )


class _FakeDataset(VideoDataset):
    """A `VideoDataset` over in-memory items, skipping video loading."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.data = items

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"frames": torch.zeros(1), "label_num": self.data[idx]["label_num"]}


class _FixedLogits(torch.nn.Module):
    """Returns a preset logit row per call, in call order."""

    def __init__(self, logits: list[list[float]]) -> None:
        super().__init__()
        self.logits = iter(torch.tensor(logits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return next(self.logits).unsqueeze(0)


class TestInstanceTopK:
    def test_records_sorted_softmax_topk_per_video(self) -> None:
        items = [{"video_id": "v1", "label_num": 2}, {"video_id": "v2", "label_num": 0}]
        model = _FixedLogits([[0.0, 1.0, 3.0, 2.0], [5.0, 0.0, 0.0, 0.0]])
        loader = DataLoader(_FakeDataset(items), batch_size=1, shuffle=False)

        results = testing.test_instance_topk(model, loader, max_k=3)

        assert [r.video_id for r in results] == ["v1", "v2"]
        assert [r.target for r in results] == [2, 0]
        assert results[0].topk_idxs == [2, 3, 1]
        expected = torch.softmax(torch.tensor([0.0, 1.0, 3.0, 2.0]), dim=0)[[2, 3, 1]]
        assert results[0].topk_probs == pytest.approx(expected.tolist())

    def test_max_k_capped_at_num_classes(self) -> None:
        loader = DataLoader(
            _FakeDataset([{"video_id": "v1", "label_num": 0}]), batch_size=1, shuffle=False
        )
        results = testing.test_instance_topk(_FixedLogits([[1.0, 2.0]]), loader, max_k=20)
        assert results[0].topk_idxs == [1, 0]
