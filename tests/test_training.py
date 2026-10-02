import numpy as np
import pytest
import torch
from torch import nn, optim

from src.run_types import CosAnealInfo
from src.training import get_scheduler

BASE_LRS = (1e-3, 1e-2)  # two param groups, like the backbone and classifier


def _optimizer() -> optim.Optimizer:
    groups = [{"params": [nn.Parameter(torch.zeros(1))], "lr": lr} for lr in BASE_LRS]
    return optim.AdamW(groups)


def _cosine(
    tmax: int, eta_min: float, hold: bool, warmup_epochs: int | None = None
) -> CosAnealInfo:
    warm_up = (
        {"start_factor": 0.1, "end_factor": 1.0, "warmup_epochs": warmup_epochs}
        if warmup_epochs is not None
        else None
    )
    return CosAnealInfo.model_validate(
        {
            "type": "CosineAnnealingLR",
            "tmax": tmax,
            "eta_min": eta_min,
            "hold_after_tmax": hold,
            "warm_up": warm_up,
        }
    )


def _lr_trajectory(
    optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LRScheduler, epochs: int
) -> np.ndarray:
    """LR of each param group at the start of each epoch, stepping once per epoch as
    training.py does. Shape (epochs, n_groups)."""
    lrs = []
    for _ in range(epochs):
        lrs.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    return np.array(lrs)


def _schedule(conf: CosAnealInfo, epochs: int) -> np.ndarray:
    optimizer = _optimizer()
    return _lr_trajectory(optimizer, get_scheduler(optimizer, conf), epochs)


class TestCosineHold:
    @pytest.mark.parametrize("warmup_epochs", [None, 5])
    def test_matches_periodic_until_tmax_then_holds(self, warmup_epochs: int | None) -> None:
        tmax, eta_min = 10, 1e-5
        epochs = (warmup_epochs or 0) + 3 * tmax
        held = _schedule(_cosine(tmax, eta_min, hold=True, warmup_epochs=warmup_epochs), epochs)
        periodic = _schedule(_cosine(tmax, eta_min, hold=False, warmup_epochs=warmup_epochs), epochs)

        # The trough is at warmup + tmax, or tmax + 1 without warmup (see src/TODO.md).
        trough = int(np.argmin(periodic[:, 0]))
        np.testing.assert_allclose(held[: trough + 1], periodic[: trough + 1], rtol=1e-6)
        np.testing.assert_allclose(held[trough:], eta_min, rtol=1e-6)

    def test_periodic_default_rises_after_tmax(self) -> None:
        conf = CosAnealInfo.model_validate({"type": "CosineAnnealingLR", "tmax": 10, "eta_min": 0.0})
        assert not conf.hold_after_tmax
        lrs = _schedule(conf, 25)
        trough = int(np.argmin(lrs[:, 0]))
        np.testing.assert_allclose(lrs[trough], 0.0, atol=1e-12)
        np.testing.assert_allclose(lrs[trough + 10], BASE_LRS, rtol=1e-6)

    def test_resume_from_state_dict_continues_schedule(self) -> None:
        conf = _cosine(tmax=10, eta_min=1e-5, hold=True, warmup_epochs=5)
        full = _schedule(conf, 25)

        optimizer = _optimizer()
        scheduler = get_scheduler(optimizer, conf)
        _lr_trajectory(optimizer, scheduler, 8)
        opt_state, sched_state = optimizer.state_dict(), scheduler.state_dict()

        resumed_optimizer = _optimizer()
        resumed_scheduler = get_scheduler(resumed_optimizer, conf)
        resumed_optimizer.load_state_dict(opt_state)
        resumed_scheduler.load_state_dict(sched_state)
        np.testing.assert_allclose(
            _lr_trajectory(resumed_optimizer, resumed_scheduler, 17), full[8:], rtol=1e-6
        )
