"""Ensemble of a sweep's best trials: average the softmax of their best checkpoints on one set.

Trials are ranked by best validation loss only, so the test set takes no part in choosing the
members. Run as a script to stash the result next to the sweep's other stashes:

    PYTHONPATH=. python src/results/sweeping/ensemble.py 510lhysg asl100_cutoff_9_1_indexed
"""

import sys
from pathlib import Path

import numpy as np
import torch

from src.que.core import CompExpInfo, Que
from src.results import STASH_DIR_NAME, get_out_stub, stash_json
from src.run_types import AVAIL_SETS

TOP_KS = (1, 5, 10)


def topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Percentage of instances whose target is among the `k` most probable classes."""
    top = np.argsort(-probs, axis=1)[:, :k]
    return float((top == targets[:, None]).any(axis=1).mean() * 100)


def ensemble_scores(probs: list[np.ndarray], targets: np.ndarray) -> dict[int, float]:
    """Top-k accuracy of the mean of the members' class probabilities."""
    mean = np.mean(probs, axis=0)
    return {k: topk_accuracy(mean, targets, k) for k in TOP_KS}


def best_trials(sweep_id: str, n: int) -> list[CompExpInfo]:
    """The sweep's `n` finished trials with the lowest best validation loss."""
    trials = [r for r in Que(auto_save=False).old_runs if r.wandb.sweep_id == sweep_id]
    return sorted(trials, key=lambda r: r.results.best_val_loss)[:n]


def trial_probs(run: CompExpInfo, set_name: AVAIL_SETS) -> tuple[np.ndarray, np.ndarray]:
    """Softmax scores and targets of a trial's best checkpoint on `set_name`, in set order."""
    from src.models import get_model
    from src.video_dataset import get_data_set, get_wlasl_info

    num_classes = int(run.admin.split.split("_")[0].removeprefix("asl"))
    model = get_model(run.admin.model, num_classes, drop_p=0.0)
    model.load_state_dict(torch.load(Path(run.admin.save_path) / "best.pth"))
    model.cuda().eval()
    dataset, _, _ = get_data_set(get_wlasl_info(run.admin.split, set_name), run.data)
    probs, targets = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        with torch.no_grad():
            logits = model(item["frames"].unsqueeze(0).cuda())[0].float()
        probs.append(torch.softmax(logits, dim=0).cpu().numpy())
        targets.append(int(item["label_num"]))
    del model
    torch.cuda.empty_cache()
    return np.stack(probs), np.array(targets)


def main(sweep_id: str, split: str, n_members: int = 10, set_name: AVAIL_SETS = "test") -> None:
    trials = best_trials(sweep_id, n_members)
    scored = [trial_probs(t, set_name) for t in trials]
    targets = scored[0][1]
    result = {
        "sweep_id": sweep_id,
        "set": set_name,
        "instances": len(targets),
        "members": [t.wandb.run_id for t in trials],  # by best validation loss, best first
        "single": [
            {k: topk_accuracy(p, targets, k) for k in TOP_KS} for p, _ in scored
        ],
        "ensemble": {
            n: ensemble_scores([p for p, _ in scored[:n]], targets) for n in (1, 3, 5, 10) if n <= len(scored)
        },
    }
    model = trials[0].admin.model
    stub = get_out_stub(split, model, f"{sweep_id}_ensemble")
    stash_json(result, Path(STASH_DIR_NAME) / f"{stub}.json", indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
