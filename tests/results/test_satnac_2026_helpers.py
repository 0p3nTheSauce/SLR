import torch

from src.preprocess import Instance
from src.results.satnac_2026.helpers import VariationStepper


def _fetched(video_id: str, variation_id: int) -> tuple[Instance, torch.Tensor]:
    inst = Instance(
        bbox=[0, 0, 1, 1],
        frame_end=10,
        frame_start=0,
        instance_id=0,
        signer_id=1,
        source="src",
        split="train",
        url="",
        variation_id=variation_id,
        video_id=video_id,
        label_num=0,
        label_name="gloss",
    )
    return inst, torch.zeros(1)


def _step_ids(stepper: VariationStepper) -> list[tuple[str, int, int]]:
    return [(pick.instance.video_id, pick.position, pick.total) for pick in stepper()]


def test_steps_each_variation_in_id_order_and_drops_exhausted_ones() -> None:
    stepper = VariationStepper(
        [_fetched("a1", 1), _fetched("b0", 0), _fetched("a2", 1), _fetched("a3", 1)]
    )
    assert stepper.counts() == {0: 1, 1: 3}
    assert _step_ids(stepper) == [("b0", 1, 1), ("a1", 1, 3)]
    assert _step_ids(stepper) == [("a2", 2, 3)]
    assert _step_ids(stepper) == [("a3", 3, 3)]


def test_restarts_every_variation_after_the_longest_runs_out() -> None:
    stepper = VariationStepper([_fetched("b0", 0), _fetched("a1", 1), _fetched("a2", 1)])
    for _ in range(stepper.num_steps):
        stepper()
    assert _step_ids(stepper) == [("b0", 1, 1), ("a1", 1, 2)]
    assert stepper.step == 1


def test_no_instances_gives_empty_steps() -> None:
    stepper = VariationStepper([])
    assert stepper.num_steps == 0
    assert stepper() == []
