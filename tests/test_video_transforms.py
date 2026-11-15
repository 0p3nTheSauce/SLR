import pytest
import torch

from src.preprocess import Instance
from src.video_transforms import crop_to_person, square_around


def _instance(bbox: list[int]) -> Instance:
    return Instance.model_validate(
        {
            "bbox": bbox,
            "frame_end": 10,
            "frame_start": 0,
            "instance_id": 0,
            "signer_id": 1,
            "source": "test",
            "split": "train",
            "url": "",
            "variation_id": 0,
            "video_id": "00000",
            "label_num": 0,
            "label_name": "book",
        }
    )


class TestSquareAround:
    def test_centred_on_a_tall_box(self) -> None:
        assert square_around([100, 50, 140, 150], 256, 256) == [70, 50, 170, 150]

    def test_shifted_inside_the_frame(self) -> None:
        assert square_around([0, 0, 40, 100], 256, 256) == [0, 0, 100, 100]
        assert square_around([216, 156, 256, 256], 256, 256) == [156, 156, 256, 256]

    def test_side_capped_at_the_shorter_frame_side(self) -> None:
        assert square_around([0, 0, 300, 120], 200, 320) == [50, 0, 250, 200]


def test_crop_to_person_returns_the_resized_square() -> None:
    frames = torch.zeros(4, 3, 256, 256, dtype=torch.uint8)
    frames[:, :, 50:150, 70:170] = 255  # exactly the square around the box below
    out = crop_to_person(frames, _instance([100, 50, 140, 150]), size=64)
    assert out.shape == (4, 3, 64, 64)
    assert out.dtype == torch.uint8
    assert torch.all(out == 255)


@pytest.mark.parametrize("size", [224, 256])
def test_crop_to_person_size(size: int) -> None:
    frames = torch.zeros(2, 3, 256, 256, dtype=torch.uint8)
    assert crop_to_person(frames, _instance([54, 23, 201, 255]), size=size).shape[-2:] == (
        size,
        size,
    )
