from pathlib import Path

import cv2
import numpy as np
import pytest

from src.utils import decoded_frame_count, header_frame_count


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 25, (32, 32))
    for i in range(7):
        writer.write(np.full((32, 32, 3), i * 30, dtype=np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize("count", [header_frame_count, decoded_frame_count])
def test_frame_counts(video: Path, count) -> None:
    assert count(video) == 7


@pytest.mark.parametrize("count", [header_frame_count, decoded_frame_count])
def test_unopenable_video_raises(tmp_path: Path, count) -> None:
    with pytest.raises(FileNotFoundError):
        count(tmp_path / "missing.mp4")
