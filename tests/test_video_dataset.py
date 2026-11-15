from pathlib import Path

import pytest

from src.run_types import AugInfo, DataInfo
from src.video_dataset import DataSetInfo, get_data_set
from src.video_transforms import crop_to_person


@pytest.fixture
def set_info(tmp_path: Path) -> DataSetInfo:
    (tmp_path / "train_labels.json").write_text("[]")
    return DataSetInfo(root=tmp_path, labels=tmp_path, label_suff="labels.json", set_name="train")


@pytest.mark.parametrize("crop", [False, True])
def test_crop_to_person_sets_the_item_transform(set_info: DataSetInfo, crop: bool) -> None:
    augs = AugInfo(strict_size=False)
    data_info = DataInfo(train_augs=augs, test_augs=augs, strict_size=False, crop_to_person=crop)
    dataset, _, _ = get_data_set(set_info, data_info)
    assert dataset.item_transforms is (crop_to_person if crop else None)
