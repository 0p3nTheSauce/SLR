from torch.utils.data import Dataset

# import os
import json
import torch
from pathlib import Path
from typing import (
    Callable,
    cast,
    Optional,
    Tuple,
    Literal,
    # TypedDict,
    Union,
    List,
    TypeAlias,
    Dict,
    Any,
)
from typing_extensions import TypedDict, Unpack
# local imports
from utils import load_rgb_frames_from_video
from video_transforms import (
    # correct_num_frames,
    get_transform,
)
from run_types import DataInfo
from configs import WLASL_ROOT, RAW_DIR, LABELS_PATH, LABEL_SUFFIX, get_avail_splits
from preprocess import Instance, AVAIL_SETS

############################# Dictionaries and Types #############################


class DataSetInfo(TypedDict):
    """Necessary info to import datast"""

    root: Path
    labels: Path
    label_suff: str
    set_name: AVAIL_SETS


LOAD_DATA_POLICY: TypeAlias = Literal["strict", "accepting"]


def load_data_from_json(
    json_path: Union[str, Path], policy: LOAD_DATA_POLICY
) -> List[Instance]:
    """Load list of Instance from a json file

    Args:
        json_path (Union[str, Path]): Path to json file
    Returns:
        List[Instance]: List of Instances
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Data in {json_path} is not a list.")

    if policy == "strict":
        return [Instance.model_validate(item) for item in data]

    return data


def get_wlasl_info(
    split: str, set_name: Literal["train", "test", "val"]
) -> DataSetInfo:
    """Get wlasl dataset loading information in a tpyed dict

    Args:
        split (str): One of avail_splits, E.g. asl100
        set_name (Literal['train', 'test', 'val']): Set to use

    Raises:
        ValueError: If split is not available

    Returns:
        DataSetInfo: For get_dataloader
    """
    avail_sp = get_avail_splits()
    if split not in avail_sp:
        raise ValueError(
            f"Supplied split: {split} not one of available splits: {', '.join(avail_sp)}"
        )

    return {
        "root": Path(WLASL_ROOT) / RAW_DIR,
        "labels": Path(LABELS_PATH) / split,
        "label_suff": LABEL_SUFFIX,
        "set_name": set_name,
    }


############################ Dataset Classes ############################


class VideoDataset(Dataset):
    def __init__(
        self,
        set_info: DataSetInfo,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        item_transforms: Optional[
            Callable[[torch.Tensor, Instance], torch.Tensor]
        ] = None,
        include_meta: bool = False,
        load_policy: LOAD_DATA_POLICY = "accepting",  # NOTE: this may break
    ) -> None:
        """
        Custom video dataset, based on the structure of the WLASL dataset

        :param set_info: Dictionary containing information for the location of the dataset.
        :param num_frames: The desired number of frames.
        :type num_frames: int
        :param transforms: A Transform function to apply to raw videos.
        :type transforms: Optional[Callable[[torch.Tensor], torch.Tensor]]
        :param item_transforms: A Transform function to apply to raw videos.
        :type item_transforms: Optional[Callable[[torch.Tensor, Instance], torch.Tensor]]
        :param include_meta: Boolean flag to include extra meta information
        :type include_meta: bool
        :param load_policy: Load data that does not match List[Instance] exactly. For backwards compatibility with older preprocessing strategies.
        :type load_policy: LOAD_DATA_POLICY
        """
        self.root = set_info["root"]
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory {self.root} does not exist.")
        self.transforms = transforms

        self.include_meta = include_meta
        self.item_transforms = item_transforms
        instances_path = (
            set_info["labels"]
            / f"{set_info['set_name']}_instances_{set_info['label_suff']}"
        )
        self.data = load_data_from_json(instances_path, load_policy)
        if load_policy == "accepting":
            self.data = cast(List[Dict[str, Any]], self.data)
        elif load_policy == "strict":
            self.data = [inst.model_dump() for inst in self.data]

        self.classes = set([inst["label_num"] for inst in self.data])
        self.num_classes = len(self.classes)

    def __manual_load__(self, item):
        video_path = self.root / cast(str, item["video_id"] + ".mp4")
        if video_path.exists() is False:
            raise FileNotFoundError(f"Video file {video_path} does not exist.")

        return load_rgb_frames_from_video(
            video_path=video_path,
            start=item["frame_start"],
            end=item["frame_end"],
        ).to(torch.uint8)

    def __getitem__(self, idx):
        item = cast(Dict[str, Any], self.data[idx])
        frames = self.__manual_load__(item)

        if self.item_transforms is not None:
            frames = self.item_transforms(frames, Instance.model_validate(item))

        if self.transforms is not None:
            frames = self.transforms(frames)

        if self.include_meta:
            result = {"frames": frames} | item
        else:
            result = result = {"frames": frames, "label_num": item["label_num"]}

        return result

    def __len__(self):
        return len(self.data)


################################## Helper functions #######################################

class VideoDatasetKwargs(TypedDict, total=False):
    item_transforms: Optional[Callable[[torch.Tensor, Instance], torch.Tensor]]
    include_meta: bool
    load_policy: LOAD_DATA_POLICY
    
def get_data_set(
    set_info: DataSetInfo,
    data_info: DataInfo,
    **kwargs: Unpack[VideoDatasetKwargs]
) -> Tuple[VideoDataset, Optional[List[int]], Optional[float]]:
    """
    Get the training, val or test set. Optionally, load frames unchanged.

    :param set_info: Dictionary containing information to load the dataset.
    :type set_info: DataSetInfo
    :param norm_dict: Dictionary containing mean and standard deviation. If None, don't apply normalisation.
    :type norm_dict: Optional[NormDict]
    :param frame_size: Length of Square frame. If None, no cropping applied.
    :type frame_size: int
    :param num_frames: Number of frames.
    :type num_frames: int
    :param shuffle: Whether to shuffle frames. Defaults to False.
    :type shuffle: bool
    :param resize_by_diagonal: Resize frame so person bounding box diagonal equals target_diagonal (in this case 256). (as per wlasl)
    :type resize_by_diagonal: bool
    :param cropping: Strategy to crop frames. Cut out the:
        - Bounding box (minimum of person)
        - Centre (frame size)
        - Random (frame size)
        - Default Random for train and Centre for testing/validation (frame size)
    :type cropping: Literal['Bbox', 'Centre', 'Random', 'Default']
    :return: dataset, permutation and shannon entropy
    :rtype: Tuple[VideoDataset, List[int] | None, float | None]
    """

    aug_info = (
        data_info.train_augs if set_info["set_name"] == "train" else data_info.test_augs
    )
    if aug_info is None:
        raise ValueError(
            f"Augmentation info not provided in data_info for set: {set_info['set_name']}."
        )
    # transform(frames) -> frames
    transform, perm, sh_e = get_transform(
        norm_dict=aug_info.norm_dict,  
        temporal_aug=aug_info.temporal_aug,
        spatial_aug=aug_info.spatial_aug,
    )

    dataset = VideoDataset(
        set_info,
        transforms=transform,
        **kwargs
    )

    return dataset, perm, sh_e

if __name__ == "__main__":
    # test_crop()
    # prep_train() #--run to preprocess the training data
    # prep_test()  #--run to preprocess the test data
    # prep_val() #--run to preprocess the validation data
    # fix_bad_frame_range("./preprocessed_labels/asl100/train_instances.json",
    #                     "../data/WLASL2000/") #--run to fix bad frame ranges in the training instances
    # fix_bad_frame_range("./preprocessed_labels/asl100/test_instances.json",
    #                   "../data/WLASL2000/") #--run to fix bad frame ranges in the test instances
    # fix_bad_frame_range("./preprocessed_labels/asl100/val_instances.json",
    #                   "../data/WLASL2000/") #--run to fix bad frame ranges in the validation instances
    # fix_bad_bboxes("./preprocessed_labels/asl100/train_instances.json",
    #               "../data/WLASL2000/", output='./output') #--run to fix bad bounding boxes in the training instances
    # fix_bad_bboxes("./preprocessed_labels/asl100/test_instances.json",
    #               "../data/WLASL2000/", output='./output')
    # fix_bad_bboxes("./preprocessed_labels/asl100/val_instances.json",
    #               "../data/WLASL2000/", output='./output')
    # remove_short_samples('./output/train_instances_fixed_bboxes.json',
    #                      output='./preprocessed_labels/asl100')
    # remove_short_samples('./output/test_instances_fixed_bboxes.json',
    #                      output='./preprocessed_labels/asl100')
    # remove_short_samples('./output/val_instances_fixed_bboxes.json',
    #                      output='./preprocessed_labels/asl100')
    pass
