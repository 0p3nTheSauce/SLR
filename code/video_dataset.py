from torch.utils.data import Dataset, DataLoader

# import os
import json
import torch
from pathlib import Path
from typing import (
    Callable,
    Optional,
    Any,
    Tuple,
    Literal,
    TypedDict,
    Union,
    List,
    TypeGuard,
    TypeAlias,
)
from torchvision.transforms import v2
from video_transforms import Shuffle

# local imports
from utils import load_rgb_frames_from_video
from video_transforms import correct_num_frames, resize_by_diag, crop_frames

from configs import WLASL_ROOT, RAW_DIR, LABELS_PATH, LABEL_SUFFIX, get_avail_splits
from models import NormDict
from preprocess import InstanceDict, AVAIL_SETS
############################# Dictionaries and Types #############################


class DataSetInfo(TypedDict):
    """Necessary info to import datast"""

    root: Path
    labels: Path
    label_suff: str
    set_name: AVAIL_SETS


def is_instance_dict(obj: Any) -> TypeGuard[InstanceDict]:
    """Type guard to check if a dict is an InstanceDict

    Args:
        obj (dict): Object to check
    Returns:
        TypeGuard[InstanceDict]: True if obj is an InstanceDict, False otherwise
    """
    try:
        _ = InstanceDict(
            bbox=obj['bbox'],
            frame_end=obj['frame_end'],
            frame_start=obj['frame_start'],
            instance_id=obj['instance_id'],
            signer_id=obj['signer_id'],
            source=obj['source'],
            split=obj['split'],
            url=obj['url'],
            variation_id=obj['variation_id'],
            video_id=obj['video_id'],
            label_name=obj['label_name'],
            label_num=obj['label_num']
        )
        return True
    except Exception:
        return False


def load_data_from_json(json_path: Union[str, Path]) -> List[InstanceDict]:
    """Load list of InstanceDict from a json file

    Args:
        json_path (Union[str, Path]): Path to json file
    Returns:
        List[InstanceDict]: List of InstanceDicts
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Data in {json_path} is not a list.")

    for item in data: #NOTE: Overhead is actually mininmal on strict ~0.019 s for WLASL2000 train.
        if not is_instance_dict(item): 
            raise ValueError(f"Item {item} in {json_path} is not a valid InstanceDict.")

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
        num_frames: Optional[int] = None,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        item_transforms: Optional[
            Callable[[torch.Tensor, InstanceDict], torch.Tensor]
        ] = None,
        include_meta: bool = False,
    ) -> None:
        """
        Custom video dataset, based on the structure of the WLASL dataset

        :param root: Path to the root directory where the video files are located.
        :type root: Path
        :param instances_path: Path to the json file with data points inside
        :type instances_path: Path
        :param classes_path: Path to the json file with class names ordered by label
        :type classes_path: Path
        :param crop: Switch to turn pre-cropping on. The crop is based on YOLO predictions for where people are.
        :type crop: bool
        :param num_frames: The desired number of frames.
        :type num_frames: int
        :param transforms: A Transform function to apply to raw videos.
        :type transforms: Optional[Callable[[torch.Tensor], torch.Tensor]]
        :param include_meta: Boolean flag to include extra meta information
        :type include_meta: bool
        :param resize: Boolean flag to resize by the diagonal (wlasl strategy)
        :type resize: bool
        :param all_frames: Boolean flag to instead keep all the flags
        :type all_frames: bool
        """
        self.root = set_info["root"]
        if not self.root.exists():
            raise FileNotFoundError(f"Root directory {self.root} does not exist.")
        self.transforms = transforms
        self.num_frames = num_frames
        self.include_meta = include_meta
        self.item_transforms = item_transforms
        instances_path = (
            set_info["labels"]
            / f"{set_info['set_name']}_instances_{set_info['label_suff']}"
        )

        self.data = load_data_from_json(instances_path)
        self.classes = set([inst['label_name'] for inst in self.data])
        self.num_classes = len(self.classes)

    def __manual_load__(self, item):
        video_path = self.root / (item["video_id"] + ".mp4")
        if video_path.exists() is False:
            raise FileNotFoundError(f"Video file {video_path} does not exist.")

        frames = load_rgb_frames_from_video(
            video_path=video_path,
            start=item["frame_start"],
            end=item["frame_end"],
        )
        if self.num_frames is not None:
            sampled_frames = correct_num_frames(frames, self.num_frames)
        else:
            sampled_frames = frames

        return sampled_frames.to(torch.uint8)

    def __getitem__(self, idx):
        item = self.data[idx]
        frames = self.__manual_load__(item)

        if self.item_transforms is not None:
            frames = self.item_transforms(frames, item)

        if self.transforms is not None:
            frames = self.transforms(frames)

        result = {"frames": frames, "label_num": item["label_num"]}
        if self.include_meta:
            result.update(item)
        return result

    def __len__(self):
        return len(self.data)


################################## Helper functions #######################################


def _identity_transform(x):
    """Identity transform - returns input unchanged"""
    return x


def _normalize_to_float(x):
    """Convert tensor to float and normalize to [0, 1]"""
    return x.float() / 255.0


def _permute_time_channel(x):
    """Permute tensor from (C, T, H, W) to (T, C, H, W)"""
    return x.permute(1, 0, 2, 3)


def _resize_by_diagonal(frames, item):
    """Resize the target diagonal to 256 before random cropping as per wlasl"""
    return resize_by_diag(frames, item["bbox"], target_diag=256)


def _crop_frames(frames, item):
    """Crop out the bounding box from the frames"""
    return crop_frames(frames, item["bbox"])


Cropping_Strategy: TypeAlias = Literal["Centre", "Random"]

def get_transform(
    norm_dict: Optional[NormDict] = None,
    frame_size: Optional[int] = None,
    shuffle: bool = False,
    num_frames: Optional[int] = None,
    crop: Optional[Cropping_Strategy] = None
    ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Optional[List[int]], Optional[float]]:
    
    
    
    # transform(frames) -> frames
    if shuffle:
        assert num_frames is not None, "num_frames must be specified if shuffle is True"
        maybe_shuffle_t = Shuffle(num_frames)
        perm = maybe_shuffle_t.permutation
        sh_e = Shuffle.shannon_entropy(perm)
        perm = list(map(int, perm.numpy()))
    else:
        maybe_shuffle_t = v2.Lambda(_identity_transform)
        perm = None
        sh_e = None

    if norm_dict is not None:
        final_transform = v2.Compose(
            [
                maybe_shuffle_t,
                v2.Lambda(_normalize_to_float),
                v2.Normalize(mean=norm_dict["mean"], std=norm_dict["std"]),
                v2.Lambda(_permute_time_channel),
            ]
        )
    else:
        final_transform = v2.Compose(
            [
                maybe_shuffle_t,
                v2.Lambda(_normalize_to_float),
                v2.Lambda(_permute_time_channel),
            ]
        )
    transform = final_transform

    if frame_size is not None:
        assert crop is not None, f'Specify crop, one of {Cropping_Strategy}'
        
        if crop == "Random":
            transform = v2.Compose(
                [
                    v2.RandomCrop(frame_size),
                    v2.RandomHorizontalFlip(),
                    final_transform,
                ]
            )
        elif crop == "Centre":
            transform = v2.Compose([v2.CenterCrop(frame_size), final_transform])
            
    return transform, perm, sh_e

def get_data_set(
    set_info: DataSetInfo,
    norm_dict: Optional[NormDict] = None,
    frame_size: Optional[int] = None,
    num_frames: Optional[int] = None,
    shuffle: bool = False,
    resize_by_diagonal: bool = False,
    cropping: Literal["Bbox", "Centre", "Random", "Default"] = "Default",
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
    # item_transform(frames, item) -> frames
    item_transforms = None
    if resize_by_diagonal:
        item_transforms = _resize_by_diagonal
    elif cropping == "Bbox":
        item_transforms = _crop_frames

    if cropping == 'Default':
        crop = 'Random' if set_info['set_name'] == 'train' else 'Centre'
    elif cropping == 'Bbox':
        crop = None
    else:
        crop = cropping

    # transform(frames) -> frames
    transform, perm, sh_e = get_transform(
        norm_dict,
        frame_size,
        shuffle,
        num_frames,
        crop=crop
    )

    dataset = VideoDataset(
        set_info,
        num_frames=num_frames,
        transforms=transform,
        item_transforms=item_transforms,
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
