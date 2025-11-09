from torch.utils.data import Dataset, DataLoader

# import os
import json
import torch
from pathlib import Path
from typing import Callable, Optional, Tuple, Literal, TypedDict, Union, List
from torchvision.transforms import v2
from video_transforms import Shuffle

# local imports
from utils import load_rgb_frames_from_video, crop_frames
from video_transforms import correct_num_frames
import torchvision.transforms.v2 as transforms_v2
import numpy as np
from configs import WLASL_ROOT, RAW_DIR, LABELS_PATH, LABEL_SUFFIX, get_avail_splits 


def resize_by_diag(frames: torch.Tensor, bbox: list[int], target_diag: int):
    """
    Resize frame so person bounding box diagonal equals target_diagonal

    Args:
        frame: input video frame
        bbox: (x1, y1, x2, y2) of person bounding box
        target_diagonal: desired diagonal size in pixels
    """
    x1, y1, x2, y2 = bbox

    orig_width = x2 - x1
    orig_height = y2 - y1

    curr_diag = np.sqrt(orig_width**2 + orig_height**2)

    scale_factor = target_diag / curr_diag

    # resize the tensor
    new_width = int(frames.shape[2] * scale_factor)
    new_height = int(frames.shape[3] * scale_factor)

    transform = transforms_v2.Resize((new_height, new_width))

    return transform(frames)


############################ Dataset Classes ############################


class VideoDataset(Dataset):
    def __init__(
        self,
        root: Path,
        instances_path: Path,
        classes_path: Path,
        crop: bool = False,
        num_frames: int = 64,
        transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        include_meta: bool = False,
        resize: bool = False,
    ) -> None:
        """root is the path to the root directory where the video files are located."""
        if not root.exists():
            raise FileNotFoundError(f"Root directory {root} does not exist.")
        else:
            self.root = root
        self.transforms = transforms
        self.crop = crop
        self.resize = resize
        self.num_frames = num_frames
        self.include_meta = include_meta
        with open(instances_path, "r") as f:
            self.data = json.load(f)  # created by preprocess.py
            if self.data is None:
                raise ValueError(
                    f"No data found in {instances_path}. Please check the file."
                )
        with open(classes_path, "r") as f:
            self.classes = json.load(f)

    def __manual_load__(self, item):
        video_path = self.root / (item["video_id"] + ".mp4")
        if video_path.exists() is False:
            raise FileNotFoundError(f"Video file {video_path} does not exist.")

        frames = load_rgb_frames_from_video(
            video_path=video_path, start=item["frame_start"], end=item["frame_end"]
        )
        sampled_frames = correct_num_frames(frames, self.num_frames)

        if self.crop:
            sampled_frames = crop_frames(frames, item["bbox"])

        return sampled_frames.to(torch.uint8)

    def __getitem__(self, idx):
        item = self.data[idx]
        frames = self.__manual_load__(item)

        # in the WLASL paper, they first resize the frames so that
        # the person bounding box diagnol length is 256 pixels
        # this douesnt work for us
        if self.resize:
            frames = resize_by_diag(frames, item["bbox"], 256)

        if self.transforms is not None:
            frames = self.transforms(frames)

        result = {"frames": frames, "label_num": item["label_num"]}
        if self.include_meta:
            result.update(item)
        return result

    def __len__(self):
        return len(self.data)


################################## Helper functions #######################################


class DataSetInfo(TypedDict):
    """Necessary info to import datast"""
    root: Path
    labels: Path
    label_suff: str
    set_name: Literal["train", "test", "val"]
    
def get_wlasl_info(split: str, set_name: Literal["train", "test", "val"]) -> DataSetInfo:
    avail_sp = get_avail_splits()
    if split not in avail_sp:
        raise ValueError(f"Supplied split: {split} not one of available splits: {', '.join(avail_sp)}")
    
    return {
        "root": Path(WLASL_ROOT) / RAW_DIR, 
        "labels": Path(LABELS_PATH) / split,
        "label_suff": LABEL_SUFFIX,
        "set_name": set_name  
    }
    
#NOTE: This is probably uneccessary, and should just be, get dataset or something
def get_data_loader(
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    frame_size: int,
    num_frames: int,
    set_info: DataSetInfo,
    shuffle: bool = False,
    batch_size: Optional[int] = None
    
) -> Tuple[DataLoader[VideoDataset], int, Optional[List[int]], Optional[float]]:
    """Get test, validation and training dataloaders

    Args:
        mean (Tuple[float, float, float]): Model specific mean (normalisation)
        std (Tuple[float, float, float]): Model specific standard deviation (normalisation)
        frame_size (int): Length of Square frame.
        num_frames (int): Number of frames.
        set_info (DataSetInfo): Dictionary containing information to load the dataset.
        shuffle (bool, optional): Whether to shuffle frames. Defaults to False.
        batch_size (Optional[int], optional): Batch size for dataloader. Defaults to None.

    Returns:
        Tuple[DataLoader[VideoDataset], int, Optional[List[int]], Optional[float]]: dataloader, number of classes, permutation and shannon entropy
    """


    if shuffle:
        maybe_shuffle_t = Shuffle(num_frames)
        perm = maybe_shuffle_t.permutation
        sh_e = Shuffle.shannon_entropy(perm)
        perm = list(map(int, perm.numpy()))
    else:
        maybe_shuffle_t = v2.Lambda(lambda x: x)
        perm = None
        sh_e = None

    final_transform = v2.Compose(
        [
            maybe_shuffle_t,
            v2.Lambda(lambda x: x.float() / 255.0),
            v2.Normalize(mean=mean, std=std),
            v2.Lambda(lambda x: x.permute(1, 0, 2, 3)),
        ]
    )

    if set_info["set_name"] == "train":
        transform = v2.Compose(
            [
                v2.RandomCrop(frame_size),
                v2.RandomHorizontalFlip(),
                final_transform,
            ]
        )
    else:
        transform = v2.Compose([v2.CenterCrop(frame_size), final_transform])

    instances = set_info['labels'] / f"{set_info['set_name']}_instances_{set_info['label_suff']}"
    classes = set_info['labels'] / f"{set_info['set_name']}_classes_{set_info['label_suff']}"

    dataset = VideoDataset(
        set_info['root'],
        instances,
        classes,
        num_frames=num_frames,
        transforms=transform,
    )
    num_classes = len(set(dataset.classes))

    if set_info["set_name"] == "train":
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            drop_last=False,
        )

    return dataloader, num_classes, perm, sh_e


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
