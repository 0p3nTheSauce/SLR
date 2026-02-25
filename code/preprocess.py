import json
import torch
import tqdm
from ultralytics import YOLO  # type: ignore

# NOTE: Running this script will mess up the environment you are using, becuase of this stupid YOLO thing
# it will give a '3D conv not implemented yada yada' error message
# The solution is to delete and recreate the environment
import cv2
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict, TypedDict, TypeAlias, Literal, Optional, Any, Tuple

# local imports
from utils import load_rgb_frames_from_video
from configs import WLASL_ROOT, SPLIT_DIR, RAW_DIR, LABELS_PATH

"""Naming convention:
- set: one of train, test and val
- split: one of asl100, asl300, asl1000, asl2000"""

AVAIL_SETS: TypeAlias = Literal["train", "val", "test"]
AVAIL_SPLITS: TypeAlias = Literal["asl100", "asl300", "asl1000", "asl2000"]


class instance_dict(TypedDict):
    """Represents a single instance of a gloss in the dataset. This is the format that the data is currently in, and will be modified by the preprocessing functions."""

    bbox: List[int]
    frame_end: int
    frame_start: int
    instance_id: int
    signer_id: int
    source: str
    split: str
    url: str
    variation_id: int
    video_id: str


class InstanceDict(instance_dict):
    """Represents a single instance of a gloss in the dataset, with the label_num and label_name added. This is the format that the data will be in after preprocessing."""
    label_num: int # [x1, y1, x2, y2]
    label_name: str

def to_instance(d: instance_dict, label_num: int, label_name: str) -> InstanceDict:
    return {**d, "label_num": label_num, "label_name": label_name}


class gloss_dict(TypedDict):
    """Represents a single gloss and its associated instances."""

    gloss: str
    instances: List[instance_dict]


def get_set(
    lst_gloss_dicts: List[gloss_dict], set_name: AVAIL_SETS
) -> List[InstanceDict]:
    """Filters list of gloss dict, based on whether the instances are from the provided set_name.

    Args:
        lst_gloss_dicts (List[gloss_dict]): Straight from one of the split.json files in WLASL/splits
        set_name (AVAIL_SETS): One of train, val or test

    Returns:
        List[instance]: The individual instances inside the gloss dicts, with label_num and label_name added
        - class_names: The corresponding text gloss labels of the mod_instances
    """
    mod_instances = []
    for i, gloss_d in enumerate(lst_gloss_dicts):
        for inst in gloss_d["instances"]:
            if inst["split"] == set_name:
                mod_instances.append(to_instance(inst, i, gloss_d["gloss"]))           
    return mod_instances


def fix_bad_frame_range(
    raw_path: Path,
    instances: List[InstanceDict],
    log_dir: Path,
    remove_policy: Literal["strict", "reset_frames"] = "strict",
    file_extension: str = "bad_frame_ranges.txt",
) -> List[InstanceDict]:
    """Remove videos where the file cannot be read, or the start or end frame are impossible.
    Also logs any bad frames found for particular videos

    Args:
        raw_path (Path): Path to root directory of videos
        instances (List[instance]): Instance dictionaries
        log_dir (Path): Directory to place logged errors
        remove_policy (Literal["strict", "reset_frames"], optional): If the remove policy is strict,
            then the instance is removed, otherwise an attempt is made to fix the index. In the case of
            start: the start frame is set to 0
            end: the end frame is set to start + numframes
        . Defaults to "strict".
        file_extension (str, optional): Name of output file. Defaults to "bad_frame_ranges.txt".

    Returns:
        List[instance]: Instances filtered of incorrect indeces
    """

    bad_frames = []
    fixed_instances = []
    for instance in tqdm.tqdm(instances, desc="fixing frame ranges"):
        vid_path = raw_path / (instance["video_id"] + ".mp4")

        cap = cv2.VideoCapture(
            vid_path,
        )
        if not cap.isOpened():
            bad_frames.append(
                f"Error: Could not open video {instance['video_id']}. Removing"
            )
            continue
        else:
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start = instance["frame_start"]
        end = instance["frame_end"]
        if start < 0 or start >= num_frames:
            message = f"Invalid start frame {start} for video {instance['video_id']} with length {num_frames}."
            if remove_policy == "strict":
                bad_frames.append(message + " Removing instance.")
                continue
            elif remove_policy == "reset_frames":
                bad_frames.append(message + " Setting to 0.")
            start = 0
        if end <= start or end > (start + num_frames):
            message = f"Invalid end frame {end} for video {instance['video_id']} with length {num_frames} and start frame {start}."
            if remove_policy == "strict":
                bad_frames.append(message + " Removing instance.")
                continue
            elif remove_policy == "reset_frames":
                bad_frames.append(message + " Setting to num_frames.")
            end = start + num_frames
        instance["frame_start"] = start
        instance["frame_end"] = end
        fixed_instances.append(instance)

    log_path = log_dir / f"{remove_policy}_{file_extension}"
    if bad_frames:
        with open(log_path, "a") as log_file:
            log_file.write(
                f"\n New bad frames: Policy: {remove_policy} Total removed videos: {len(bad_frames)}\n "
            )
            for line in bad_frames:
                log_file.write(line + "\n")
        print(f"Bad frame ranges logged to {log_path}.")
    else:
        print("No bad frame ranges found")

    return fixed_instances


def get_largest_bbox(bboxes: List[List[float]]) -> Optional[List[float]]:
    """Given a list of bounding boxes, returns the largest bounding box that encompasses all of them, if one exists.

    Args:
        bboxes (List[List[float]]): List of bounding boxes, where each bounding box is represented as a list of four floats: [x_min, y_min, x_max, y_max].

    Returns:
        Optional[List[float]]: The largest bounding box that encompasses all of the input bounding boxes, or None if the input list is empty.
    """
    if not bboxes:
        return None
    x_min, y_min, x_max, y_max = bboxes[0]
    for box in bboxes:
        x1, y1, x2, y2 = box
        if x1 < x_min:
            x_min = x1
        if y1 < y_min:
            y_min = y1
        if x2 > x_max:
            x_max = x2
        if y2 > y_max:
            y_max = y2
    return [x_min, y_min, x_max, y_max]


def fix_bad_bboxes(
    raw_path: Path,
    instances: List[InstanceDict],
    log_dir: Path,
    remove_policy: Literal["strict", "reset_bbox"] = "strict",
    file_extension: str = "bad_bboxes.txt",
) -> List[InstanceDict]:
    """Fix bad bounding boxes by running a pre-trained YOLOv8 model on the video,
    and taking the largest bounding box across all frames. If no bounding boxes are found,
    the instance is either removed or the bbox is set to the whole frame, depending on the remove_policy.

    Args:
        raw_path (Path): Path to root directory of videos
        instances (List[instance]): Instance dictionaries
        log_dir (Path): Directory to place logged errors
        remove_policy (Literal["strict", "reset_bbox"], optional): Policy for handling bad bounding boxes. Defaults to "strict".
        file_extension (str, optional): File extension for the log file. Defaults to "bad_bboxes.txt".

    Raises:
        ValueError: If an invalid remove_policy is provided.

    Returns:
        List[instance]: Instances with corrected bounding boxes, or removed if the policy is strict and no bounding box could be found.
    """

    # TODO: could be a bit faster
    #NOTE: This function can sometimes bork your conda env, blame ultralytics. If it does remake the env. 
    model = YOLO("yolov8n.pt")  # Load a pre-trained YOLO model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    new_instances = []
    bad_bboxes = []

    for instance in tqdm.tqdm(instances, desc="Fixing bounding boxes"):
        vid_path = raw_path / (instance["video_id"] + ".mp4")
        frames = load_rgb_frames_from_video(
            vid_path, instance["frame_start"], instance["frame_end"]
        )
        frames = frames.float() / 255.0

        results = model(frames, device=device, verbose=False)
        bboxes = []
        for result in results:
            person_bboxes = result.boxes.xyxy[result.boxes.cls == 0]
            if len(person_bboxes) > 0:
                bboxes.extend(person_bboxes.tolist())

        if not bboxes:
            message = f"No bounding boxes found for video {instance['video_id']}."
            if remove_policy == "strict":
                bad_bboxes.append(message + " Removing instance.")
                continue
            elif remove_policy == "reset_bbox":
                bad_bboxes.append(
                    f"No bounding boxes found for video {instance['video_id']}. Using whole frame"
                )
                largest_bbox = [0, 0, frames.shape[3], frames.shape[2]]
            else:
                raise ValueError(f"Invalid remove_policy: {remove_policy}")
        else:
            largest_bbox = get_largest_bbox(bboxes)
            assert largest_bbox is not None, "largest_bbox can not be None here"

        largest_bbox = [
            round(coord) for coord in largest_bbox
        ]  # Round the coordinates to integers
        new_instances.append(
            instance | {"bbox": largest_bbox}
        )  # update with fresh bbox

    log_path = log_dir / f"{remove_policy}_{file_extension}"

    if bad_bboxes:
        with open(log_path, "a") as log_file:
            log_file.write(
                f"\n New bad bboxes Policy: {remove_policy} Total removed videos: {len(bad_bboxes)} \n "
            )
            for line in bad_bboxes:
                log_file.write(line + "\n")
        print(f"Bad bounding boxes logged to {log_path}.")
    else:
        print("No bad bounding boxes")

    return new_instances


def remove_short_samples(
    instances: List[InstanceDict],
    log_dir: Path,
    cutoff: int=9,
    file_extension: str = "removed_short_samples.txt",
    ) -> List[InstanceDict]:
    """Remove samples where the number of frames is less than or equal to the cutoff. Logs any removed samples.

    Args:
        instances (List[InstanceDict]): Instance dictionaries
        log_dir (Path): Directory to place logged errors
        cutoff (int, optional): Minimum number of frames required. Defaults to 9.
        file_extension (str, optional): File name for logging removed samples. Defaults to "removed_short_samples.txt".

    Returns:
        List[InstanceDict]: Instances with short samples removed.
    """
    
    mod_instances = []
    short_samples = []
    for inst in instances:
        num_frame = inst["frame_end"] - inst["frame_start"]
        if num_frame > cutoff:
            mod_instances.append(inst)
        else:
            short_samples.append(
                f"bad number of frames {num_frame} for video {inst['video_id']}, removing."
            )

    log_path = log_dir / f"cutoff_{cutoff}_{file_extension}"

    if short_samples:
        with open(log_path, "a") as log_file:
            log_file.write(f"\n New short samples Total removed: {len(short_samples)} \n ")
            for line in short_samples:
                log_file.write(line + "\n")
        print(f"short samples logged to {log_path}")
    else:
        print("no short samples")

    return mod_instances


def print_v(s: str, y: bool) -> None:
    if y:
        print(s)


def check_paths(
    split_path: Path, raw_path: Path, output_path: Path, verbose: bool
) -> bool:
    """Checks if the provided paths exist and are of the correct type (file or directory).
    Returns True if all paths are valid, False otherwise. Prints the status of each path based on the verbose flag."""
    if split_path.exists() and split_path.is_file():
        print_v(f"split path: {split_path}, found", verbose)
    else:
        print(f"split path: {split_path}, not found")
        return False
    if raw_path.exists() and raw_path.is_dir():
        print_v(f"raw path: {raw_path}, found", verbose)
    else:
        print(f"raw path: {raw_path}, not found")
        return False
    if output_path.exists() and output_path.is_dir():
        print_v(f"output path: {output_path}, found", verbose)
    else:
        print(f"output path: {output_path}, not found")
        return False
    return True




def preprocess_split(
    split_path: Path,
    raw_path: Path,
    output_base: Path,
    verbose: bool = False,
    file_extension: str = "_fixed_frange_bboxes_len.json"
) -> None:
    """Preprocesses a split of the WLASL dataset by fixing bad frame ranges, fixing bad bounding boxes, and removing short samples. 
    The processed data is saved to the output directory, and any issues found during preprocessing are
    logged.

    Args:
        split_path (Path): Path to the split json file, which contains the glosses and their associated instances for a particular split (train, val or test).
        raw_path (Path): Path to the raw video directory.
        output_base (Path): Path to the output base directory.
        verbose (bool, optional): If True, print verbose output. Defaults to False.
        file_extension (str, optional): File extension for the processed instances. Defaults to "_fixed_frange_bboxes_len.json".
    """


    if not check_paths(split_path, raw_path, output_base, verbose):
        return

    with open(split_path, "r") as f:
        asl_num = json.load(f)

    if not asl_num:
        print(f"no data found in {split_path}")
        return

    # create train, test, val splits
    train_instances = get_set(asl_num, "train")
    test_instances = get_set(asl_num, "test")
    val_instances = get_set(asl_num, "val")

    # setup storage
    base_name = split_path.name.replace(".json", "")
    output_dir = output_base / base_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print_v(f"Processing {base_name}", verbose)
    for subset, instances in [
        ("train", train_instances),
        ("test", test_instances),
        ("val", val_instances),
    ]:
        print_v(f"For split: {subset}", verbose)
        # fix badly labeled frame ranges
        print_v("Fixing frame ranges", verbose)
        instances = fix_bad_frame_range(
            raw_path=raw_path,
            instances=instances,
            log_dir=output_dir,
            remove_policy="strict",
            file_extension=f"bad_frame_ranges_{subset}.txt",
        )

        # fix badly labeled bounding boxes
        print_v("Fixing bounding boxes", verbose)
        instances = fix_bad_bboxes(
            raw_path=raw_path,
            instances=instances,
            log_dir=output_dir,
            remove_policy="reset_bbox",
            file_extension=f"bad_bboxes_{subset}.txt",
        )

        # finally, remove short samples
        print_v("Removing small samples", verbose)

        instances = remove_short_samples(
            instances=instances,
            log_dir=output_dir,
            cutoff=9,
            file_extension=f"removed_short_samples_{subset}.txt",
        )

        # save
        print_v("Saving results", verbose)
        inst_path = output_dir / f"{subset}_instances{file_extension}"
        with open(inst_path, "w") as f:
            json.dump(instances, f, indent=2)

    print()
    print("------------------------- finished preprocessing ---------------")
    print()


# NOTE: it is slow, especially for the bigger datasets, mostly held up
# by fixing the bounding boxes, but this doesn't totally exhause the GPU.
# so could potentially allocate more processes to the task
if __name__ == "__main__":
    parser = ArgumentParser(description="preprocess.py")
    parser.add_argument(
        "asl_split",
        type=str,
        choices=["asl100", "asl300", "asl1000", "asl2000"],
        help="Which WLASL split to preprocess",
    )
    parser.add_argument(
        "-rt",
        "--root",
        type=str,
        help=f"WLASL root if not {WLASL_ROOT}",
        default=WLASL_ROOT,
    )
    parser.add_argument(
        "-sd",
        "--split_dir",
        type=str,
        help=f"Split directory if not {SPLIT_DIR}",
        default=SPLIT_DIR,
    )
    parser.add_argument(
        "-rd",
        "--raw_dir",
        type=str,
        help=f"Video directory if not {RAW_DIR}",
        default=RAW_DIR,
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        type=str,
        help=f"Output directory if not {LABELS_PATH}",
        default=LABELS_PATH,
    )
    parser.add_argument("-ve", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()

    root = Path(args.root)
    split_path = root / args.split_dir / f"{args.asl_split}.json"
    raw_dir = root / args.raw_dir
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    preprocess_split(
        split_path=split_path,
        raw_path=raw_dir,
        output_base=output_dir,
        verbose=args.verbose,
    )
