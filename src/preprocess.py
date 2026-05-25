from typing import (
    List,
    Literal,
    Optional,
    Any,
    Tuple,
    TypeGuard,
    Union,
)
from pydantic import BaseModel, TypeAdapter
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

# local imports
from src.utils import load_rgb_frames_from_video
from src.configs import LABELS_PATH
from src.run_types import AVAIL_SETS, WLASL_ROOT, SPLIT_DIR, RAW_DIR


"""Naming convention:
- set: one of train, test and val
- split: one of asl100, asl300, asl1000, asl2000"""

class RawInstance(BaseModel):
    """Represents a single raw instance of a gloss in the dataset."""
    bbox: List[int]  # [x_min, y_min, x_max, y_max]
    frame_end: int
    frame_start: int
    instance_id: int
    signer_id: int
    source: str
    split: str
    url: str
    variation_id: int
    video_id: str


class Instance(RawInstance):
    """Represents a single instance of a gloss in the dataset, with the label_num and label_name added."""
    label_num: int
    label_name: str


class WLASLClass(BaseModel):
    """Represents a single gloss and its associated raw instances."""
    gloss: str
    instances: List[RawInstance]


class BadInstance(Instance):
    """Adds reason to an instance that was discarded/modified from the dataset"""
    reason: str


class ErrLog(BaseModel):
    """Format for storing bad instances"""
    policy: str
    num_offenders: int
    instances: List[BadInstance]


def is_processed_instance(obj: Any) -> TypeGuard[Instance]:
    """Type guard to check if an object is a valid Instance dict/object."""
    try:
        Instance.model_validate(obj)
        return True
    except Exception:
        return False


def instance_to_processed(d: RawInstance, label_num: int, label_name: str) -> Instance:
    """Convert a RawInstance to a Instance by adding labels."""
    return Instance(
        **d.model_dump(),
        label_num=label_num,
        label_name=label_name,
    )


def processed_to_bad(d: Instance, reason: str) -> BadInstance:
    """Convert a Instance to a BadInstance by adding a reason."""
    return BadInstance(**d.model_dump(), reason=reason)


def get_set(
    lst_wlasl_class_dicts: List[WLASLClass], set_name: AVAIL_SETS
) -> List[Instance]:
    """Filters list of WLASLClass based on whether the instances are from the provided set_name."""
    mod_instances = []
    for i, gloss_d in enumerate(lst_wlasl_class_dicts):
        for inst in gloss_d.instances:
            if inst.split == set_name:
                mod_instances.append(instance_to_processed(inst, i, gloss_d.gloss))
    return mod_instances


def output_bad(
    bad_instances: List[BadInstance],
    remove_policy: str,
    log_path: Union[str, Path],
    fixing_description: str,
) -> None:
    """Output offending instances to a file using Pydantic's JSON serialization."""
    if len(bad_instances) != 0:
        err_dict = ErrLog(
            policy=remove_policy,
            num_offenders=len(bad_instances),
            instances=bad_instances,
        )
        with open(log_path, "w") as log_file:
            # use model_dump_json to serialize safely
            log_file.write(err_dict.model_dump_json(indent=4))
        print(f"Bad {fixing_description} logged to {log_path}.")
    else:
        print(f"No {fixing_description} ranges found")


def fix_bad_frame_range(
    raw_path: Path,
    instances: List[Instance],
    log_dir: Path,
    remove_policy: Literal["strict", "reset"] = "strict",
    file_extension: str = "bad_frame_ranges.json",
) -> List[Instance]:
    """Remove videos where the file cannot be read, or the start or end frame are impossible."""
    bad_instances: List[BadInstance] = []
    clean_instances: List[Instance] = []
    
    for instance in tqdm.tqdm(instances, desc="fixing frame ranges"):
        vid_path = raw_path / f"{instance.video_id}.mp4"

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            message = f"Could not open video {instance.video_id}. Removing"
            bad_instances.append(processed_to_bad(instance, message))
            continue
        else:
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start = instance.frame_start
        end = instance.frame_end
        
        if start < 0 or start >= num_frames:
            message = f"Invalid start frame {start} for video {instance.video_id} with length {num_frames}."
            if remove_policy == "strict":
                bad_instances.append(processed_to_bad(instance, message + " Removing instance."))
                continue
            elif remove_policy == "reset_frames":
                bad_instances.append(processed_to_bad(instance, message + " Setting to 0."))
            start = 0
            
        if end <= start or end > (start + num_frames):
            message = f"Invalid end frame {end} for video {instance.video_id} with length {num_frames} and start frame {start}."
            if remove_policy == "strict":
                bad_instances.append(processed_to_bad(instance, message + " Removing instance."))
                continue
            elif remove_policy == "reset_frames":
                bad_instances.append(processed_to_bad(instance, message + " Setting to num_frames."))
            end = start + num_frames
            
        instance.frame_start = start
        instance.frame_end = end
        clean_instances.append(instance)

    log_path = log_dir / f"{remove_policy}_{file_extension}"
    output_bad(
        bad_instances=bad_instances,
        remove_policy=remove_policy,
        log_path=log_path,
        fixing_description="frame range",
    )

    return clean_instances


def get_largest_bbox(bboxes: List[List[float]]) -> Optional[List[float]]:
    """Given a list of bounding boxes, returns the largest bounding box that encompasses all of them, if one exists."""
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
    instances: List[Instance],
    log_dir: Path,
    remove_policy: Literal["strict", "reset"] = "strict",
    file_extension: str = "bad_bboxes.json",
) -> List[Instance]:
    """Fix bad bounding boxes by running a pre-trained YOLOv8 model on the video."""
    model = YOLO("yolov8n.pt")  # Load a pre-trained YOLO model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bad_instances: List[BadInstance] = []
    clean_instances: List[Instance] = []

    for instance in tqdm.tqdm(instances, desc="Fixing bounding boxes"):
        vid_path = raw_path / f"{instance.video_id}.mp4"
        frames = load_rgb_frames_from_video(
            str(vid_path), instance.frame_start, instance.frame_end
        )
        frames = frames.float() / 255.0

        results = model(frames, device=device, verbose=False)
        bboxes = []
        for result in results:
            person_bboxes = result.boxes.xyxy[result.boxes.cls == 0]
            if len(person_bboxes) > 0:
                bboxes.extend(person_bboxes.tolist())

        if not bboxes:
            message = f"No bounding boxes found for video {instance.video_id}."
            if remove_policy == "strict":
                bad_instances.append(processed_to_bad(instance, message + " Removing instance."))
                continue
            elif remove_policy == "reset_bbox":
                bad_instances.append(processed_to_bad(instance, message + " Using whole frame."))
                largest_bbox = [0, 0, frames.shape[3], frames.shape[2]]
            else:
                raise ValueError(f"Invalid remove_policy: {remove_policy}")
        else:
            largest_bbox = get_largest_bbox(bboxes)
            assert largest_bbox is not None, "largest_bbox can not be None here"

        # Round the coordinates to integers and update the Pydantic model
        largest_bbox = [round(coord) for coord in largest_bbox] 
        instance.bbox = largest_bbox
        clean_instances.append(instance)

    log_path = log_dir / f"{remove_policy}_{file_extension}"

    output_bad(
        bad_instances=bad_instances,
        remove_policy=remove_policy,
        log_path=log_path,
        fixing_description="bounding boxes",
    )

    return clean_instances


def remove_short_samples(
    instances: List[Instance],
    log_dir: Path,
    cutoff: int = 9,
    file_extension: str = "removed_short_samples.json",
) -> List[Instance]:
    """Remove samples where the number of frames is less than or equal to the cutoff."""
    clean_instances = []
    short_samples = []
    
    for inst in instances:
        num_frame = inst.frame_end - inst.frame_start
        if num_frame > cutoff:
            clean_instances.append(inst)
        else:
            # Fixed bug: Append a BadInstance instead of a raw string
            short_samples.append(
                processed_to_bad(inst, f"bad number of frames {num_frame} for video {inst.video_id}, removing.")
            )

    log_path = log_dir / f"cutoff_{cutoff}_{file_extension}"

    output_bad(
        bad_instances=short_samples,
        remove_policy="strict",
        log_path=log_path,
        fixing_description="short samples",
    )

    return clean_instances


def print_v(s: str, y: bool) -> None:
    if y:
        print(s)


def check_paths(
    split_path: Path, raw_path: Path, output_path: Path, verbose: bool
) -> bool:
    """Checks if the provided paths exist and are of the correct type."""
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
    file_extension: str = "_fixed_frange_bboxes_len.json",
    strictness: Tuple[Literal['strict', 'reset'], Literal['strict', 'reset']] = ('strict', 'strict'),
    do_bboxes: bool = True,
    length_cuttoff: int = 9
) -> None:
    """Preprocesses a split of the WLASL dataset."""

    if not check_paths(split_path, raw_path, output_base, verbose):
        return

    with open(split_path, "r") as f:
        raw_json_data = json.load(f)

    if not raw_json_data:
        print(f"no data found in {split_path}")
        return

    # Use Pydantic TypeAdapter to validate the incoming JSON dynamically 
    wlasl_adapter = TypeAdapter(List[WLASLClass])
    asl_num = wlasl_adapter.validate_python(raw_json_data)

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
        print_v("Fixing frame ranges", verbose)
        instances = fix_bad_frame_range(
            raw_path=raw_path,
            instances=instances,
            log_dir=output_dir,
            remove_policy=strictness[0],
            file_extension=f"bad_frame_ranges_{subset}.json",
        )

        if do_bboxes:
            print_v("Fixing bounding boxes", verbose)
            instances = fix_bad_bboxes(
                raw_path=raw_path,
                instances=instances,
                log_dir=output_dir,
                remove_policy=strictness[1],
                file_extension=f"bad_bboxes_{subset}.json",
            )

        print_v("Removing small samples", verbose)
        instances = remove_short_samples(
            instances=instances,
            log_dir=output_dir,
            cutoff=length_cuttoff,
            file_extension=f"removed_short_samples_{subset}.json",
        )

        print_v("Saving results", verbose)
        inst_path = output_dir / f"{subset}_instances{file_extension}"
        with open(inst_path, "w") as f:
            # Serialize back to JSON list using model_dump
            json.dump([inst.model_dump() for inst in instances], f, indent=2)

    print("\n------------------------- finished preprocessing ---------------\n")


if __name__ == "__main__":
    avail_splits = ["asl100", "asl300", "asl1000", "asl2000"]

    parser = ArgumentParser(description="preprocess.py")
    parser.add_argument(
        "asl_split",
        type=str,
        choices=avail_splits + ['all'],
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
    parser.add_argument('-ss', '--strictness', nargs=2, choices=['strict', 'reset'], default=['reset', 'reset'], help='The strictness levels for frame range, and bounding boxes respectively. Reset takes the full video/frame. Strict disgards. Both log.')
    parser.add_argument('-nb', '--no_bbox', action='store_true', help='Skip intense bbox step')
    parser.add_argument('-lc', '--length_cutoff', type=int, default=9, help='Minimum number of frames for a sample to be kept.')
    args = parser.parse_args()

    root = Path(args.root)
    raw_dir = root / args.raw_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.asl_split == 'all':
        todo_splits = avail_splits
    else:
        todo_splits = [args.asl_split]

    for split in todo_splits:
        split_path = root / args.split_dir / f"{split}.json"
        preprocess_split(
            split_path=split_path,
            raw_path=raw_dir, 
            output_base=output_dir,
            verbose=args.verbose,
            strictness=tuple(args.strictness),
            do_bboxes=(not args.no_bbox),
            length_cuttoff=args.length_cutoff,
        )
