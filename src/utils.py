import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from types import ModuleType
from typing import Literal, get_args

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb

# locals
from src.run_types import RUNS_PATH, SRC_ROOT, StopEvent

################ GPU ###################


class gpu_manager:
    @classmethod
    def get_gpu_memory_usage(cls, gpu_id: int = 0) -> tuple[float, float]:
        """Get GPU usage across all processes

        Args:
            gpu_id (int, optional): ID of GPU. Defaults to 0.

        Raises:
            RuntimeError: nvidia-smi failed with subprocess
            ValueError: Unexpected nvidia-smi output
            ValueError: Failed to parse GPU memory usage:

        Returns:
            tuple[float, float]: used, total in GiB
        """

        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
        output = result.stdout.strip()
        parts = output.split(",")
        if len(parts) != 2:
            raise ValueError(f"Unexpected nvidia-smi output: '{output}'")
        try:
            used, total = map(float, parts)
        except Exception as e:
            raise ValueError(f"Failed to parse GPU memory usage: '{output}'") from e

        return used / 1024, total / 1024  # In GB

    @classmethod
    def output(cls, logger: Logger | None, message: str) -> None:
        if logger is None:
            print(message)
        else:
            logger.info(message)

    @classmethod
    def wait_for_completion(
        cls,
        check_interval: int = 3600,  # 1 hour
        confirm_interval: int = 60,  # 1 minute
        num_checks: int = 5,  # confirm consistency over 5 minutes
        gpu_id: int = 0,
        max_util_gb: float = 1.0,  # Maximum memory usage in GB
        logger: Logger | None = None,
        event: StopEvent | None = None,
    ) -> bool:
        """Wait for GPU memory to be free before proceeding

        Args:
                                                                        check_interval (int, optional): Wait period before checking again. Defaults to 3600 (1 hour).
                                                                        confirm_interval (int, optional): Wait period before checking again when confirming. Defaults to 60 (1 minute).
                                                                        num_checks: (int, optional): Confirm consistency over how many checks. Defaults to 5 (5 minutes).
                                                                        gpu_id (int, optional): CUDA GPU. Defaults to 0.
                                                                        max_util_gb (float, optional): Threshold GPU usage to trigger waiting. Defaults to 1.0 (GB).
                                                                        logger: (Logger, optional): Optionally supply a logger
                                                                        event: (StopEvent | None, optional): Optionally supply a stopping event.
        Returns:
                                                                        bool: Whether monitoring was killed by the user, either through CTRL+C or stop event.
        """
        assert torch.cuda.is_available(), "CUDA is not available"

        used, total = cls.get_gpu_memory_usage(gpu_id)

        proceed = used > max_util_gb

        while proceed:
            cls.output(logger, "")
            cls.output(
                logger,
                f"Monitoring GPU: {gpu_id}, current memory usage: {used:.2f}/{total:.2f} GB ({used / total * 100:.1f}%)",
            )
            cls.output(logger, f"Last checked at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            try:
                time.sleep(check_interval)
                used, _ = cls.get_gpu_memory_usage(gpu_id)
            except KeyboardInterrupt:
                cls.output(logger, "")
                cls.output(logger, "Monitoring interrupted by user")
                return False

            if used <= max_util_gb and cls._confirm_usage(
                confirm_interval, num_checks, gpu_id, max_util_gb
            ):
                break
            elif event is not None and event.is_set():
                cls.output(logger, "")
                cls.output(logger, "Stop event detected")
                return False

        cls.output(logger, "")
        cls.output(
            logger,
            f"GPU: {gpu_id} is available, current memory usage: {used:.2f}/{total:.2f} GB ({used / total * 100:.1f}%)",
        )

        return True

    @classmethod
    def _confirm_usage(
        cls,
        confirm_interval: int = 60,  # 1 minute
        num_checks: int = 5,  # confirm consistency over 5 minutes
        gpu_id: int = 0,
        max_util_gb: float = 1.0,
    ) -> bool:
        for _ in range(num_checks):
            used, _ = cls.get_gpu_memory_usage(gpu_id)
            if used > max_util_gb:
                return False
            time.sleep(confirm_interval)
        return True


############## wandb ##################
class wandb_manager:
    @classmethod
    def get_run_id(
        cls, run_name, entity: str, project: str, idx: int | None = None
    ) -> str | None:
        api = wandb.Api()

        runs = api.runs(f"{entity}/{project}")
        ids = []
        for run in runs:
            if run.name == run_name:
                ids.append(run.id)

        if len(ids) == 0:
            print(f"No runs found with name: {run_name}")
            return None
        elif len(ids) > 1:
            print(f"Multiple runs found with name: {run_name}")
            if isinstance(idx, int) and abs(idx) < len(runs):
                print(f"Returning id for idx: {idx}")
                return ids[idx]
            else:
                print("No idx supplied, returning None")
                return None
        else:
            return ids[0]

    @classmethod
    def list_runs(
        cls,
        entity: str,
        project: str,
        disp: bool = False,
    ) -> list:
        """
        list all runs for a given entity and project.

        Args:
                                        entity (str): The wandb entity.
                                        project (str): The wandb project.
                                        disp (bool, optional): If True, prints details of each run.

        Returns:
                                        list: A list of wandb run objects, not strings.
        """
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")

        if disp:
            for run in runs:
                print(f"Run ID: {run.id}")
                print(f"Run name: {run.name}")
                print(f"State: {run.state}")
                print(f"Created: {run.created_at}")
                print("---")

        return runs

    @classmethod
    def run_present(cls, run_id: str, runs: list) -> bool:
        return any(run.id == run_id for run in runs)

    @classmethod
    def validate_runId(cls, run_id: str, entity: str, project: str) -> bool:
        return cls.run_present(run_id, cls.list_runs(entity, project))

    @classmethod
    def open_wandb(
        cls,
        entity: str,
        project: str,
        run_id: str | None,
    ) -> None:
        """Open wandb page in default browser.
        Args:
                                        entity (str): wandb entity
                                        project (str): wandb project
                                        run_id (Optional[str]): wandb run id. If None, opens project page.
        """

        if run_id is None:
            url = f"https://wandb.ai/{entity}/{project}/"
        else:
            url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
        subprocess.run(["xdg-open", url], check=False)


################# Loading #####################
def load_module_from_path(path: Path, module_prefix: str = "_module") -> ModuleType:
    """Load an arbitrary .py file by path as a standalone module.

    This works regardless of where `path` lives on disk - it doesn't need to
    be on sys.path or part of any package.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    module_name = f"{module_prefix}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")

    module = importlib.util.module_from_spec(spec)
    # Registering in sys.modules first lets the module's own top-level code
    # (e.g. dataclasses, or anything doing `import module_name`) resolve
    # correctly, and avoids it being garbage-collected mid-exec.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_rgb_frames_from_video(
    video_path: str | Path, start: int, end: int, all: bool = False
) -> torch.Tensor:
    """Load frames from video into torch Tensor

    Args:
                    video_path (Union[str, Path]): Path to video file
                    start (int): Starting frame index (inclusive)
                    end (int): Ending frame index (exclusive)
                    all (bool, optional): If True, returns all frames. If False, returns frames from start to end. Defaults to False.

    Returns:
                    torch.Tensor: 4D tensor of shape (T, C, H, W) in RGB format
    """
    return cv_to_torch(cv_load(video_path, start, end, all))


def cv_load(
    video_path: str | Path, start: int, end: int, all: bool = False
) -> np.ndarray:
    """Load rgb frames from a video file

    Args:
                    video_path (Union[str, Path]): Path to video file
                    start (int): Starting frame index (inclusive)
                    end (int): Ending frame index (exclusive)
                    all (bool, optional): If True, returns all frames. If False, returns frames from start to end. Defaults to False.

    Raises:
                    FileNotFoundError: If the video file does not exist
                    ValueError: If no frames were loaded for the video file

    Returns:
                    np.ndarray: 4D array of shape (T, H, W, C) in BGR format (OpenCV format)
    """

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"File {video_path} does not exist")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if len(frames) > 0:
        if all:
            return np.asarray(frames)
        else:
            return np.asarray(frames[start:end])
    else:
        raise ValueError(f"No frames were loaded for file {video_path}")


def header_frame_count(video_path: str | Path) -> int:
    """A video's frame count as stored in its container header (`CAP_PROP_FRAME_COUNT`).

    Fast, as nothing is decoded, but the header can disagree with the number of frames
    `cv_load` actually returns -- use `decoded_frame_count` when that matters.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def decoded_frame_count(video_path: str | Path) -> int:
    """The number of frames OpenCV decodes from a video, i.e. `len(cv_load(..., all=True))`.

    Slow (every frame is decoded), but exact, unlike `header_frame_count`.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    return count


################## Saving #####################


def save_video(frames, path, fps=30):
    """Arguments:
    frames : numpy tensor (T H W C) BGR (cv format)
    path : output path (string)
    fps : int"""

    if len(frames.shape) != 4:
        raise ValueError(f"Expected 4D tensor (T,H,W,C), got shape {frames.shape}")

    # Ensure frames are uint8
    if frames.dtype != np.uint8:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    width = frames.shape[2]
    height = frames.shape[1]
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")

    for frame in frames:
        out.write(frame)
    out.release()


################## Displaying  #####################

################  CV based ####################


def watch_video(frames=None, path="", wait=33, title="Video"):
    if frames is None and not path:
        raise ValueError("pass either a tensor or path")
    elif frames is not None:
        if not (isinstance(frames, (torch.Tensor, np.ndarray))):
            raise ValueError("frames must be torch.Tensor or np.ndarray")
        if frames.dtype == torch.uint8:
            frames = torch_to_cv(frames)  # type: ignore
        elif frames.dtype != np.uint8:
            raise ValueError(
                "frames must be either torch.Tensor with dtype torch.uint8 and shape (T, C, H, W) in RGB format, or np.ndarray with dtype np.uint8 and shape (T, H, W, C) in BGR format"
            )
        for img in frames:
            cv2.imshow(f"{title} from tensor", img)  # type: ignore
            key = cv2.waitKey(wait) & 0xFF
            if key == ord("q") or key == 27:
                break
            cv2.imshow(f"{title} from tensor", img)
            key = cv2.waitKey(wait) & 0xFF
            if key == ord("q") or key == 27:
                break
    else:
        cap = cv2.VideoCapture(path)
        beg = True
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                print("no frames")
                if beg:
                    continue
                else:
                    print("finished")
                    break  # works for files, not well for webcam
            cv2.imshow(f"{title} from file", img)
            beg = False
            key = cv2.waitKey(wait) & 0xFF
            if key == ord("q") or key == 27:
                break
    cv2.destroyAllWindows()


def show_bbox(frames, bbox):
    # frames has shape (num_frames, channels, height, width)
    # bbox is a list of [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    print("Bounding box coordinates:", bbox)
    for i in range(frames.shape[0]):
        frame = (
            frames[i].permute(1, 2, 0).cpu().numpy()
        )  # Change to (height, width, channels)
        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (255, 0, 0), 2
        )  # Draw the bounding box
        cv2.imshow("Frame with Bounding Box", frame)
        cv2.waitKey(0)  # Display each frame for 100 ms
    cv2.destroyAllWindows()


def cv_display(frames: np.ndarray, output: str | Path | None = None):
    for i, frame in enumerate(frames):
        cv2.imshow("Cropped Frames", frame)  # Display the first frame
        cv2.waitKey(100)  # Wait for a key press to close the window
        if output is not None:
            cv2.imwrite(f"{output}/frame_{i:04d}.jpg", frame)
    cv2.destroyAllWindows()


def cv_save(frames: np.ndarray, output: str | Path):
    Path(output).mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(frames):
        cv2.imwrite(f"{output}/frame_{i:04d}.jpg", img)  # Save each frame as an image
    print(f"Cropped frames saved to {output}.")

    ################# Plot based #########################


################## MATPLOT based ###################

def plt_display(
    frames: torch.Tensor,
    num: int,
    size: tuple[float, float] = (5.0, 5.0),
    adapt: bool = False,
    output_dir: str | Path | None = None,
    file_suffix: str = ".png",
    dpi : float | None = None,
    interpolation: str = 'antialiased'
):
    """
    Visualise a subset of frames using matplotlib.

    Args:
        frames (torch.Tensor): 4D tensor of shape (T, C, H, W). Channels expected in RGB order.
        num (int): Number of frames to display (evenly sampled). Must be >= 1.
        size (tuple, optional): Figure size (width, height) in inches. Defaults to (5, 5).
        adapt (bool, optional): If True, scale `size` based on frame resolution. Defaults to False.
        output (str | Path | None, optional): If provided, directory to save each plotted frame as 'frame{i}.file_suffix'.

    Returns:
        None

    Raises:
        ValueError: If num < 1 or frames has incompatible shape.
    """
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = None

    if adapt:
        factor = 5 / 256
        w, h = frames.shape[2], frames.shape[3]
        size = (w * factor, h * factor)

    if num < 1:
        raise ValueError("num must be >= 1")

    num_frames = len(frames)
    if num_frames <= num:
        step = 1
    else:
        step = num_frames // num

    for i, frame in enumerate(frames[::step]):
        # Permute from (C, H, W) to (H, W, C)
        np_frame = frame.permute(1, 2, 0).cpu().numpy()

        # Normalize to [0, 1] range
        np_frame = (np_frame - np_frame.min()) / (np_frame.max() - np_frame.min())

        plt.figure(figsize=size,dpi=dpi,)
        plt.imshow(np_frame, interpolation=interpolation)
        plt.axis("off")
        if output:
            plt.savefig(
                (output / f"frame{i}").with_suffix(file_suffix),
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.show()


def plt_display_grid(
    frames: torch.Tensor,
    num: int,
    size: tuple[float, float] = (5.0, 5.0),
    adapt: bool = False,
    output: str | Path | None = None,
    cols: int = 8,
):
    """
    Visualise a subset of frames in a grid layout using matplotlib.

    Args:
        frames (torch.Tensor): 4D tensor of shape (T, C, H, W). Channels expected in RGB order.
        num (int): Number of frames to display (evenly sampled). Must be >= 1.
        size (tuple, optional): Size (width, height) in inches per cell. Defaults to (5, 5).
        adapt (bool, optional): If True, scale `size` based on frame resolution. Defaults to False.
        output (str | Path, optional): If provided, saves the grid figure as output.
        cols (int, optional): Maximum number of frames per row. Defaults to 8.

    Returns:
        None

    Raises:
        ValueError: If num < 1 or frames has incompatible shape.
    """
    if num < 1:
        raise ValueError("num must be >= 1")

    if adapt:
        factor = 5 / 256
        w, h = frames.shape[2], frames.shape[3]
        size = (w * factor, h * factor)

    num_frames = len(frames)
    step = 1 if num_frames <= num else num_frames // num
    sampled = frames[::step][:num]

    rows = math.ceil(len(sampled) / cols)
    _fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(size[0] * cols, size[1] * rows),
        squeeze=False,
    )

    for i, frame in enumerate(sampled):
        np_frame = frame.permute(1, 2, 0).cpu().numpy()
        np_frame = (np_frame - np_frame.min()) / (np_frame.max() - np_frame.min())
        ax = axes[i // cols][i % cols]
        ax.imshow(np_frame)
        ax.axis("off")

    # Hide any unused cells in the last row
    for j in range(len(sampled), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    if output:
        plt.savefig(output, bbox_inches="tight", pad_inches=0)

    plt.show()


################### Conversions #####################


def torch_to_cv(frames: torch.Tensor) -> np.ndarray:
    """convert 4D torch tensor (T C H W) uint8 to opencv format"""
    frames = frames.permute(0, 2, 3, 1)  # change to T H W C
    np_frames = frames.numpy()
    np_frames_col = np.asarray(
        [cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR) for np_frame in np_frames]
    )
    return np_frames_col


def torch_to_mediapipe(frames: torch.Tensor) -> np.ndarray:
    """convert 4D torch.uint8 (T C H W) to mediapipe format"""
    frames = frames.permute(0, 2, 3, 1)  # convert to (T H W C)
    np_array = frames.numpy()
    return np_array


def cv_to_torch(frames):
    """convert 4D opencvformat to torch Tensor (T C H W)"""
    frames_rgb = np.asarray(
        [cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB) for cv_frame in frames]
    )
    torch_frames = torch.from_numpy(frames_rgb)
    torch_frames = torch_frames.permute(0, 3, 1, 2)
    return torch_frames


####################     Plotting utilities  ####################


def plot_from_lists(
    train_loss,
    val_loss=None,
    title="Training Loss Curve",
    xlabel="Epochs",
    ylabel="Loss",
    save_path=None,
    show=True,
):
    # Extract the three components from training loss
    # wall_times = [point[0] for point in train_loss]
    # steps = [point[1] for point in train_loss]
    # values = [point[2] for point in train_loss]
    steps = list(range(len(train_loss)))
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(steps, train_loss, color="orange", linewidth=2, label="Training Loss")

    # Plot validation loss if provided
    if val_loss:
        # val_steps = [point[1] for point in val_loss]
        # val_values = [point[2] for point in val_loss]

        plt.plot(steps, val_loss, color="blue", linewidth=2, label="Validation Loss")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()


##################### Misc ###################################


def enum_dir(path: str | Path, make: bool = False, decimals: int = 3) -> Path:
    """Enumerate until the next available directory name

    Args:
        path (str | Path): Directory path
        make (bool, optional): Make the new directory. Defaults to False.
        decimals (int, optional): Number of digits to zfill by. Defaults to 3.

    Returns:
        Path: A unique path. If the path provided doesn't exist, it returns unchanged.
        Otherwise, it returns with zfilled number appended
    Example:
    >>> unique_dir_name = 'new_dir'
    >>> print(enum_dir(unique_dir_name, make=True))
    new_dir
    >>> print(enum_dir(unique_dir_name))
    new_dir000


    """
    path = Path(path)
    path_str = str(path)
    if path.exists():
        if not path_str[-1].isdigit():
            path_str += "0".zfill(decimals)
        path = Path(path_str)
        while path.exists():
            try: 
                num = int(path_str[-decimals:])
            except ValueError:
                raise ValueError(f"Failed to parse number from path: {path_str}")
            path_str = path_str[:-decimals] + str(num + 1).zfill(decimals)
            path = Path(path_str)
    if make:
        path.mkdir(parents=True, exist_ok=True)
    return path


##################### Runs cleaning ###################################

CleanReason = Literal["resume checkpoint", "sweep trial outside top-k", "abandoned run"]

CHECKPOINT_DIR_RE = re.compile(r"checkpoints\d*")
BEST_WEIGHTS = "best.pth"
DATA_INFO = "data_info.json"
SWEEP_DIR_PREFIX = "sweep_"
REFERENCE_SUFFIXES = (".py", ".ipynb", ".toml", ".md")


@dataclass(frozen=True)
class Removal:
    """A file or directory the runs cleaner would delete, why, and its size in bytes."""

    path: Path
    reason: CleanReason
    size: int


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _removal(path: Path, reason: CleanReason) -> Removal:
    return Removal(path, reason, _size(path))


def checkpoint_dirs(run_dir: Path) -> list[Path]:
    """A run's `checkpoints`/`checkpointsNNN` directories (several if the run was re-run)."""
    return sorted(
        p
        for p in run_dir.iterdir()
        if p.is_dir() and CHECKPOINT_DIR_RE.fullmatch(p.name)
    )


def find_run_dirs(runs_dir: Path) -> list[Path]:
    """Every run directory under `runs_dir`: one holding a checkpoints directory."""
    return sorted(
        {
            p.parent
            for p in runs_dir.rglob("checkpoints*")
            if p.is_dir() and CHECKPOINT_DIR_RE.fullmatch(p.name)
        }
    )


def resume_checkpoint_removals(run_dir: Path) -> list[Removal]:
    """The run's resume checkpoints (`checkpoint_NNN.pth`), wherever `best.pth` sits beside them.

    A resume checkpoint adds the optimizer, scheduler and RNG state to the weights (about 3x the
    size of `best.pth`), which only matters for resuming the run; testing, ensembling and
    initialising new runs (`training.load_pretrained`) all use `best.pth`. Checkpoint directories
    without a `best.pth` are left alone: in runs predating it, the last checkpoint is the run's
    only copy of its weights.
    """
    return [
        _removal(f, "resume checkpoint")
        for d in checkpoint_dirs(run_dir)
        if (d / BEST_WEIGHTS).is_file()
        for f in sorted(d.glob("checkpoint_*.pth"))
    ]


def abandoned_run_removals(run_dir: Path) -> list[Removal]:
    """The whole run directory, if it holds no files but `data_info.json` (it never trained)."""
    if all(f.name == DATA_INFO for f in run_dir.rglob("*") if f.is_file()):
        return [_removal(run_dir, "abandoned run")]
    return []


def best_val_loss(run_dir: Path) -> float | None:
    """Lowest best validation loss over the run's `results*/best_val_loss.json`, if tested."""
    losses = [
        float(json.loads(p.read_text())["best_val_loss"])
        for p in run_dir.glob("results*/best_val_loss.json")
    ]
    return min(losses, default=None)


def sweep_weight_removals(
    sweep_dir: Path, keep_top: int, keep_ids: set[str]
) -> list[Removal]:
    """Weights (`*.pth`) of a sweep's trials, except its `keep_top` best and those in `keep_ids`.

    Trials are ranked by best validation loss, as `results/sweeping/ensemble.py` picks its
    members; untested trials rank last. Only weights are removed, so each trial's results stay.
    """
    trials = [d for d in sweep_dir.iterdir() if d.is_dir()]
    losses = {t: best_val_loss(t) for t in trials}
    ranked = sorted(trials, key=lambda t: (losses[t] is None, losses[t] or 0.0))
    return [
        _removal(f, "sweep trial outside top-k")
        for t in ranked[keep_top:]
        if t.name not in keep_ids
        for f in sorted(t.rglob("*.pth"))
    ]


def referenced_run_ids(run_ids: set[str], search_dir: Path = SRC_ROOT) -> set[str]:
    """The `run_ids` named in any code, notebook, config or markdown file under `search_dir`.

    JSON is not searched: sweep stashes list every trial, so would protect them all.
    """
    found: set[str] = set()
    for f in search_dir.rglob("*"):
        if f.suffix in REFERENCE_SUFFIXES and f.is_file():
            words = re.findall(r"[a-z0-9]+", f.read_text(errors="ignore"))
            found |= run_ids & set(words)
    return found


def queued_run_dirs() -> set[Path]:
    """Run directories of the Que's queued and running runs, which must not be cleaned."""
    from src.que.core import Que

    que = Que(auto_save=False)
    return {Path(r.admin.save_path).parent for r in [*que.to_run, *que.cur_run]}


def plan_cleanup(
    runs_dir: Path,
    reasons: set[CleanReason],
    keep_top: int,
    keep_ids: set[str],
    protected: set[Path],
) -> list[Removal]:
    """Everything the cleaner would delete under `runs_dir`, for the strategies in `reasons`.

    Args:
        runs_dir: Base runs directory (normally `RUNS_PATH`).
        reasons: Which strategies to apply.
        keep_top: Trials per sweep whose weights are kept (see `sweep_weight_removals`).
        keep_ids: Sweep trial ids whose weights are kept regardless of rank (e.g. those from
            `referenced_run_ids`).
        protected: Run directories never cleaned (e.g. `queued_run_dirs()`); a sweep holding
            one is skipped entirely.

    Returns:
        list[Removal]: One entry per path, so a file is never counted twice.
    """
    run_dirs = [
        r
        for r in find_run_dirs(runs_dir)
        if r.resolve() not in {p.resolve() for p in protected}
    ]
    removals: dict[Path, Removal] = {}

    def add(found: list[Removal]) -> None:
        for r in found:
            removals.setdefault(r.path, r)

    for run_dir in run_dirs:
        if "abandoned run" in reasons:
            add(abandoned_run_removals(run_dir))
        if "resume checkpoint" in reasons:
            add(resume_checkpoint_removals(run_dir))
    if "sweep trial outside top-k" in reasons:
        protected_sweeps = {p.resolve().parent for p in protected}
        for sweep_dir in sorted(
            {r.parent for r in run_dirs if r.parent.name.startswith(SWEEP_DIR_PREFIX)}
        ):
            if sweep_dir.resolve() not in protected_sweeps:
                add(sweep_weight_removals(sweep_dir, keep_top, keep_ids))
    return list(removals.values())


def sweep_trial_ids(runs_dir: Path) -> set[str]:
    """The ids (directory names) of every sweep trial under `runs_dir`."""
    return {
        r.name
        for r in find_run_dirs(runs_dir)
        if r.parent.name.startswith(SWEEP_DIR_PREFIX)
    }


def summarise_removals(
    removals: list[Removal], runs_dir: Path, verbose: bool = False
) -> str:
    """Count and size of the removals per reason (and each path, if `verbose`)."""
    lines: list[str] = []
    for reason in get_args(CleanReason):
        group = [r for r in removals if r.reason == reason]
        if verbose:
            lines += [
                f"  {r.path.relative_to(runs_dir)} ({r.size / 1e6:.0f} MB)"
                for r in group
            ]
        lines.append(
            f"{reason}: {len(group)} paths, {sum(r.size for r in group) / 1e9:.1f} GB"
        )
    lines.append(f"total: {sum(r.size for r in removals) / 1e9:.1f} GB")
    return "\n".join(lines)


def apply_removals(removals: list[Removal]) -> None:
    """Delete each removal's path (directories with their contents)."""
    for r in removals:
        if r.path.is_dir():
            shutil.rmtree(r.path)
        else:
            r.path.unlink(missing_ok=True)


def main() -> None:
    reason_flags: dict[str, CleanReason] = {
        "resume": "resume checkpoint",
        "sweeps": "sweep trial outside top-k",
        "abandoned": "abandoned run",
    }
    parser = ArgumentParser(
        description="Free space in the runs directory. Dry run unless --delete is given."
    )
    parser.add_argument("--runs_dir", type=Path, default=RUNS_PATH)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(reason_flags),
        help="Strategies to apply (default: all)",
    )
    parser.add_argument(
        "-k",
        "--keep_top",
        type=int,
        default=10,
        help="Trials per sweep to keep weights for",
    )
    parser.add_argument(
        "--keep",
        nargs="+",
        default=[],
        help="Sweep trial ids to keep weights for, besides those named in code",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="List every path")
    parser.add_argument("--delete", action="store_true", help="Actually delete")
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Delete without confirming"
    )
    args = parser.parse_args()

    reasons: set[CleanReason] = {reason_flags[f] for f in (args.only or reason_flags)}
    keep_ids = set(args.keep) | referenced_run_ids(sweep_trial_ids(args.runs_dir))
    removals = plan_cleanup(
        args.runs_dir, reasons, args.keep_top, keep_ids, queued_run_dirs()
    )
    print(summarise_removals(removals, args.runs_dir, args.verbose))
    if not args.delete or not removals:
        return
    if args.yes or input("Delete all of the above? [y/N]: ").strip().lower() == "y":
        apply_removals(removals)
        print("Deleted.")


if __name__ == "__main__":
    main()
