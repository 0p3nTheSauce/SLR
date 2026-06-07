from sklearn.metrics import confusion_matrix
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional, Callable, List, Union, Any
from pathlib import Path
import logging
from logging import Logger
from torch.utils.data import DataLoader, Dataset
from typing_extensions import TypedDict, Unpack
from torch import Tensor
# locals
from src.run_types import CentreCropConfig, OG_Sampler
from src.configs import get_class_list, CLASSES_PATH
from src.video_dataset import (
    get_wlasl_info,
    get_labels_path,
    load_data_from_json,
    get_video_path,
    get_transform,
    
)
from src.utils import plt_display_grid, load_rgb_frames_from_video
from src.preprocess import Instance
from src.stats import (
    AVAIL_SETS,
    AVAIL_SPLITS,
    reverse_preproc_format,
)

# Set style for better-looking plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


VERBOSITY = 0


visualise_logger = logging.getLogger(__name__)

BG_FIGSIZE_3G=(8,5)
BG_WiDTH_3G=0.2

def set_font_size(size: int = 14) -> None:
    """Set the font size for all matplotlib plots."""
    plt.rcParams.update({"font.size": size})

set_font_size()  # Set default font size



def get_all_sets(
    split_name: AVAIL_SPLITS,
    set_options: List[AVAIL_SETS] = ["train", "test", "val"], #type: ignore
    classes: List[str] = get_class_list(),
    logger: Logger = visualise_logger
) -> dict:
    all_sets = {}
    for set_name in set_options:
        set_path_info = get_wlasl_info(split_name, set_name)
        set_path = get_labels_path(
            set_name, set_path_info["labels"], set_path_info["label_suff"]
        )
        all_sets[set_name] = reverse_preproc_format(
            load_data_from_json(set_path, policy="strict"), classes
        )
        logger.debug(f"Num classes of {set_name}: {len(all_sets[set_name])}")

    return all_sets


class MiniSetKwargsRequired(TypedDict):
    cls_idx: int
    all_sets: dict[str, Any]
    set_name: AVAIL_SETS
    split_name: AVAIL_SPLITS

class MiniSetKwargs(MiniSetKwargsRequired, total=False):
    classes: List[str]
    target_length: int
    frame_size: int
    logger: Logger


class MiniSet(Dataset):
    def __init__(
        self,
        cls_idx: int,
        all_sets: dict[str, Any],
        set_name: AVAIL_SETS,
        split_name: AVAIL_SPLITS,
        classes: List[str] = get_class_list(),
        target_length: int = 16,
        frame_size: int = 224,
        logger: Logger = visualise_logger,
        transform: Optional[Callable[[Tensor], Tensor]] = None
    ) -> None:

        self.logger = logger
        self.cls_idx = cls_idx
        self.set_name = set_name
        self.split_name = split_name
        self.classes = classes
        self.target_length = target_length
        self.frame_size = frame_size

        if transform is None:
            self.transform, _, _ = get_transform(
                temporal_aug=[OG_Sampler(target_length=target_length)],
                spatial_aug=[CentreCropConfig(frame_size=frame_size)],
                normalise_to_float=False,
                permute_time_channel=False,
            )
        else:
            self.transform = transform

        
        
        self.set_path_info = get_wlasl_info(split_name, set_name)
        self.data = all_sets[self.set_name][self.cls_idx]["instances"]
        self.tot_samples = len(self.data)


    def __getitem__(self, idx):
        self.logger.info(f"From: {self.split_name}S/{self.set_name}")
        self.logger.info(f'Example videos for class: "{self.classes[self.cls_idx]}"')
        self.logger.info(f"Instance: {idx + 1}/{self.tot_samples}")
        
        next_example = Instance.model_validate(self.data[idx])
        ex_path = get_video_path(next_example.video_id, self.set_path_info["root"])

        self.logger.info(f"Next example video path: {ex_path}")


        return self.transform(
            load_rgb_frames_from_video(
                ex_path, next_example.frame_start, next_example.frame_end
            )
        )
            
    def __len__(self):
        return self.tot_samples


class FrameVisualiser:
    def __init__(
        self,
        **kwargs: Unpack[MiniSetKwargs]
    ):
        if 'target_length' in kwargs:
            self.target_frames = kwargs['target_length']
        else:
            self.target_frames = 16
        self.iter_loader = iter(DataLoader(
            MiniSet(**kwargs), 
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=False
        ))
        
    def __call__(self):
        frames = next(self.iter_loader)[0]
        if len(frames.shape) == 5:
            frames = frames.squeeze(dim=0)
        if frames.shape[1] != 3: 
            frames = frames.permute(1, 0, 2, 3) #swap T and C
        plt_display_grid(frames, self.target_frames)
        


# results

# Simple plotting utilities for visualising testing results


def plot_heatmap(
    report: Dict[str, Dict[str, float]],
    classes_path: Union[str, Path] = CLASSES_PATH,
    title: str = "Classification Report Heatmap",
    save_path: Optional[Union[str, Path]] = None,
    disp: bool = True,
) -> None:
    """Plot a heatmap visualization of a classification report.

    Creates a seaborn heatmap showing precision, recall, and F1-score metrics
    for each class in the classification report.

    Args:
            report (Dict[str, Dict[str, float]]): Classification report dictionary,
                    typically from sklearn.metrics.classification_report with output_dict=True.
            classes_path (Union[str, Path]): Path to JSON file containing list of class names.
            title (str, optional): Title for the heatmap plot.
                    Defaults to "Classification Report Heatmap".
            save_path (Optional[Union[str, Path]], optional): Path to save the figure.
                    If None, figure is not saved. Defaults to None.
            disp (bool, optional): Whether to display the plot with plt.show().
                    Defaults to True.

    Returns:
            None

    Example:
            >>> from sklearn.metrics import classification_report
            >>> report = classification_report(y_true, y_pred, output_dict=True)
            >>> plot_heatmap(report, "classes.json", save_path="heatmap.png")
    """
    with open(classes_path, "r") as f:
        test_classes: List[str] = json.load(f)

    df = pd.DataFrame(report).iloc[:-1, :].T
    num_classes_to_plot = min(len(df) - 2, len(test_classes))

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        df.iloc[:num_classes_to_plot, :3],
        annot=True,
        cmap="Blues",
        fmt=".2f",
        xticklabels=["Precision", "Recall", "F1-Score"],
        yticklabels=[test_classes[i] for i in range(num_classes_to_plot)],
    )
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if disp:
        plt.show()


def plot_bar_graph(
    report: Dict[str, Dict[str, float]],
    classes_path: Union[str, Path] = CLASSES_PATH,
    title: str = "Classification Report - Per Class Metrics",
    save_path: Optional[Union[str, Path]] = None,
    disp: bool = True,
) -> None:
    """Plot a horizontal bar graph of classification metrics per class.

    Creates a bar plot showing precision, recall, and F1-score for each class
    as horizontal bars with different colors.

    Args:
            report (Dict[str, Dict[str, float]]): Classification report dictionary,
                    typically from sklearn.metrics.classification_report with output_dict=True.
            classes_path (Union[str, Path]): Path to JSON file containing list of class names.
            title (str, optional): Title for the bar graph.
                    Defaults to "Classification Report - Per Class Metrics".
            save_path (Optional[Union[str, Path]], optional): Path to save the figure.
                    If None, figure is not saved. Defaults to None.
            disp (bool, optional): Whether to display the plot with plt.show().
                    Defaults to True.

    Returns:
            None

    Example:
            >>> from sklearn.metrics import classification_report
            >>> report = classification_report(y_true, y_pred, output_dict=True)
            >>> plot_bar_graph(report, "classes.json", save_path="bar_graph.png")
    """
    with open(classes_path, "r") as f:
        test_classes: List[str] = json.load(f)

    classes = list(report.keys())[
        :-3
    ]  # Exclude 'accuracy', 'macro avg', 'weighted avg'

    # Prepare data for plotting
    precision = [report[cls]["precision"] for cls in classes]
    recall = [report[cls]["recall"] for cls in classes]
    f1_score = [report[cls]["f1-score"] for cls in classes]

    # Create bar plot
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 18))
    _ = ax.barh(x - width, precision, height=width, label="Precision", alpha=0.8)
    _ = ax.barh(x, recall, height=width, label="Recall", alpha=0.8)
    _ = ax.barh(x + width, f1_score, height=width, label="F1-Score", alpha=0.8)

    ax.set_ylabel("Classes")
    ax.set_xlabel("Scores")
    ax.set_title(title)
    ax.set_yticks(x)

    # Fix: Only use as many class names as we have classes in the report
    class_labels = [
        test_classes[int(cls)] if int(cls) < len(test_classes) else f"Class_{cls}"
        for cls in classes
    ]
    ax.set_yticklabels(class_labels)

    ax.legend()
    ax.set_xlim(0, 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if disp:
        plt.show()


def plot_confusion_matrix(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    classes_path: Optional[Union[str, Path]] = CLASSES_PATH,
    num_classes: int = 100,
    title: str = "Confusion Matrix",
    size: tuple[int, int] = (10, 8),
    row_perc: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    disp: bool = True,
) -> None:
    """Plot confusion matrix from true and predicted labels.

    Creates a heatmap visualization of the confusion matrix, optionally normalized
    by row (true class) to show percentage distributions.

    Args:
            y_true (Union[np.ndarray, List[int]]): Array-like of true labels.
            y_pred (Union[np.ndarray, List[int]]): Array-like of predicted labels.
            classes_path (Optional[Union[str, Path]], optional): Path to JSON file
                    containing list of class names. If None, numeric labels are used.
                    Defaults to None.
            num_classes (int, optional): Number of classes to display in the matrix.
                    Defaults to 100.
            title (str, optional): Title for the confusion matrix plot.
                    Defaults to "Confusion Matrix".
            size (tuple[int, int], optional): Figure size as (width, height) in inches.
                    Defaults to (10, 8).
            row_perc (bool, optional): If True, normalize each row to show percentages.
                    Defaults to True.
            save_path (Optional[Union[str, Path]], optional): Path to save the figure.
                    If None, figure is not saved. Defaults to None.
            disp (bool, optional): Whether to display the plot with plt.show().
                    Defaults to True.

    Returns:
            None

    Example:
            >>> plot_confusion_matrix(
            ...     y_true, y_pred,
            ...     classes_path="classes.json",
            ...     num_classes=50,
            ...     save_path="confusion_matrix.png"
            ... )
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Load class names if provided
    class_names: Optional[List[str]] = None
    if classes_path is not None and num_classes:
        with open(classes_path, "r") as f:
            test_classes: List[str] = json.load(f)

        class_names = test_classes[:num_classes]

    if row_perc:
        cm_row_percent = cm / cm.sum(axis=1, keepdims=True) * 100  # Normalize each row
        cm_row_percent = np.nan_to_num(cm_row_percent).round(
            2
        )  # Handle division by zero
        cm = cm_row_percent
        title += " rowise normalised"

    plt.figure(figsize=size)
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,  # Add gridlines between cells
        linecolor="gray",  # Gridline color (e.g., gray, white, black)
    )
    plt.title(title)
    plt.xticks(
        ticks=np.arange(num_classes), labels=class_names, rotation=90, fontsize=8
    )  # type: ignore
    plt.yticks(ticks=np.arange(num_classes), labels=class_names, rotation=0, fontsize=8)  # type: ignore
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if disp:
        plt.show()


