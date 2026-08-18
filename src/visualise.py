from typing import (
    Dict,
    Optional,
    Callable,
    List,
    Union,
    Any,
    Literal,
    Tuple
)
from typing_extensions import TypedDict, Unpack
from sklearn.metrics import confusion_matrix
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
from logging import Logger
from torch.utils.data import DataLoader, Dataset

from torch import Tensor

# locals
from src.run_types import CentreCropConfig, OG_Sampler
from src.configs import get_class_list, CLASSES_PATH
from src.video_dataset import (
    get_wlasl_info,
    get_video_path,
    get_transform,
)
from src.utils import plt_display_grid, load_rgb_frames_from_video
from src.preprocess import Instance
from src.stats import (
    AVAIL_SETS,
    AVAIL_SPLITS,
    HistoGram,
    instance_stats,
    
)

# Set style for better-looking plots
# plt.style.use("seaborn-v0_8-darkgrid")
# sns.set_palette("husl")


VERBOSITY = 0


visualise_logger = logging.getLogger(__name__)

BG_FIGSIZE_3G = (8, 5)
BG_WiDTH_3G = 0.2
FRAME_WIDTH: int = 256
FRAME_HEIGHT: int = 256
PALETTE = {
    "train": "#4C72B0",
    "val": "#55A868",
    "test": "#DD8452",
    "default_cmap": "viridis",
    "categorical_short": "Dark2",
    "categorical_cmap": "tab20",
}


def set_font_size(size: int = 14) -> None:
    """Set the font size for all matplotlib plots."""
    plt.rcParams.update({"font.size": size})


set_font_size()  # Set default font size


# Frame visualiser


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
        transform: Optional[Callable[[Tensor], Tensor]] = None,
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
    def __init__(self, **kwargs: Unpack[MiniSetKwargs]):
        self.target_frames = kwargs.get("target_length", 16)
        self.frame_size = kwargs.get('frame_size', 224)
        self.iter_loader = iter(
            DataLoader(
                MiniSet(**kwargs),
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=False,
            )
        )

    def __call__(self):
        frames = next(self.iter_loader)[0]
        if len(frames.shape) == 5:
            frames = frames.squeeze(dim=0)
        if frames.shape[1] != 3:
            frames = frames.permute(1, 0, 2, 3)  # swap T and C
        
        plt_display_grid(frames, self.target_frames)
        
class FrameFetcher:
    def __init__(self, **kwargs: Unpack[MiniSetKwargs]):
        self.frames: Tensor | None = None
        self.iter_loader = iter(
            DataLoader(
                MiniSet(**kwargs),
                batch_size=1,
                shuffle=False,
                num_workers=4,
                pin_memory=False,
            )
        )

    def __getitem__(self, key: int) -> Tensor:
        if self.frames is not None:
            return self.frames[key]
        else:
            raise IndexError('No frames set, call frame_fetcher()')

    def __call__(self) -> Tensor:
        frames = next(self.iter_loader)[0]
        self.frames = frames
        if len(frames.shape) == 5:
            frames = frames.squeeze(dim=0)
        if frames.shape[1] != 3:
            frames = frames.permute(1, 0, 2, 3)  # swap T and C
        
        return frames


# Standardised plots for results


# def 


# half claude
def plot_distribution(
    histogram: HistoGram,
    set_name: AVAIL_SETS,
    split_name: AVAIL_SPLITS,
    metric: str,
    gloss: str = "",
    unit: str = "",
    categorical: bool = False,
    hist_or_bar: Literal["hist", "bar"] = "hist",
    bins: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 4),
    show_nums_on_bars: bool = True,
    no_statsy_lines: bool = False,
    out_path: str = "",
    x_label_step: Optional[int] = 1,  # None = no labels, N = show every Nth label
) -> None:

    sorted_items = sorted(histogram.items(), key=lambda x: x[0])
    values, counts = zip(*sorted_items)
    expanded_values = [value for value, count in sorted_items for _ in range(count)]

    plt.figure(figsize=figsize)

    if categorical:
        cmap = plt.get_cmap(PALETTE["categorical_cmap"])
        x_pos = range(len(values))
        bar_colors = [cmap(i / max(len(values) - 1, 1)) for i in range(len(values))]
        plt.bar(x_pos, counts, width=0.8, color=bar_colors)
        if show_nums_on_bars:
            for x, y in zip(x_pos, counts):
                plt.text(x, y, str(y), ha="center", va="bottom")
        if x_label_step is None:
            plt.xticks([])
        else:
            visible_pos = [x for x in x_pos if x % x_label_step == 0]
            visible_labels = [values[x] for x in visible_pos]
            plt.xticks(visible_pos, visible_labels, rotation=45, ha="right")

    elif hist_or_bar == "hist":
        cmap = plt.get_cmap(PALETTE["default_cmap"])
        n, bins_out, patches_list = plt.hist(expanded_values, bins=bins)
        norm = plt.Normalize(n.min(), n.max())  # type: ignore
        for patch, count in zip(patches_list, n):  # type: ignore
            patch.set_facecolor(cmap(norm(count)))
        if show_nums_on_bars:
            for patch, count in zip(patches_list, n):  # type: ignore
                x = patch.get_x() + patch.get_width() / 2
                if count > 0:
                    plt.text(x, count, str(int(count)), ha="center", va="bottom")

    else:
        cmap = plt.get_cmap(PALETTE["default_cmap"])
        norm = plt.Normalize(min(counts), max(counts))  # type: ignore
        bar_colors = [cmap(norm(c)) for c in counts]
        plt.bar(values, counts, width=2, color=bar_colors)
        if x_label_step is None:
            plt.xticks([])
        elif x_label_step > 1:
            visible = [v for i, v in enumerate(values) if i % x_label_step == 0]
            plt.xticks(visible, rotation=45, ha="right")
        if show_nums_on_bars:
            for x, y in zip(values, counts):
                plt.text(x, y, str(y), ha="center", va="bottom")

        if not no_statsy_lines:
            statsy_listy = [
                ("mean", lambda x: np.mean(x), "red"),
                ("median", lambda x: np.median(x), "blue"),
                ("lower quartile", lambda x: np.percentile(x, 25), "brown"),
                ("upper quartile", lambda x: np.percentile(x, 75), "brown"),
            ]
            for metric_name, stat_func, colour in statsy_listy:
                metric_val = stat_func(expanded_values)
                plt.axvline(
                    metric_val,
                    color=colour,
                    linestyle="--",
                    label=f"{metric_name}: {metric_val:.1f}{' ' + unit if unit else ''}",
                )

    plt.legend()
    plt.xlabel(f"{metric}{' (' + unit + ')' if unit else ''}")
    plt.ylabel("Count")
    title = f"{metric.capitalize()} Distribution: {split_name} / {set_name}"
    if gloss:
        title += f' / "{gloss}"'
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    plt.show()


def plot_dimension_distributions(
    instances: List[Instance],
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """Histograms of bbox width and height across all instances."""
    widths = [inst.bbox[2] - inst.bbox[0] for inst in instances]
    heights = [inst.bbox[3] - inst.bbox[1] for inst in instances]

    cmap = plt.get_cmap("coolwarm")
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, data, label, colour in zip(
        axes,
        [widths, heights],
        ["Width (px)", "Height (px)"],
        [cmap(0.2), cmap(0.8)],
    ):
        ax.hist(data, bins=30, color=colour, edgecolor="white")
        statsy_listy = [
            ("mean", lambda x: np.mean(x), "red"),
            ("median", lambda x: np.median(x), "blue"),
            ("lower quartile", lambda x: np.percentile(x, 25), "brown"),
            ("upper quartile", lambda x: np.percentile(x, 75), "brown"),
        ]
        for metric_name, stat_func, line_colour in statsy_listy:
            metric_val = stat_func(data)
            ax.axvline(
                metric_val,
                color=line_colour,
                linestyle="--",
                label=f"{metric_name}: {metric_val:.1f} (px)",
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle("BBox Dimension Distributions")
    plt.tight_layout()
    plt.show()


def barplot_metric(
    per_class: Dict[str, instance_stats],
    metric: str,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> None:
    if metric not in {"num_instances", "num_signers", "num_variations"}:
        raise ValueError(f"Invalid metric: {metric}")

    items = [(g, stats[metric]) for g, stats in per_class.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    if top_n:
        items = items[:top_n]

    glosses, values = zip(*items)
    cmap = plt.get_cmap(PALETTE["default_cmap"])
    colors = [cmap(i / max(len(values) - 1, 1)) for i in range(len(values))]

    plt.figure(figsize=figsize)
    plt.bar(glosses, values, color=colors)
    plt.xticks(rotation=90)
    plt.ylabel(metric.replace("_", " ").title())
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def histogram_metric(
    per_class: Dict[str, instance_stats],
    metric: str,
    bins: int = 20,
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> None:
    if metric not in {"num_instances", "num_signers", "num_variations"}:
        raise ValueError(f"Invalid metric: {metric}")

    values = [stats[metric] for stats in per_class.values()]

    plt.figure(figsize=figsize)
    plt.hist(values, bins=bins, color=PALETTE["default_cmap"], edgecolor="white")
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Frequency")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def scatter_instances_vs_signers(
    per_class: Dict[str, instance_stats],
    figsize: tuple = (6, 5),
    title: Optional[str] = None,
) -> None:
    x = [c["num_instances"] for c in per_class.values()]
    y = [len(c["signers_distribution"]) for c in per_class.values()]

    plt.figure(figsize=figsize)
    scatter = plt.scatter(x, y, alpha=0.7, c=x, cmap=PALETTE["default_cmap"])
    plt.colorbar(scatter, label="Num Instances")
    plt.xlabel("Number of Instances")
    plt.ylabel("Number of Signers")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# claude/me
AverageMethod = Literal["mean", "median"]


def average_bboxes(
    instances: List[Instance],
    method: AverageMethod = "mean",
) -> List[Instance]:
    """Return one representative Instance per class with an averaged bounding box."""
    from collections import defaultdict
    import numpy as np

    groups: dict[str, list[Instance]] = defaultdict(list)
    for inst in instances:
        groups[inst.label_name].append(inst)

    avg_fn = np.mean if method == "mean" else np.median

    averaged = []
    for _, insts in groups.items():
        boxes = np.array([inst.bbox for inst in insts], dtype=float)  # (N, 4)
        avg_box = avg_fn(boxes, axis=0).round().astype(int).tolist()
        # Use the first instance as a template, just swap the bbox
        averaged.append(insts[0].model_copy(update={"bbox": avg_box}))

    return averaged


# claude/me
def plot_bboxes_on_canvas(
    instances: List[Instance],
    figsize: Tuple[int, int] = (6, 6),
    average: bool = True,
    method: AverageMethod = "mean",
    title: str = "Bounding Boxes by Class",
    out_path: str = "",
) -> None:
    """Draw bounding boxes for each class on a blank 256x256 canvas, coloured by class."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, FRAME_WIDTH)
    ax.set_ylim(0, FRAME_HEIGHT)
    ax.invert_yaxis()  # image coordinates: y increases downward
    # ax.set_facecolor("#1a1a1a")
    # fig.patch.set_facecolor("#1a1a1a")

    unique_labels = list(set(inst.label_name for inst in instances))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    colour_map = {label: cmap(i) for i, label in enumerate(unique_labels)}

    if average:
        instances = average_bboxes(instances, method)

    for inst in instances:
        x1, y1, x2, y2 = inst.bbox
        w, h = x2 - x1, y2 - y1
        colour = colour_map[inst.label_name]
        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=1, edgecolor=colour, facecolor=(*colour[:3], 0.05)
        )
        ax.add_patch(rect)

    # ax.set_title(title, color="white")
    # ax.tick_params(colors="white")
    ax.set_title(
        title,
    )
    # ax.tick_params()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
    plt.show()


# Simple plotting utilities used by testing


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
