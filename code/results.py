"""
Utility functions for generating LaTeX tables and bar graphs
from WLASL SATNAC model results.

Usage:
    from model_results_utils import *
    data = load_data("wlasl_satnac_16_only_improved_summarised.json")
    
    # LaTeX tables
    print(latex_individual_table("MViTv2_S", data))
    print(latex_comparative_table(data))
    
    # Plots
    plot_individual_bar("MViTv2_S", data)
    plot_comparative_bar(data)
    plot_comparative_grouped_bar(data)
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_data(filepath: str) -> dict:
    """Load summarised results JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# LaTeX helpers
# ─────────────────────────────────────────────

def _fmt(val: float) -> str:
    """Format a float as a percentage string (2 d.p.)."""
    return f"{val * 100:.2f}\\%"


def latex_individual_table(model_name: str, data: dict) -> str:
    """
    Return a LaTeX table for a single model showing Top-1, Top-5, Top-10 accuracy.

    Parameters
    ----------
    model_name : str
        Key in *data* (e.g. ``"MViTv2_S"``).
    data : dict
        Loaded summarised results dict.

    Returns
    -------
    str
        LaTeX table string ready to paste into a .tex document.
    """
    if model_name not in data:
        raise KeyError(f"Model '{model_name}' not found. Available: {list(data.keys())}")

    entry = data[model_name]
    acc = entry["top_k_average_per_class_acc"]
    exp = entry["exp"]

    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        rf"    \caption{{Top-$k$ average per-class accuracy for \textbf{{{model_name}}} (Exp.~{exp})}}",
        rf"    \label{{tab:{model_name.lower().replace('(', '').replace(')', '').replace('+', 'p')}}}",
        r"    \begin{tabular}{lc}",
        r"        \toprule",
        r"        \textbf{Metric} & \textbf{Accuracy} \\",
        r"        \midrule",
        rf"        Top-1  & {_fmt(acc['top1'])} \\",
        rf"        Top-5  & {_fmt(acc['top5'])} \\",
        rf"        Top-10 & {_fmt(acc['top10'])} \\",
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_all_individual_tables(data: dict) -> str:
    """
    Return LaTeX for every model's individual table, separated by blank lines.

    Parameters
    ----------
    data : dict
        Loaded summarised results dict.

    Returns
    -------
    str
        Concatenated LaTeX tables.
    """
    tables = [latex_individual_table(m, data) for m in data]
    return "\n\n".join(tables)


def latex_comparative_table(data: dict, caption: str = None, label: str = "tab:comparative") -> str:
    """
    Return a LaTeX table comparing all models across Top-1, Top-5, Top-10.

    Parameters
    ----------
    data : dict
        Loaded summarised results dict.
    caption : str, optional
        Custom caption. Defaults to a generic one.
    label : str
        LaTeX label for the table.

    Returns
    -------
    str
        LaTeX table string.
    """
    if caption is None:
        caption = "Comparative top-$k$ average per-class accuracy across all models on WLASL-100 (SATNAC split)."

    # Find best value per column for bolding
    top1_vals = {m: data[m]["top_k_average_per_class_acc"]["top1"] for m in data}
    top5_vals = {m: data[m]["top_k_average_per_class_acc"]["top5"] for m in data}
    top10_vals = {m: data[m]["top_k_average_per_class_acc"]["top10"] for m in data}
    best_top1 = max(top1_vals.values())
    best_top5 = max(top5_vals.values())
    best_top10 = max(top10_vals.values())

    def cell(val, best):
        s = _fmt(val)
        return rf"\textbf{{{s}}}" if val == best else s

    rows = []
    for model in data:
        acc = data[model]["top_k_average_per_class_acc"]
        exp = data[model]["exp"]
        model_tex = model.replace("_", r"\_")
        rows.append(
            f"        {model_tex} & {exp} & "
            f"{cell(acc['top1'], best_top1)} & "
            f"{cell(acc['top5'], best_top5)} & "
            f"{cell(acc['top10'], best_top10)} \\\\"
        )

    lines = [
        r"\begin{table}[h]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}}",
        r"    \begin{tabular}{llccc}",
        r"        \toprule",
        r"        \textbf{Model} & \textbf{Exp.} & \textbf{Top-1} & \textbf{Top-5} & \textbf{Top-10} \\",
        r"        \midrule",
    ] + rows + [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_comparative_table_by_metric(data: dict, metric: str = "top1") -> str:
    """
    Return a comparative table sorted by a single metric (descending).

    Parameters
    ----------
    data : dict
        Loaded summarised results dict.
    metric : str
        One of ``"top1"``, ``"top5"``, ``"top10"``.

    Returns
    -------
    str
        LaTeX table string sorted by *metric*.
    """
    valid = {"top1", "top5", "top10"}
    if metric not in valid:
        raise ValueError(f"metric must be one of {valid}")

    sorted_models = sorted(
        data.keys(),
        key=lambda m: data[m]["top_k_average_per_class_acc"][metric],
        reverse=True,
    )

    caption = (
        f"Models ranked by Top-{metric[-1]} average per-class accuracy "
        f"on WLASL-100 (SATNAC split)."
    )
    label = f"tab:ranked_{metric}"

    sorted_data = {m: data[m] for m in sorted_models}
    # Temporarily redirect to comparative table builder with sorted data
    raw = latex_comparative_table(sorted_data, caption=caption, label=label)
    return raw


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

_METRICS = ["top1", "top5", "top10"]
_METRIC_LABELS = ["Top-1", "Top-5", "Top-10"]
_PALETTE = ["#2563EB", "#16A34A", "#DC2626"]   # blue, green, red


def plot_individual_bar(
    model_name: str,
    data: dict,
    ax: plt.Axes = None,
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Bar chart for a single model showing Top-1, Top-5, Top-10 accuracy.

    Parameters
    ----------
    model_name : str
        Key in *data*.
    data : dict
        Loaded summarised results dict.
    ax : plt.Axes, optional
        Existing axes to draw on. If None, a new figure is created.
    save_path : str, optional
        If provided, the figure is saved to this path.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    plt.Figure
    """
    if model_name not in data:
        raise KeyError(f"Model '{model_name}' not found.")

    acc = data[model_name]["top_k_average_per_class_acc"]
    values = [acc[m] * 100 for m in _METRICS]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    bars = ax.bar(_METRIC_LABELS, values, color=_PALETTE, width=0.5, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.2f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Average Per-Class Accuracy (%)")
    ax.set_title(f"{model_name}  (Exp. {data[model_name]['exp']})", fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

    return fig


def plot_all_individual_bars(
    data: dict,
    ncols: int = 4,
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Grid of individual bar charts, one per model.

    Parameters
    ----------
    data : dict
        Loaded summarised results dict.
    ncols : int
        Number of columns in the grid.
    save_path : str, optional
        Save path for the combined figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    plt.Figure
    """
    models = list(data.keys())
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes_flat = axes.flatten()

    for i, model in enumerate(models):
        plot_individual_bar(model, data, ax=axes_flat[i], show=False)

    # Hide unused axes
    for j in range(len(models), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Individual Model Results – WLASL-100 (SATNAC Split)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_comparative_bar(
    data: dict,
    metric: str = "top1",
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Horizontal bar chart comparing all models on a single metric.

    Parameters
    ----------
    data : dict
        Loaded summarised results dict.
    metric : str
        One of ``"top1"``, ``"top5"``, ``"top10"``.
    save_path : str, optional
        Save path for the figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    plt.Figure
    """
    valid = {"top1", "top5", "top10"}
    if metric not in valid:
        raise ValueError(f"metric must be one of {valid}")

    models = list(data.keys())
    values = [data[m]["top_k_average_per_class_acc"][metric] * 100 for m in models]

    # Sort descending
    order = np.argsort(values)[::-1]
    models_sorted = [models[i] for i in order]
    values_sorted = [values[i] for i in order]

    cmap = plt.cm.Blues
    colors = [cmap(0.4 + 0.5 * (v - min(values_sorted)) / (max(values_sorted) - min(values_sorted) + 1e-9))
              for v in values_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(models_sorted, values_sorted, color=colors, edgecolor="white", linewidth=1)

    for bar, val in zip(bars, values_sorted):
        ax.text(
            val + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center", fontsize=9, fontweight="bold",
        )

    metric_label = {"top1": "Top-1", "top5": "Top-5", "top10": "Top-10"}[metric]
    ax.set_xlabel(f"{metric_label} Average Per-Class Accuracy (%)")
    ax.set_title(f"Model Comparison — {metric_label} Accuracy\nWLASL-100 (SATNAC Split)", fontweight="bold")
    ax.set_xlim(0, max(values_sorted) + 8)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_comparative_grouped_bar(
    data: dict,
    save_path: str = None,
    show: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart comparing all models across Top-1, Top-5, Top-10.

    Parameters
    ----------
    data : dict
        Loaded summarised results dict.
    save_path : str, optional
        Save path for the figure.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    plt.Figure
    """
    models = list(data.keys())
    n = len(models)
    top1 = [data[m]["top_k_average_per_class_acc"]["top1"] * 100 for m in models]
    top5 = [data[m]["top_k_average_per_class_acc"]["top5"] * 100 for m in models]
    top10 = [data[m]["top_k_average_per_class_acc"]["top10"] * 100 for m in models]

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))

    b1 = ax.bar(x - width, top1,  width, label="Top-1",  color=_PALETTE[0], edgecolor="white")
    b5 = ax.bar(x,          top5,  width, label="Top-5",  color=_PALETTE[1], edgecolor="white")
    b10 = ax.bar(x + width, top10, width, label="Top-10", color=_PALETTE[2], edgecolor="white")

    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5,
                f"{h:.1f}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    _label_bars(b1)
    _label_bars(b5)
    _label_bars(b10)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Average Per-Class Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Model Comparison — Top-k Accuracy\nWLASL-100 (SATNAC Split)", fontweight="bold")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ─────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # DATA_FILE = "/home/luke/Code/SLR/code/results/wlasl_satnac_16_only_improved_summarised.json"
    DATA_FILE = "/home/luke/Code/SLR/code/results/wlasl_satnac_16_only_summary.json"
    if not os.path.exists(DATA_FILE):
        print(f"Data file '{DATA_FILE}' not found – place it in the same directory.")
    else:
        # data = load_data(DATA_FILE)
        data = load_data(DATA_FILE)['asl100']

        # ── LaTeX ──────────────────────────────────
        # print("=" * 60)
        # print("INDIVIDUAL TABLE — MViTv2_S")
        # print("=" * 60)
        # print(latex_individual_table("MViTv2_S", data))

        print("\n" + "=" * 60)
        print("COMPARATIVE TABLE (all models)")
        print("=" * 60)
        print(latex_comparative_table(data))

        print("\n" + "=" * 60)
        print("COMPARATIVE TABLE sorted by Top-1")
        print("=" * 60)
        print(latex_comparative_table_by_metric(data, metric="top1"))

        # ── Plots ──────────────────────────────────
        # plot_individual_bar("MViTv2_S", data,
        #                     save_path="individual_bar_MViTv2_S.png", show=False)
        # print("\nSaved: individual_bar_MViTv2_S.png")

        plot_all_individual_bars(data,
                                 save_path="all_individual_bars.png", show=False)
        print("Saved: all_individual_bars.png")

        plot_comparative_bar(data, metric="top1",
                             save_path="comparative_bar_top1.png", show=False)
        print("Saved: comparative_bar_top1.png")

        plot_comparative_grouped_bar(data,
                                     save_path="comparative_grouped_bar.png", show=False)
        print("Saved: comparative_grouped_bar.png")