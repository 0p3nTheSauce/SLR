"""
Utility functions for generating LaTeX tables and visualisations
for WLASL dataset statistics.

This module expects precomputed statistics using the data structures:
`split_stats`, `set_stats`, `class_stats`, etc.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from typing import TypedDict, Literal, TypeAlias

# -------------------------------------------------------------------------
# Type aliases (copied for isolation — safe to remove if already imported)
# -------------------------------------------------------------------------
AVAIL_SETS: TypeAlias = Literal["train", "val", "test"]
AVAIL_SPLITS: TypeAlias = Literal["asl100", "asl300", "asl1000", "asl2000"]

class class_stats(TypedDict):
    num_instances: int
    num_signers: int
    num_variations: int

class set_stats(TypedDict):
    num_instances: int
    num_signers: int
    per_class_stats: Dict[str, class_stats]

class split_stats(TypedDict):
    num_classes: int
    num_instances: int
    num_signers: int
    per_set_stats: Dict[AVAIL_SETS, set_stats]

# =============================================================================
#                       LaTeX TABLE GENERATION
# =============================================================================

def latex_set_summary_table(split_name: AVAIL_SPLITS, stats: split_stats) -> str:
    """
    Generate a LaTeX table summarising statistics for each set (train/val/test)
    for a given split.

    Parameters
    ----------
    split_name : AVAIL_SPLITS
        Name of the dataset split, e.g. "asl100".
    stats : split_stats
        Statistics structure for the split.

    Returns
    -------
    str
        A LaTeX table environment as a string.
    """
    rows = []
    for set_name, s in stats["per_set_stats"].items():
        rows.append(
            f"{set_name} & {s['num_instances']} & {s['num_signers']} & "
            f"{len(s['per_class_stats'])} \\\\"
        )

    table = r"""
        \begin{table}[h]
        \centering
        \begin{tabular}{lccc}
        \hline
        Set & Instances & Signers & Classes \\
        \hline
        """ + "\n".join(rows) + r"""
        \hline
        \end{tabular}
        \caption{Statistics summary for split %s.}
        \end{table}
        """ % split_name

    return table.strip()


def latex_class_stats_table(
    split_name: AVAIL_SPLITS,
    set_name: AVAIL_SETS,
    set_stats_obj: set_stats,
) -> str:
    """
    Generate a LaTeX table listing per-class statistics for a chosen subset.

    Parameters
    ----------
    split_name : AVAIL_SPLITS
        Name of the dataset split (e.g., "asl100").
    set_name : AVAIL_SETS
        The subset to summarise (train/val/test).
    set_stats_obj : set_stats
        The statistics object for the chosen set.

    Returns
    -------
    str
        A LaTeX table environment as a string.
    """
    rows = []
    for gloss, cstats in set_stats_obj["per_class_stats"].items():
        rows.append(
            f"{gloss} & {cstats['num_instances']} & "
            f"{cstats['num_signers']} & {cstats['num_variations']} \\\\"
        )

    table = r"""
        \begin{table}[h]
        \centering
        \begin{tabular}{lccc}
        \hline
        Gloss & Instances & Signers & Variations \\
        \hline
        """ + "\n".join(rows) + r"""
        \hline
        \end{tabular}
        \caption{Class-level statistics for %s (%s split).}
        \end{table}
        """ % (set_name, split_name)

    return table.strip()


# =============================================================================
#                       VISUALISATION HELPERS
# =============================================================================

def barplot_metric(
    per_class: Dict[str, class_stats],
    metric: str,
    top_n: Optional[int] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> None:
    """
    Create a bar plot for a chosen metric across glosses.

    Parameters
    ----------
    per_class : dict
        Mapping gloss → class_stats.
    metric : str
        One of {"num_instances", "num_signers", "num_variations"}.
    top_n : int, optional
        If provided, only plot the top-N glosses for the chosen metric.
    title : str, optional
        Plot title.
    figsize : tuple
        Matplotlib figure size.
    """
    if metric not in {"num_instances", "num_signers", "num_variations"}:
        raise ValueError(f"Invalid metric: {metric}")

    # Extract metric values per gloss
    items = [(g, stats[metric]) for g, stats in per_class.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    if top_n:
        items = items[:top_n]

    glosses, values = zip(*items)

    plt.figure(figsize=figsize)
    plt.bar(glosses, values)
    plt.xticks(rotation=90)
    plt.ylabel(metric.replace("_", " ").title())
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def histogram_metric(
    per_class: Dict[str, class_stats],
    metric: str,
    bins: int = 20,
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> None:
    """
    Plot a histogram of a chosen metric across glosses.

    Parameters
    ----------
    per_class : dict
        Mapping gloss → class_stats.
    metric : str
        One of {"num_instances", "num_signers", "num_variations"}.
    bins : int
        Number of histogram bins.
    title : str, optional
        Plot title.
    figsize : tuple
        Matplotlib figure size.
    """
    if metric not in {"num_instances", "num_signers", "num_variations"}:
        raise ValueError(f"Invalid metric: {metric}")

    values = [stats[metric] for stats in per_class.values()]

    plt.figure(figsize=figsize)
    plt.hist(values, bins=bins)
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Frequency")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def scatter_instances_vs_signers(
    per_class: Dict[str, class_stats],
    figsize: tuple = (6, 5),
    title: Optional[str] = None,
) -> None:
    """
    Scatter plot of num_instances vs num_signers for each gloss.

    Parameters
    ----------
    per_class : dict
        Mapping gloss → class_stats.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    """
    x = [c["num_instances"] for c in per_class.values()]
    y = [c["num_signers"] for c in per_class.values()]

    plt.figure(figsize=figsize)
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("Number of Instances")
    plt.ylabel("Number of Signers")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()