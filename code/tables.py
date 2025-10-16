import json
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Optional
import pandas as pd

def compare_splits_latex_table(wlasl_100_data, wlasl_300_data):
	"""
	Create a LaTeX table comparing WLASL-100 and WLASL-300 splits.
	
	Args:
		wlasl_100_data (dict): Dictionary with train/val/test data for WLASL-100
		wlasl_300_data (dict): Dictionary with train/val/test data for WLASL-300
	
	Returns:
		str: LaTeX table code
	"""
	
	latex_code = r"""\begin{table}[h!]
	\centering
	\begin{tabular}{|l|c|c|c|c|c|c|}
	\hline
	\textbf{Split} & \multicolumn{3}{c|}{\textbf{WLASL-100}} & \multicolumn{3}{c|}{\textbf{WLASL-300}} \\
	\cline{2-7}
	& \textbf{Examples} & \textbf{Signers} & \textbf{Mean Ex/Sign} & \textbf{Examples} & \textbf{Signers} & \textbf{Mean Ex/Sign} \\
	\hline
	"""
	
	# Define the splits and their display names
	splits = [('train', 'Train'), ('val', 'Validation'), ('test', 'Test')]
	
	for split_key, split_name in splits:
		wlasl_100 = wlasl_100_data[split_key]
		wlasl_300 = wlasl_300_data[split_key]
		
		latex_code += f"{split_name} & "
		latex_code += f"{wlasl_100['num_ex']} & "
		latex_code += f"{wlasl_100['num_s']} & "
		latex_code += f"{wlasl_100['mean_ex']:.2f} & "
		latex_code += f"{wlasl_300['num_ex']} & "
		latex_code += f"{wlasl_300['num_s']} & "
		latex_code += f"{wlasl_300['mean_ex']:.2f} \\\\\n"
		latex_code += "\\hline\n"
	
	latex_code += r"""\end{tabular}
	\caption{Comparison of WLASL-100 and WLASL-300 dataset splits}
	\label{tab:wlasl_comparison}
	\end{table}"""
	
	return latex_code

def generate_latex_table(results: Dict, dataset: str = 'asl100', 
						split: str = 'test', metric_type: str = 'both',
						caption:Optional[str] = None,
	  					label: Optional[str] = None) -> str:
	"""
	Generate a LaTeX table for the results.
	
	Args:
		results: Dictionary with results
		dataset: 'asl100' or 'asl300'
		split: 'val' or 'test'
		metric_type: 'avg' for average per class, 'inst' for per instance, 'both' for both
		caption: Table caption
		label: Table label for referencing
	
	Returns:
		LaTeX table as string
	"""
	# Extract data
	data = []
	for model in results[dataset]:
		split_key = f'best_{split}'
		if split_key in results[dataset][model]:
			metrics = results[dataset][model][split_key]
			row = {'Model': model.replace('_', '\\_')}  # Escape underscores for LaTeX
			
			if metric_type in ['avg', 'both']:
				row['Top-1'] = f"{metrics['top_k_average_per_class_acc']['top1']*100:.1f}"
				row['Top-5'] = f"{metrics['top_k_average_per_class_acc']['top5']*100:.1f}"
				row['Top-10'] = f"{metrics['top_k_average_per_class_acc']['top10']*100:.1f}"
			
			if metric_type == 'inst':
				row['Top-1'] = f"{metrics['top_k_per_instance_acc']['top1']*100:.1f}"
				row['Top-5'] = f"{metrics['top_k_per_instance_acc']['top5']*100:.1f}"
				row['Top-10'] = f"{metrics['top_k_per_instance_acc']['top10']*100:.1f}"
			
			if metric_type == 'both':
				row['Top-1 (Inst)'] = f"{metrics['top_k_per_instance_acc']['top1']*100:.1f}"
				row['Top-5 (Inst)'] = f"{metrics['top_k_per_instance_acc']['top5']*100:.1f}"
				row['Top-10 (Inst)'] = f"{metrics['top_k_per_instance_acc']['top10']*100:.1f}"
			
			data.append(row)
	
	# Create DataFrame and sort by Top-1 accuracy
	df = pd.DataFrame(data)
	sort_col = 'Top-1' if 'Top-1' in df.columns else list(df.columns)[1]
	df[sort_col] = df[sort_col].astype(float)
	df = df.sort_values(sort_col, ascending=False)
	df[sort_col] = df[sort_col].apply(lambda x: f"{x:.1f}")
	
	# Generate LaTeX
	latex = "\\begin{table}[htbp]\n"
	latex += "\\centering\n"
	
	if caption:
		latex += f"\\caption{{{caption}}}\n"
	if label:
		latex += f"\\label{{{label}}}\n"
	
	# Determine column alignment
	num_cols = len(df.columns)
	col_align = 'l' + 'c' * (num_cols - 1)
	
	latex += f"\\begin{{tabular}}{{{col_align}}}\n"
	latex += "\\toprule\n"
	
	# Header
	header = " & ".join(df.columns)
	latex += header + " \\\\\n"
	latex += "\\midrule\n"
	
	# Data rows
	for _, row in df.iterrows():
		row_str = " & ".join(str(val) for val in row.values)
		latex += row_str + " \\\\\n"
	
	latex += "\\bottomrule\n"
	latex += "\\end{tabular}\n"
	latex += "\\end{table}\n"
	
	return latex

def calc_diffs(res_dict1: dict, res_dict2: dict, met_key: str) -> dict:
    """res_dict1 - res_dict2 (e.g. flipped - orig). Assumes structure
    produced by test.py summarise"""
    res = {}
    for split in res_dict1.keys():
        res[split] = {}
        for arch in res_dict1[split].keys():
            res[split][arch] = {}

            topk1 = res_dict1[split][arch][met_key]
            topk2 = res_dict2[split][arch][met_key]
            topkeys = ["top1", "top5", "top10"]
            diff = {}

            for top_key in topkeys:
                diff[top_key] = topk1[top_key] - topk2[top_key]

            res[split][arch][met_key] = diff
    return res


def fix_name(name: str) -> str:
    """
    Escape special characters in model names for LaTeX.

    Args:
            name: Model name like 'MViT_V1_B'

    Returns:
            LaTeX-safe string with escaped characters
    """
    # LaTeX special characters that need escaping
    latex_escapes = {
        "_": r"\_",  # Underscore (most common in model names)
        "#": r"\#",  # Hash
        "$": r"\$",  # Dollar sign
        "%": r"\%",  # Percent
        "&": r"\&",  # Ampersand
        "^": r"\textasciicircum{}",  # Caret
        "~": r"\textasciitilde{}",  # Tilde
        "{": r"\{",  # Left brace
        "}": r"\}",  # Right brace
        # '\\': r'\textbackslash{}',   # Backslash
    }

    result = name
    for char, escape in latex_escapes.items():
        result = result.replace(char, escape)

    return result


def gen_compare_table(
    res_dict1: dict,
    res_dict2: dict,
    met_key: str = "top_k_average_per_class_acc",
    split: str = "asl100",
    caption: str = "Differences between table 1 and 2",
    label: str = "tab:compare",
    footnotes: list[str] = [],
    precision: int = 2,
) -> str:
    """res_dict1 - res_dict2 (e.g. flipped - orig)"""
    diffs = calc_diffs(res_dict1, res_dict2, met_key)

    return gen_single_split_table(
        split_dict=diffs[split],
        met_key=met_key,
        caption=caption,
        label=label,
        footnotes=footnotes,
        precision=precision,
    )


def compare_shuffled(
    shuffled_path: str | Path = "wlasl_shuffled_summary.json",
    normal_path: str | Path = "wlasl_runs_summary.json",
    split: str = "asl100",
):
    with open(shuffled_path, "r") as f:
        flip_dict = json.load(f)
    if not flip_dict:
        raise ValueError("no flipped data")

    with open(normal_path, "r") as f:
        norm_dict = json.load(f)
    if not norm_dict:
        raise ValueError("no normal run data")

    print(
        gen_compare_table(
            res_dict1=flip_dict,
            res_dict2=norm_dict,
            met_key="top_k_average_per_class_acc",
            split=split,
            caption=f"Differences on shuffled videos (shuffled - normal) for {split}",
            label="tab:shuffled_diffs",
            footnotes=["Difference in average per class accuracy"],
        )
    )


def gen_single_split_table(
    split_dict: dict[str, dict[str, str | dict[str, float]]],
    met_key: str = "top_k_average_per_class_acc",
    caption: str = "Table Caption",
    label: str = "tab:label",
    footnotes: list[str] = [],
    precision: int = 2,
) -> str:
    table = "\\begin{table}[t]\n"
    table += "\\begin{center} \n"
    table += f"\\caption{{{caption}}}\n"
    table += f"\\label{{{label}}}\n"
    table += "\\begin{tabular}{|l|ccc|}\n"
    table += "\\hline"
    table += """\\textbf{Model} & \\textbf{Acc@1} & \\textbf{Acc@5} & \\textbf{Acc@10} \\\\ \n"""
    table += "\\hline \n"

    for arch in split_dict.keys():
        table += fix_name(arch) + " "
        topk = split_dict[arch][met_key]

        assert isinstance(topk, dict), (
            f"Expected dict for {arch}[{met_key}], got {type(topk)}"
        )

        for top_key in topk.keys():
            table += f"& {round(topk[top_key] * 100, precision)} "

        table += "\\\\ \n"

    table += "\\hline \n"

    for f in footnotes:
        table += f"\\multicolumn{{4}}{{l}}{{{f}}} \\\\ \n"

    table += "\\end{tabular} \n"
    table += "\\end{center} \n"
    table += "\\end{table} \n"
    return table


if __name__ == "__main__":
    # flip_sum = Path('wlasl_flipped_summary.json')
    # reformat(flip_sum)
    #   compare_flipped()
    parser = ArgumentParser(description="Generate LaTeX tables from JSON results")

    subparsers = parser.add_subparsers(
        dest="command", help="Sub-command to run", required=True
    )

    # single split table
    single_parser = subparsers.add_parser(
        "single",
        help="Generate a LaTeX table for a single dataset split from a JSON results  (summarised) file",
    )
    single_parser.add_argument(
        "-j",
        "--json_path",
        type=Path,
        help="Path to the JSON results file",
        required=True,
    )
    single_parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="asl100",
        help="Dataset split to use (default: 'asl100')",
    )
    single_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="top_k_average_per_class_acc",
        help="Metric to use (default: 'top_k_average_per_class_acc')",
    )
    single_parser.add_argument(
        "-c",
        "--caption",
        type=str,
        help="Caption for the LaTeX table",
        default="table caption",
    )
    single_parser.add_argument(
        "-l", "--label", type=str, help="Label for the LaTeX table", default="tab:label"
    )
    single_parser.add_argument(
        "-f",
        "--footnote",
        type=str,
        action="append",
        default=[],
        help="Footnotes for the LaTeX table (can be used multiple times)",
    )
    single_parser.add_argument(
        "-p",
        "--precision",
        type=int,
        default=2,
        help="Decimal precision for the metrics (default: 2)",
    )

    # compare two result sets
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two JSON results (summarised) files and generate a LaTeX table of the differences",
    )
    compare_parser.add_argument(
        "-j1",
        "--json_path1",
        type=Path,
        help="Path to the first JSON results file",
        required=True,
    )
    compare_parser.add_argument(
        "-j2",
        "--json_path2",
        type=Path,
        help="Path to the second JSON results file",
        required=True,
    )
    compare_parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="asl100",
        help="Dataset split to use (default: 'asl100')",
    )
    compare_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        default="top_k_average_per_class_acc",
        help="Metric to use (default: 'top_k_average_per_class_acc')",
    )

    args = parser.parse_args()

    if args.command == "single":
        with open(args.json_path, "r") as f:
            data_dict = json.load(f)
        print(
            gen_single_split_table(
                split_dict=data_dict[args.split],
                met_key=args.metric,
                caption=args.caption,
                label=args.label,
                footnotes=args.footnote,
                precision=args.precision,
            )
        )

    elif args.command == "compare":
        compare_shuffled(
            shuffled_path=args.json_path1, normal_path=args.json_path2, split=args.split
        )
