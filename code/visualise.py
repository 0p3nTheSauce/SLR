from sklearn.metrics import confusion_matrix
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional, Callable, List

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(filepath: str) -> Dict:
	"""Load results from JSON file."""
	with open(filepath, 'r') as f:
		return json.load(f)



def plot_heatmap(
    report,
    classes_path,
    title="Classification Report Heatmap",
    save_path=None,
    disp=True,
):
    with open(classes_path, "r") as f:
        test_classes = json.load(f)

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
        plt.savefig(
            save_path,
        )
    if disp:
        plt.show()

def plot_bar_graph_reports_metric(reports, classes_path, metric, names):
    with open(classes_path, "r") as f:
        test_classes = json.load(f)
    classes = list(reports[0].keys())[
        :-3
    ]  # Exclude 'accuracy', 'macro avg', 'weighted avg'

    num_reports = len(reports)
    assert num_reports == len(names)  # it may be better to extract names from reports

    # Create bar plot
    x = np.arange(len(classes))
    width = 0.8 / num_reports  # 0.8 gives good spacing, adjust as needed

    fig, ax = plt.subplots(figsize=(10, 18))
    for i, report in enumerate(reports):
        metric_list = [report[cls][metric] for cls in classes]
        offset = (i - (num_reports - 1) / 2) * width  # Center the bars
        ax.barh(x + offset, metric_list, height=width, label=f"{names[i]}", alpha=0.8)

    ax.set_ylabel("Classes")
    ax.set_xlabel(f"{metric} scores")
    ax.set_title(f"{metric}")

    class_labels = [
        test_classes[int(cls)] if int(cls) < len(test_classes) else f"Class_{cls}"
        for cls in classes
    ]
    ax.set_yticks(x)  # This is the key missing line!
    ax.set_yticklabels(class_labels)

    ax.legend()
    ax.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()



def plot_bar_graph(
    report,
    classes_path,
    title="Classification Report - Per Class Metrics",
    save_path=None,
    disp=True,
):
    with open(classes_path, "r") as f:
        test_classes = json.load(f)

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

def plot_heatmap_reports_metric(reports, classes_path, metric, names):
    with open(classes_path, "r") as f:
        test_classes = json.load(f)

    assert len(reports) == len(names)

    classes = list(reports[0].keys())[:-3]
    metric_scores = [[report[cls][metric] for cls in classes] for report in reports]

    df = pd.DataFrame(metric_scores, index=names, columns=classes)
    df = df.T

    num_to_plot = min(len(classes), len(test_classes))

    plt.figure(figsize=(10, 10))
    sns.heatmap(
        df.iloc[:num_to_plot, :],
        annot=True,
        cmap="Blues",
        fmt=".2f",
        xticklabels=names,
        yticklabels=[test_classes[i] for i in range(num_to_plot)],
    )
    plt.title(f"Classification Report Heatmap - {metric.title()}")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes_path=None,
    num_classes=100,
    title="Confusion Matrix",
    size=(10, 8),
    row_perc=True,
    save_path=None,
    disp=True,
):
    """
    Plot confusion matrix from true and predicted labels

    Parameters:
    y_true: array-like, true labels
    y_pred: array-like, predicted labels
    classes_path: str, path to JSON file with class names (optional)
    title: str, plot title
    """

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Load class names if provided
    class_names = None
    if classes_path is not None and num_classes:
        with open(classes_path, "r") as f:
            test_classes = json.load(f)

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

def extract_metrics_df(results: Dict) -> pd.DataFrame:
	"""Extract metrics into a pandas DataFrame for easier manipulation."""
	data = []
	for dataset in results:
		for model in results[dataset]:
			for split in ['best_val', 'best_test']:
				if split in results[dataset][model]:
					metrics = results[dataset][model][split]
					row = {
						'dataset': dataset,
						'model': model,
						'split': split.replace('best_', ''),
						'top1_avg': metrics['top_k_average_per_class_acc']['top1'],
						'top5_avg': metrics['top_k_average_per_class_acc']['top5'],
						'top10_avg': metrics['top_k_average_per_class_acc']['top10'],
						'top1_inst': metrics['top_k_per_instance_acc']['top1'],
						'top5_inst': metrics['top_k_per_instance_acc']['top5'],
						'top10_inst': metrics['top_k_per_instance_acc']['top10'],
					}
					data.append(row)
	return pd.DataFrame(data)

def extract_metrics_df_sumed(results: Dict, split: str='as1100') -> pd.DataFrame:
	"""Extract metrics into a pandas DataFrame for easier manipulation."""
	data = []
	split_dict = results[split]
	print(results.keys())
	for arch in split_dict:
		top_k_scores = split_dict[arch]['top_k_average_per_class_acc']
		row = {
			'dataset': split,
			'model': arch,
			'split': 'test',
			'top1_avg': top_k_scores['top1'],
			'top5_avg': top_k_scores['top5'],
			'top10_avg': top_k_scores['top10'],
		}
		data.append(row)
	return pd.DataFrame(data)


def plot_model_comparison_bar(df: pd.DataFrame, dataset: str = 'asl100', 
							  metric_type: str = 'avg', split: str = 'test'):
	"""
	Create a grouped bar plot comparing models for a specific dataset.
	
	Args:
		df: DataFrame with extracted metrics
		dataset: 'asl100' or 'asl300'
		metric_type: 'avg' for average per class, 'inst' for per instance
		split: 'val' or 'test'
	"""
	# Filter data
	plot_df = df[(df['dataset'] == dataset) & (df['split'] == split)].copy()
	
	# Prepare data for plotting
	metrics = [f'top1_{metric_type}', f'top5_{metric_type}', f'top10_{metric_type}']
	plot_df = plot_df.sort_values(f'top1_{metric_type}', ascending=False)
	
	# Create figure
	fig, ax = plt.subplots(figsize=(12, 6))
	
	# Bar plot
	x = np.arange(len(plot_df))
	width = 0.25
	
	for i, metric in enumerate(metrics):
		label = f"Top-{metric.split('_')[0].replace('top', '')}"
		ax.bar(x + i*width, plot_df[metric] * 100, width, label=label)
	
	# Customize plot
	ax.set_xlabel('Model', fontsize=12)
	ax.set_ylabel('Accuracy (%)', fontsize=12)
	title_type = 'Average per Class' if metric_type == 'avg' else 'Per Instance'
	ax.set_title(f'{dataset.upper()} - {title_type} Accuracy ({split.capitalize()} Set)', fontsize=14)
	ax.set_xticks(x + width)
	ax.set_xticklabels(plot_df['model'].values, rotation=45, ha='right')
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.show()

def plot_heatmap_comparison(df: pd.DataFrame, metric: str = 'top1_avg', split: str = 'test'):
	"""
	Create a heatmap comparing all models across both datasets.
	
	Args:
		df: DataFrame with extracted metrics
		metric: specific metric to visualize
		split: 'val' or 'test'
	"""
	# Pivot data for heatmap
	heatmap_df = df[df['split'] == split].pivot(
		index='model', 
		columns='dataset', 
		values=metric
	) * 100
	
	# Create figure
	fig, ax = plt.subplots(figsize=(8, 10))
	
	# Create heatmap
	sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlOrRd', 
				cbar_kws={'label': 'Accuracy (%)'}, ax=ax)
	
	# Customize plot
	metric_name = metric.replace('_', ' ').title().replace('Avg', '(Avg per Class)').replace('Inst', '(Per Instance)')
	ax.set_title(f'{metric_name} - {split.capitalize()} Set', fontsize=14)
	ax.set_xlabel('Dataset', fontsize=12)
	ax.set_ylabel('Model', fontsize=12)
	
	plt.tight_layout()
	plt.show()

def plot_top_k_progression(df: pd.DataFrame, dataset: str = 'asl100', 
						  split: str = 'test', metric_type: str = 'avg'):
	"""
	Create a line plot showing how accuracy improves from top-1 to top-10.
	
	Args:
		df: DataFrame with extracted metrics
		dataset: 'asl100' or 'asl300'
		split: 'val' or 'test'
		metric_type: 'avg' or 'inst'
	"""
	# Filter data
	plot_df = df[(df['dataset'] == dataset) & (df['split'] == split)]
	
	# Create figure
	fig, ax = plt.subplots(figsize=(10, 6))
	
	k_values = []
 
	# Plot lines for each model
	for model in plot_df['model'].values:
		model_data = plot_df[plot_df['model'] == model].iloc[0]
		k_values = [1, 5, 10]
		accuracies = [
			model_data[f'top1_{metric_type}'] * 100,
			model_data[f'top5_{metric_type}'] * 100,
			model_data[f'top10_{metric_type}'] * 100
		]
		ax.plot(k_values, accuracies, marker='o', label=model, linewidth=2)
	
	
 
	# Customize plot
	ax.set_xlabel('Top-K', fontsize=12)
	ax.set_ylabel('Accuracy (%)', fontsize=12)
	title_type = 'Average per Class' if metric_type == 'avg' else 'Per Instance'
	ax.set_title(f'{dataset.upper()} - Top-K {title_type} Accuracy ({split.capitalize()} Set)', fontsize=14)
	ax.set_xticks(k_values)
	ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.show()

def plot_dataset_comparison_scatter(df: pd.DataFrame, metric: str = 'top1_avg', split: str = 'test'):
	"""
	Create a scatter plot comparing model performance between ASL100 and ASL300.
	"""
	# Prepare data
	asl100_df = df[(df['dataset'] == 'asl100') & (df['split'] == split)][['model', metric]]
	asl300_df = df[(df['dataset'] == 'asl300') & (df['split'] == split)][['model', metric]]
	
	# Merge on model name
	merged = pd.merge(asl100_df, asl300_df, on='model', suffixes=('_100', '_300'))
	
	# Create figure
	fig, ax = plt.subplots(figsize=(8, 8))
	
	# Scatter plot
	ax.scatter(merged[f'{metric}_100'] * 100, merged[f'{metric}_300'] * 100, s=100, alpha=0.7)
	
	# Add model labels
	for idx, row in merged.iterrows():
		ax.annotate(row['model'], 
				   (row[f'{metric}_100'] * 100, row[f'{metric}_300'] * 100),
				   xytext=(5, 5), textcoords='offset points', fontsize=9)
	
	# Add diagonal line
	lims = [
		np.min([ax.get_xlim(), ax.get_ylim()]),
		np.max([ax.get_xlim(), ax.get_ylim()]),
	]
	ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
	
	# Customize plot
	metric_name = metric.replace('_', ' ').title().replace('Avg', '(Avg per Class)').replace('Inst', '(Per Instance)')
	ax.set_xlabel(f'ASL100 {metric_name} (%)', fontsize=12)
	ax.set_ylabel(f'ASL300 {metric_name} (%)', fontsize=12)
	ax.set_title(f'Model Performance: ASL100 vs ASL300 ({split.capitalize()} Set)', fontsize=14)
	ax.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.show()





# Example usage
def run_visualizations(filepath: str,
					   splits: List,
					   formatter=None,
					   extractor:Optional[Callable]=None, 
					   hm:bool=False,
					   sc:bool=False,
					   tb:bool=False):
	"""Run all visualizations."""
	# Load results
	results = load_results(filepath)
	
	if formatter:
		results = formatter(results)
	
	# Extract to DataFrame
	if extractor:
		df = extractor(results)
	else:
		df = extract_metrics_df(results)
	
	# 1. Model comparison bar plots
	for split in splits:
		plot_model_comparison_bar(df, dataset=split, metric_type='avg', split='test')
	
	# 2. Heatmap comparison
	if hm:
		plot_heatmap_comparison(df, metric='top1_avg', split='test')
	
	# 3. Top-K progression
	for split in splits:
		plot_top_k_progression(df, dataset=split, split='test', metric_type='avg')
	
	
	# 4. Dataset comparison scatter
	if sc:
		plot_dataset_comparison_scatter(df, metric='top1_avg', split='test')
	
	return df, results

# If running in Jupyter notebook
if __name__ == "__main__":
	# Assuming you have saved your JSON data to 'results.json'
	df, results = run_visualizations('./results/wlasl_satnac_16_only_summary.json', extractor=extract_metrics_df_sumed, splits=['asl100'])
	# with open('wlasl_100_stats_ref.json', 'r') as f:
	# 	asl100 = json.load(f)
	# with open('wlasl_300_stats_ref.json', 'r') as f:
	# 	asl300 = json.load(f)
	# table = compare_splits_latex_table(asl100, asl300)
	# print(table)
	
	# with open('wlasl_flipped_summary.json', 'r') as f:
	# 	res = json.load(f)
	# print(generate_latex_table(res, 'asl100', metric_type='avg', caption='Flipped results', label='wlasl_results_flipped'))
	# print(generate_latex_table(res, 'asl300', metric_type='avg', caption='Flipped results', label='wlasl_results_flipped'))
	# print()
 
	
