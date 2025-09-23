#credit to claude for initial version of this code
# Visualizations for model performance comparison

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
	
	if tb:
		for split in splits:
			# 5. Generate LaTeX tables
			latex = generate_latex_table(
				results, 
				dataset=split, 
				split='test',
				metric_type='both',
				caption=f'{split} Test Set Results (\\% Accuracy)',
				label=f'tab:{split}_results'
			)
			print(f"{split} LaTeX Table:\n")
			print(latex)
	
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
 
	
