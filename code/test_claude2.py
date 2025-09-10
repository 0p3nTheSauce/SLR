import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics_df(results: Dict, condition_name: str = "normal") -> pd.DataFrame:
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
                        'condition': condition_name,
                        'top1_avg': metrics['top_k_average_per_class_acc']['top1'],
                        'top5_avg': metrics['top_k_average_per_class_acc']['top5'],
                        'top10_avg': metrics['top_k_average_per_class_acc']['top10'],
                        'top1_inst': metrics['top_k_per_instance_acc']['top1'],
                        'top5_inst': metrics['top_k_per_instance_acc']['top5'],
                        'top10_inst': metrics['top_k_per_instance_acc']['top10'],
                    }
                    data.append(row)
    return pd.DataFrame(data)

def sum_conv(n_dict):
  #essentially convert it back to the old format 
  new_dict = {}
  for split in n_dict.keys():
    new_dict[split] = {}
    for arch in n_dict[split].keys():
      new_dict[split][arch] = {}
      new_dict[split][arch]['best_test'] = n_dict[split][arch]['test set']
  return new_dict

def combine_results(normal_filepath: str, flipped_filepath: str) -> pd.DataFrame:
    """
    Load and combine results from both normal and temporal-flipped experiments.
    
    Args:
        normal_filepath: Path to normal results JSON
        flipped_filepath: Path to temporal-flipped results JSON
    
    Returns:
        Combined DataFrame with both conditions
    """
    # Load both datasets
    normal_results = load_results(normal_filepath)
    flipped_results = load_results(flipped_filepath)
    flipped_results = sum_conv(flipped_results)
    # Extract metrics
    normal_df = extract_metrics_df(normal_results, "Normal")
    flipped_df = extract_metrics_df(flipped_results, "Temporal Flipped")
    
    # Combine DataFrames
    combined_df = pd.concat([normal_df, flipped_df], ignore_index=True)
    
    return combined_df

def plot_temporal_comparison_bar(df: pd.DataFrame, dataset: str = 'asl100', 
                                metric_type: str = 'avg', topk: str = 'top1',
                                split: str = 'test'):
    """
    Create a grouped bar plot comparing normal vs temporal-flipped results.
    
    Args:
        df: Combined DataFrame with both conditions
        dataset: 'asl100' or 'asl300'
        metric_type: 'avg' for average per class, 'inst' for per instance
        topk: 'top1', 'top5', or 'top10'
        split: 'val' or 'test'
    """
    # Filter data
    plot_df = df[(df['dataset'] == dataset) & (df['split'] == split)].copy()
    
    # Prepare data for plotting
    metric_col = f'{topk}_{metric_type}'
    
    # Pivot to have conditions as columns
    pivot_df = plot_df.pivot_table(
        index='model', 
        columns='condition', 
        values=metric_col,
        aggfunc='first'
    ).reset_index()
    
    # Sort by normal condition performance
    if 'Normal' in pivot_df.columns:
        pivot_df = pivot_df.sort_values('Normal', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Bar plot
    x = np.arange(len(pivot_df))
    width = 0.35
    
    conditions = [col for col in pivot_df.columns if col != 'model']
    colors = ['#1f77b4', '#ff7f0e']  # Blue for normal, orange for flipped
    
    for i, condition in enumerate(conditions):
        values = pivot_df[condition].values * 100
        bars = ax.bar(x + (i - 0.5) * width, values, width, 
                     label=condition, color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    
    title_type = 'Average per Class' if metric_type == 'avg' else 'Per Instance'
    topk_display = topk.replace('top', 'Top-')
    ax.set_title(f'{dataset.upper()} - {topk_display} {title_type} Accuracy ({split.capitalize()} Set)\nNormal vs Temporal Flipped', 
                fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df['model'].values, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add performance difference annotations
    if len(conditions) == 2:
        normal_col, flipped_col = conditions[0], conditions[1]
        for i, (_, row) in enumerate(pivot_df.iterrows()):
            diff = (row[flipped_col] - row[normal_col]) * 100
            color = 'green' if diff > 0 else 'red'
            ax.text(i, max(row[normal_col], row[flipped_col]) * 100 + 2,
                   f'{diff:+.1f}%', ha='center', va='bottom', 
                   color=color, fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return pivot_df

def plot_all_topk_comparison(df: pd.DataFrame, dataset: str = 'asl100', 
                            metric_type: str = 'avg', split: str = 'test'):
    """
    Create a comprehensive comparison showing Top-1, Top-5, and Top-10 accuracies.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    topk_metrics = ['top1', 'top5', 'top10']
    
    for idx, topk in enumerate(topk_metrics):
        # Filter data
        plot_df = df[(df['dataset'] == dataset) & (df['split'] == split)].copy()
        
        # Prepare data for plotting
        metric_col = f'{topk}_{metric_type}'
        
        # Pivot to have conditions as columns
        pivot_df = plot_df.pivot_table(
            index='model', 
            columns='condition', 
            values=metric_col,
            aggfunc='first'
        ).reset_index()
        
        # Sort by normal condition performance
        if 'Normal' in pivot_df.columns:
            pivot_df = pivot_df.sort_values('Normal', ascending=False)
        
        ax = axes[idx]
        
        # Bar plot
        x = np.arange(len(pivot_df))
        width = 0.35
        
        conditions = [col for col in pivot_df.columns if col != 'model']
        colors = ['#1f77b4', '#ff7f0e']  # Blue for normal, orange for flipped
        
        for i, condition in enumerate(conditions):
            values = pivot_df[condition].values * 100
            ax.bar(x + (i - 0.5) * width, values, width, 
                  label=condition, color=colors[i % len(colors)], alpha=0.8)
        
        # Customize subplot
        topk_display = topk.replace('top', 'Top-')
        ax.set_title(f'{topk_display} Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_df['model'].values, rotation=45, ha='right')
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    title_type = 'Average per Class' if metric_type == 'avg' else 'Per Instance'
    fig.suptitle(f'{dataset.upper()} - {title_type} Accuracy Comparison ({split.capitalize()} Set)\nNormal vs Temporal Flipped', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()

def plot_difference_heatmap(df: pd.DataFrame, metric_type: str = 'avg', split: str = 'test'):
    """
    Create a heatmap showing the performance difference (Flipped - Normal) for each model and dataset.
    """
    # Calculate differences for each metric
    results = []
    
    for dataset in df['dataset'].unique():
        for model in df['model'].unique():
            dataset_model_df = df[(df['dataset'] == dataset) & 
                                 (df['model'] == model) & 
                                 (df['split'] == split)]
            
            if len(dataset_model_df) == 2:  # Both conditions present
                normal_row = dataset_model_df[dataset_model_df['condition'] == 'Normal'].iloc[0]
                flipped_row = dataset_model_df[dataset_model_df['condition'] == 'Temporal Flipped'].iloc[0]
                
                for topk in ['top1', 'top5', 'top10']:
                    metric_col = f'{topk}_{metric_type}'
                    diff = (flipped_row[metric_col] - normal_row[metric_col]) * 100
                    
                    results.append({
                        'dataset': dataset,
                        'model': model,
                        'metric': topk.replace('top', 'Top-'),
                        'difference': diff
                    })
    
    if not results:
        print("No matching data found for comparison")
        return
    
    results_df = pd.DataFrame(results)
    
    # Create pivot table for heatmap
    for dataset in results_df['dataset'].unique():
        dataset_df = results_df[results_df['dataset'] == dataset]
        pivot_df = dataset_df.pivot(index='model', columns='metric', values='difference')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                    cbar_kws={'label': 'Accuracy Difference (%)'}, ax=ax)
        
        # Customize plot
        title_type = 'Average per Class' if metric_type == 'avg' else 'Per Instance'
        ax.set_title(f'{dataset.upper()} - Performance Difference\n(Temporal Flipped - Normal) - {title_type}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        plt.tight_layout()
        plt.show()

def generate_comparison_summary(df: pd.DataFrame, split: str = 'test') -> pd.DataFrame:
    """
    Generate a summary table comparing normal vs temporal flipped performance.
    """
    summary_data = []
    
    for dataset in df['dataset'].unique():
        for metric_type in ['avg', 'inst']:
            for topk in ['top1', 'top5', 'top10']:
                metric_col = f'{topk}_{metric_type}'
                
                dataset_df = df[(df['dataset'] == dataset) & (df['split'] == split)]
                
                normal_mean = dataset_df[dataset_df['condition'] == 'Normal'][metric_col].mean()
                flipped_mean = dataset_df[dataset_df['condition'] == 'Temporal Flipped'][metric_col].mean()
                
                if pd.notna(normal_mean) and pd.notna(flipped_mean):
                    diff = (flipped_mean - normal_mean) * 100
                    
                    summary_data.append({
                        'Dataset': dataset.upper(),
                        'Metric Type': 'Avg per Class' if metric_type == 'avg' else 'Per Instance',
                        'Top-K': topk.replace('top', 'Top-'),
                        'Normal (%)': f'{normal_mean * 100:.1f}',
                        'Temporal Flipped (%)': f'{flipped_mean * 100:.1f}',
                        'Difference (%)': f'{diff:+.1f}'
                    })
    
    return pd.DataFrame(summary_data)

def run_temporal_comparison(normal_filepath: str, flipped_filepath: str):
    """
    Run complete comparison between normal and temporal-flipped results.
    
    Args:
        normal_filepath: Path to normal results JSON
        flipped_filepath: Path to temporal-flipped results JSON
    """
    print("Loading and combining results...")
    combined_df = combine_results(normal_filepath, flipped_filepath)
    
    print("Creating visualizations...\n")
    
    # 1. Individual Top-1 comparisons for both datasets
    print("1. Top-1 Accuracy Comparisons:")
    plot_temporal_comparison_bar(combined_df, dataset='asl100', metric_type='avg', 
                               topk='top1', split='test')
    plot_temporal_comparison_bar(combined_df, dataset='asl300', metric_type='avg', 
                               topk='top1', split='test')
    
    # 2. Comprehensive Top-K comparisons
    print("2. Comprehensive Top-K Comparisons:")
    plot_all_topk_comparison(combined_df, dataset='asl100', metric_type='avg', split='test')
    plot_all_topk_comparison(combined_df, dataset='asl300', metric_type='avg', split='test')
    
    # 3. Difference heatmaps
    print("3. Performance Difference Heatmaps:")
    plot_difference_heatmap(combined_df, metric_type='avg', split='test')
    
    # 4. Summary table
    print("4. Summary Statistics:")
    summary_df = generate_comparison_summary(combined_df, split='test')
    print(summary_df.to_string(index=False))
    
    return combined_df, summary_df

# Example usage:
if __name__ == "__main__":
    # Replace with your actual file paths
    normal_file = "normal_results.json"
    flipped_file = "temporal_flipped_results.json"
    
    # Run the complete comparison
    # combined_df, summary = run_temporal_comparison(normal_file, flipped_file)
    
    # You can also run individual comparisons:
    # combined_df = combine_results(normal_file, flipped_file)
    # plot_temporal_comparison_bar(combined_df, dataset='asl100', topk='top1')
    pass