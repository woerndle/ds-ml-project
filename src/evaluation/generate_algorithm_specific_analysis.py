#!/usr/bin/env python3
# src/evaluation/generate_algorithm_specific_analysis.py

import os
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_classifier_family(model_name):
    """Extract the classifier family from the model name."""
    model_name_lower = model_name.lower()
    if 'knn' in model_name_lower or 'kneighbors' in model_name_lower:
        return 'kNN'
    elif 'rf' in model_name_lower or 'randomforest' in model_name_lower:
        return 'RF'
    elif 'svc' in model_name_lower or 'svm' in model_name_lower or 'linearsvc' in model_name_lower:
        return 'SVM'
    else:
        return 'Other'

def parse_hyperparameters(model_name, classifier_family):
    """Parse hyperparameters from the model name."""
    import re
    hyperparams_dict = {}

    if classifier_family == 'kNN':
        try:
            match = re.search(r'\((.*?)\)', model_name)
            if match:
                params = match.group(1).split(',')
                params = [param.strip() for param in params]
                if len(params) >= 4:
                    hyperparams_dict['weights'] = params[0]
                    hyperparams_dict['algorithm'] = params[1]
                    hyperparams_dict['metric'] = params[2]
                    k_param = params[3]
                    k_match = re.search(r'k=(\d+)', k_param)
                    hyperparams_dict['k'] = int(k_match.group(1)) if k_match else None
                else:
                    print(f"Unexpected number of hyperparameters in model name: {model_name}")
        except Exception as e:
            print(f"Error parsing kNN hyperparameters from model name '{model_name}': {e}")
    return hyperparams_dict

def filter_valid_data(df, required_columns):
    """Filter rows where required columns are not null."""
    return df.dropna(subset=required_columns)

def save_plot_with_directory_check(plot_filename):
    """Ensure the directory for the plot exists and save the plot."""
    Path(plot_filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot: {plot_filename}")

def plot_kNN_metrics(df_family, dataset_plot_dir):
    """Plot kNN metrics."""
    df_family = filter_valid_data(df_family, ['k', 'accuracy', 'metric', 'weights'])
    if df_family.empty:
        print("No valid data for kNN. Skipping kNN plots.")
        return

    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        data=df_family,
        x='k',
        y='accuracy',
        hue='metric',
        style='weights',
        size='k',
        sizes=(50, 200),
        alpha=0.7
    )
    plt.title(f'kNN Performance on {df_family["dataset"].iloc[0]}')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.legend(title='Metric & Weights', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_filename = dataset_plot_dir / 'kNN_accuracy_vs_k.png'
    save_plot_with_directory_check(plot_filename)

    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        data=df_family,
        x='peak_memory_usage_mb',
        y='accuracy',
        hue='metric',
        style='weights',
        size='k',
        sizes=(50, 200),
        alpha=0.7
    )
    plt.title(f'kNN Peak Memory Usage vs. Accuracy on {df_family["dataset"].iloc[0]}')
    plt.xlabel('Peak Memory Usage (MB)')
    plt.ylabel('Accuracy')
    plt.legend(title='Metric & Weights', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_filename_memory = dataset_plot_dir / 'kNN_memory_vs_accuracy.png'
    save_plot_with_directory_check(plot_filename_memory)

def load_all_metrics(output_dir, datasets):
    """Load metrics from JSON files for all models across multiple datasets."""
    all_data = []
    for dataset in datasets:
        dataset_dir = Path(output_dir) / dataset
        if not dataset_dir.exists():
            print(f"Dataset directory {dataset_dir} does not exist. Skipping.")
            continue
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            metrics_file = model_dir / f"{model_name}_metrics.json"
            if not metrics_file.exists():
                print(f"Metrics file for model '{model_name}' not found in {model_dir}. Skipping.")
                continue
            with open(metrics_file, 'r') as f:
                try:
                    metrics = json.load(f)
                except json.JSONDecodeError:
                    print(f"Invalid JSON format in file: {metrics_file}. Skipping.")
                    continue
            classifier_family = get_classifier_family(model_name)
            hyperparams_dict = parse_hyperparameters(model_name, classifier_family)
            combined = {
                'dataset': dataset,
                'model_name': model_name,
                'classifier_family': classifier_family,
                'accuracy': metrics.get('accuracy'),
                'elapsed_time_seconds': metrics.get('elapsed_time_seconds'),
                'peak_memory_usage_mb': metrics.get('peak_memory_usage_mb')
            }
            combined.update(hyperparams_dict)
            all_data.append(combined)
    return pd.DataFrame(all_data)

def plot_metrics(df, plots_output_dir):
    """Generate and save plots for all datasets and classifier families."""
    sns.set_theme(style="whitegrid")
    os.makedirs(plots_output_dir, exist_ok=True)
    datasets = df['dataset'].unique()
    classifier_families = ['kNN']
    for dataset in datasets:
        df_dataset = df[df['dataset'] == dataset]
        for family in classifier_families:
            df_family = df_dataset[df_dataset['classifier_family'] == family]
            if family == 'kNN':
                plot_kNN_metrics(df_family, Path(plots_output_dir) / dataset)

def main():
    datasets = ['amazon_reviews', 'congressional_voting', 'traffic_prediction', 'wine_reviews']
    output_results_dir = 'output_results_cv_mm'
    plots_output_dir = 'visualizations'
    df = load_all_metrics(output_results_dir, datasets)
    if df.empty:
        print("No data loaded. Exiting.")
        return
    combined_df_path = Path(plots_output_dir) / "combined_metrics.csv"
    os.makedirs(plots_output_dir, exist_ok=True)
    df.to_csv(combined_df_path, index=False)
    print(f"Combined metrics saved to {combined_df_path}")
    plot_metrics(df, plots_output_dir)

if __name__ == "__main__":
    main()
