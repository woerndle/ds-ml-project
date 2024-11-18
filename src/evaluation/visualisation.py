# src/evaluation/visualisation.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_classification_report(report_str):
    """Parse the classification report string into a dictionary."""
    lines = report_str.strip().split('\n')
    metrics = {}
    for line in lines:
        line = line.strip()
        if line.startswith(('accuracy', 'macro avg', 'weighted avg')):
            parts = line.split()
            if line.startswith('accuracy'):
                metrics['accuracy'] = float(parts[-2])
            else:
                avg_type = parts[0] + ' ' + parts[1]
                metrics[f'{avg_type}_precision'] = float(parts[2])
                metrics[f'{avg_type}_recall'] = float(parts[3])
                metrics[f'{avg_type}_f1-score'] = float(parts[4])
    return metrics

def collect_metrics(output_dir):
    """Collect metrics from JSON files in the output_results directory."""
    results = []
    datasets = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    for dataset in datasets:
        dataset_dir = os.path.join(output_dir, dataset)
        models = [m for m in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, m))]
        for model in models:
            model_dir = os.path.join(dataset_dir, model)
            metrics_file = os.path.join(model_dir, f'{model}_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    # Extract metrics
                    metrics = {'dataset': dataset, 'model': model}
                    # Accuracy
                    metrics['accuracy'] = metrics_data.get('accuracy', None)
                    # Parse classification report
                    report_str = metrics_data.get('classification_report', '')
                    report_metrics = parse_classification_report(report_str)
                    metrics.update(report_metrics)
                    results.append(metrics)
            else:
                print(f"Metrics file not found for model {model} in dataset {dataset}.")
    return pd.DataFrame(results)

def plot_metrics(df):
    """Plot comparative bar charts for each metric across datasets and models."""
    # Metrics to plot
    metrics_to_plot = ['accuracy', 'macro avg_precision', 'macro avg_recall',
                       'macro avg_f1-score', 'weighted avg_precision',
                       'weighted avg_recall', 'weighted avg_f1-score']
    
    # Create output directory for plots
    plots_dir = os.path.join('output_results', 'visualisations')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set(style='whitegrid')

    for metric in metrics_to_plot:
        if metric not in df.columns:
            continue  # Skip metrics not present in the data

        # Create a figure with subplots for each dataset
        datasets = df['dataset'].unique()
        num_datasets = len(datasets)
        fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 6), sharey=True)

        if num_datasets == 1:
            axes = [axes]  # Make it iterable

        for ax, dataset in zip(axes, datasets):
            subset = df[df['dataset'] == dataset]
            # Sort models by metric value
            subset = subset.sort_values(by=metric, ascending=False)
            sns.barplot(x='model', y=metric, data=subset, ax=ax, palette='viridis')
            ax.set_title(f'{metric} on {dataset}')
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.capitalize())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Annotate bars with metric values
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 5),
                            textcoords='offset points')

        plt.tight_layout()
        # Save the plot
        plot_file = os.path.join(plots_dir, f'{metric}_comparison.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Plot saved: {plot_file}")

def main():
    output_dir = 'output_results'
    # Collect metrics from JSON files
    df = collect_metrics(output_dir)
    if df.empty:
        print("No metrics found to plot.")
        return

    # Plot the metrics
    plot_metrics(df)

if __name__ == '__main__':
    main()
