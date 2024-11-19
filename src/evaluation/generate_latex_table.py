import os
import json
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_classifier_family(model_name):
    """Extract the classifier family from the model name."""
    if model_name.startswith('KNN') or model_name.startswith('kNN') or 'KNeighbors' in model_name:
        return 'kNN'
    elif model_name.startswith('RF') or 'RandomForest' in model_name:
        return 'RF'
    elif model_name.startswith('SVC') or 'SVM' in model_name or 'LinearSVC' in model_name:
        return 'SVM'
    else:
        return 'Other'


def extract_metric_from_report(report, metric_name):
    """Extract average metric value from a classification report."""
    try:
        lines = report.strip().split('\n')
        for line in lines:
            if 'weighted avg' in line:
                parts = line.strip().split()
                if len(parts) >= 5:
                    if metric_name == 'precision':
                        return float(parts[-4])
                    elif metric_name == 'recall':
                        return float(parts[-3])
                    elif metric_name == 'f1-score':
                        return float(parts[-2])
            elif 'accuracy' in line and metric_name == 'accuracy':
                parts = line.strip().split()
                return float(parts[-1])
    except Exception as e:
        logging.warning(f"Failed to extract {metric_name} from report: {e}")
    return None


def load_results(results_dir):
    """Load results from JSON files into a DataFrame."""
    if not os.path.exists(results_dir):
        logging.error(f"Results directory '{results_dir}' does not exist.")
        return pd.DataFrame()

    all_results = []
    for dataset in os.listdir(results_dir):
        dataset_dir = os.path.join(results_dir, dataset)
        if os.path.isdir(dataset_dir):
            for model_dir in os.listdir(dataset_dir):
                model_path = os.path.join(dataset_dir, model_dir)
                if os.path.isdir(model_path):
                    metrics_file = os.path.join(model_path, f"{model_dir}_metrics.json")
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                                accuracy = metrics.get('accuracy', None)
                                classification_report = metrics.get('classification_report', '')
                                precision = extract_metric_from_report(classification_report, 'precision')
                                recall = extract_metric_from_report(classification_report, 'recall')
                                f1 = extract_metric_from_report(classification_report, 'f1-score')
                                model_name = metrics.get('model_name', model_dir)
                                classifier_family = get_classifier_family(model_name)
                                all_results.append({
                                    'Dataset': dataset,
                                    'Classifier': classifier_family,
                                    'Model': model_name,
                                    'Accuracy_mean': accuracy,
                                    'Accuracy_std': None,  # To be calculated later
                                    'Precision_mean': precision,
                                    'Precision_std': None,  # To be calculated later
                                    'Recall_mean': recall,
                                    'Recall_std': None,  # To be calculated later
                                    'F1_mean': f1,
                                    'F1_std': None  # To be calculated later
                                })
                        except Exception as e:
                            logging.warning(f"Failed to process metrics file '{metrics_file}': {e}")
    return pd.DataFrame(all_results)


def validate_numeric_columns(df, columns):
    """Ensure numeric columns contain valid data."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def process_results(df):
    """Process results to calculate mean and std for each metric."""
    df = validate_numeric_columns(df, ['Accuracy_mean', 'Precision_mean', 'Recall_mean', 'F1_mean'])

    # Group by Dataset and Classifier to calculate mean and std
    aggregated_df = df.groupby(['Dataset', 'Classifier']).agg(
        Accuracy_mean=('Accuracy_mean', 'mean'),
        Accuracy_std=('Accuracy_mean', 'std'),
        Precision_mean=('Precision_mean', 'mean'),
        Precision_std=('Precision_mean', 'std'),
        Recall_mean=('Recall_mean', 'mean'),
        Recall_std=('Recall_mean', 'std'),
        F1_mean=('F1_mean', 'mean'),
        F1_std=('F1_mean', 'std')
    ).reset_index()

    return aggregated_df


def generate_latex_table(df, caption, label):
    """Generate LaTeX table code from DataFrame."""
    latex_table = '\\begin{table}[h]\n'
    latex_table += f'\\caption{{{caption}}}\n'
    latex_table += f'\\label{{{label}}}\n'
    latex_table += '\\centering\n'
    latex_table += '\\begin{tabular}{llllll}\n'  # 6 columns
    latex_table += '\\hline\n'
    latex_table += '\\textbf{Dataset} & \\textbf{Classifier} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\\n'
    latex_table += '\\hline\n'

    # Group by Dataset
    grouped = df.groupby('Dataset')
    for dataset, group in grouped:
        num_classifiers = len(group)
        dataset_escaped = dataset.replace('_', '\\_')  # Escape underscores for LaTeX
        classifiers = group.sort_values(by='Accuracy_mean', ascending=False)  # Optional: sort classifiers

        for idx, row in classifiers.iterrows():
            classifier = row['Classifier']
            accuracy = f"{row['Accuracy_mean']:.2f} $\\pm$ {row['Accuracy_std']:.2f}" if pd.notnull(row['Accuracy_std']) else f"{row['Accuracy_mean']:.2f}"
            precision = f"{row['Precision_mean']:.2f} $\\pm$ {row['Precision_std']:.2f}" if pd.notnull(row['Precision_std']) else f"{row['Precision_mean']:.2f}"
            recall = f"{row['Recall_mean']:.2f} $\\pm$ {row['Recall_std']:.2f}" if pd.notnull(row['Recall_std']) else f"{row['Recall_mean']:.2f}"
            f1 = f"{row['F1_mean']:.2f} $\\pm$ {row['F1_std']:.2f}" if pd.notnull(row['F1_std']) else f"{row['F1_mean']:.2f}"

            if idx == group.index[0]:
                # First classifier for this dataset: include multirow for Dataset
                latex_table += f'\\multirow{{{num_classifiers}}}{{*}}{{{dataset_escaped}}} & {classifier} & {accuracy} & {precision} & {recall} & {f1} \\\\\n'
            else:
                # Subsequent classifiers: leave Dataset column empty
                latex_table += f' & {classifier} & {accuracy} & {precision} & {recall} & {f1} \\\\\n'
        latex_table += '\\hline\n'

    latex_table += '\\end{tabular}\n'
    latex_table += '\\end{table}\n'
    return latex_table


def process_runtime_memory(results_dir):
    """Extract runtime and memory usage metrics grouped by classifier family."""
    runtime_memory_results = []
    for dataset in os.listdir(results_dir):
        dataset_dir = os.path.join(results_dir, dataset)
        if os.path.isdir(dataset_dir):
            for model_dir in os.listdir(dataset_dir):
                model_path = os.path.join(dataset_dir, model_dir)
                if os.path.isdir(model_path):
                    metrics_file = os.path.join(model_path, f"{model_dir}_metrics.json")
                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics = json.load(f)
                                runtime = metrics.get('elapsed_time_seconds', None)
                                memory = metrics.get('peak_memory_usage_mb', None)
                                model_name = metrics.get('model_name', model_dir)
                                classifier_family = get_classifier_family(model_name)
                                runtime_memory_results.append({
                                    'Dataset': dataset,
                                    'Classifier_Family': classifier_family,
                                    'Model': model_name,
                                    'Runtime': runtime,
                                    'Memory_Usage_MB': memory
                                })
                        except Exception as e:
                            logging.warning(f"Failed to process metrics file '{metrics_file}': {e}")
    return pd.DataFrame(runtime_memory_results)


def generate_runtime_memory_table(df, caption, label):
    """Generate LaTeX table for runtime and memory usage grouped by classifier family."""
    # First, calculate mean and std for Runtime and Memory Usage
    aggregated_df = df.groupby(['Dataset', 'Classifier_Family']).agg(
        Runtime_mean=('Runtime', 'mean'),
        Runtime_std=('Runtime', 'std'),
        Memory_mean=('Memory_Usage_MB', 'mean'),
        Memory_std=('Memory_Usage_MB', 'std')
    ).reset_index()

    latex_table = '\\begin{table}[h]\n'
    latex_table += f'\\caption{{{caption}}}\n'
    latex_table += f'\\label{{{label}}}\n'
    latex_table += '\\centering\n'
    latex_table += '\\begin{tabular}{llll}\n'  # 4 columns
    latex_table += '\\hline\n'
    latex_table += '\\textbf{Dataset} & \\textbf{Classifier Family} & \\textbf{Runtime (s)} & \\textbf{Memory Usage (MB)} \\\\\n'
    latex_table += '\\hline\n'

    # Group by Dataset
    grouped = aggregated_df.groupby('Dataset')
    for dataset, group in grouped:
        num_families = len(group)
        dataset_escaped = dataset.replace('_', '\\_')  # Escape underscores for LaTeX

        for idx, row in group.iterrows():
            classifier_family = row['Classifier_Family']
            runtime = f"{row['Runtime_mean']:.2f} $\\pm$ {row['Runtime_std']:.2f}" if pd.notnull(row['Runtime_std']) else f"{row['Runtime_mean']:.2f}"
            memory = f"{row['Memory_mean']:.2f} $\\pm$ {row['Memory_std']:.2f}" if pd.notnull(row['Memory_std']) else f"{row['Memory_mean']:.2f}"

            if idx == group.index[0]:
                # First classifier family for this dataset: include multirow for Dataset
                latex_table += f'\\multirow{{{num_families}}}{{*}}{{{dataset_escaped}}} & {classifier_family} & {runtime} & {memory} \\\\\n'
            else:
                # Subsequent classifier families: leave Dataset column empty
                latex_table += f' & {classifier_family} & {runtime} & {memory} \\\\\n'
        latex_table += '\\hline\n'

    latex_table += '\\end{tabular}\n'
    latex_table += '\\end{table}\n'
    return latex_table


def main():
    # Load and process classification metrics
    results_dir = 'output_results_holdout'  # Adjust this path if needed
    metrics_df = load_results(results_dir)
    aggregated_metrics_df = process_results(metrics_df)

    # Generate and save LaTeX tables for classification metrics
    classification_table = generate_latex_table(aggregated_metrics_df, 'Comprehensive Performance Results', 'tab:comprehensive_results')
    with open('comprehensive_results_table.tex', 'w') as f:
        f.write(classification_table)

    # Process runtime and memory usage results
    runtime_memory_df = process_runtime_memory(results_dir)
    runtime_memory_table = generate_runtime_memory_table(runtime_memory_df, 'Runtime and Memory Usage', 'tab:runtime_memory')
    with open('runtime_memory_table.tex', 'w') as f:
        f.write(runtime_memory_table)

    print("LaTeX tables generated and saved as 'comprehensive_results_table.tex' and 'runtime_memory_table.tex'.")


if __name__ == "__main__":
    main()
