# src/evaluation/generate_tables.py

import os
import json
import pandas as pd
import numpy as np

def get_classifier_family(model_name):
    """Extract the classifier family from the model name."""
    if model_name.startswith('KNN') or model_name.startswith('kNN') or 'KNeighbors' in model_name:
        return 'kNN'
    elif model_name.startswith('RF') or 'RandomForest' in model_name:
        return 'RF'
    elif model_name.startswith('SVC') or 'SVC' in model_name or 'SVM' in model_name or 'LinearSVC' in model_name:
        return 'SVM'
    else:
        return 'Other'

def extract_metric_from_report(report, metric_name):
    """Extract average metric value from classification report."""
    lines = report.strip().split('\n')
    for line in lines:
        if 'weighted avg' in line:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Adjust index based on whether class labels are numeric or strings
                if metric_name == 'precision':
                    return float(parts[-4])
                elif metric_name == 'recall':
                    return float(parts[-3])
                elif metric_name == 'f1-score':
                    return float(parts[-2])
        elif 'accuracy' in line and metric_name == 'accuracy':
            parts = line.strip().split()
            return float(parts[-1])
    return None

def load_results(results_dir):
    """Load results from JSON files into a DataFrame."""
    all_results = []
    for dataset in os.listdir(results_dir):
        dataset_dir = os.path.join(results_dir, dataset)
        if os.path.isdir(dataset_dir):
            for model_dir in os.listdir(dataset_dir):
                model_path = os.path.join(dataset_dir, model_dir)
                if os.path.isdir(model_path):
                    metrics_file = os.path.join(model_path, f"{model_dir}_metrics.json")
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            accuracy = metrics.get('accuracy', None)
                            classification_report = metrics.get('classification_report', '')
                            precision = extract_metric_from_report(classification_report, 'precision')
                            recall = extract_metric_from_report(classification_report, 'recall')
                            f1 = extract_metric_from_report(classification_report, 'f1-score')
                            # Use model_name from metrics if available
                            model_name = metrics.get('model_name', model_dir)
                            classifier_family = get_classifier_family(model_name)
                            all_results.append({
                                'Dataset': dataset,
                                'Classifier': classifier_family,
                                'Model': model_name,
                                'Accuracy': accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1': f1
                            })
    return pd.DataFrame(all_results)

def process_results(df):
    """Process results to get top 3 and bottom 3 models per classifier family and dataset."""
    top_results = []
    bottom_results = []

    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        for classifier_family in ['SVM', 'kNN', 'RF']:
            classifier_df = dataset_df[dataset_df['Classifier'] == classifier_family]
            if not classifier_df.empty:
                # Sort by accuracy
                sorted_df = classifier_df.sort_values(by='Accuracy', ascending=False)
                # Select top 3 models
                top_models = sorted_df.head(3)
                top_mean = top_models[['Accuracy', 'Precision', 'Recall', 'F1']].mean()
                top_std = top_models[['Accuracy', 'Precision', 'Recall', 'F1']].std()
                top_results.append({
                    'Dataset': dataset,
                    'Classifier': classifier_family,
                    'Accuracy_mean': top_mean['Accuracy'],
                    'Accuracy_std': top_std['Accuracy'],
                    'Precision_mean': top_mean['Precision'],
                    'Precision_std': top_std['Precision'],
                    'Recall_mean': top_mean['Recall'],
                    'Recall_std': top_std['Recall'],
                    'F1_mean': top_mean['F1'],
                    'F1_std': top_std['F1']
                })
                # Select bottom 3 models
                bottom_models = sorted_df.tail(3)
                bottom_mean = bottom_models[['Accuracy', 'Precision', 'Recall', 'F1']].mean()
                bottom_std = bottom_models[['Accuracy', 'Precision', 'Recall', 'F1']].std()
                bottom_results.append({
                    'Dataset': dataset,
                    'Classifier': classifier_family,
                    'Accuracy_mean': bottom_mean['Accuracy'],
                    'Accuracy_std': bottom_std['Accuracy'],
                    'Precision_mean': bottom_mean['Precision'],
                    'Precision_std': bottom_std['Precision'],
                    'Recall_mean': bottom_mean['Recall'],
                    'Recall_std': bottom_std['Recall'],
                    'F1_mean': bottom_mean['F1'],
                    'F1_std': bottom_std['F1']
                })
    return pd.DataFrame(top_results), pd.DataFrame(bottom_results)

def generate_latex_table(df, caption, label):
    """Generate LaTeX table code from DataFrame."""
    latex_table = '\\begin{table}[h]\n'
    latex_table += f'\\caption{{{caption}}}\n'
    latex_table += f'\\label{{{label}}}\n'
    latex_table += '\\centering\n'
    latex_table += '\\begin{tabular}{llllll}\n'
    latex_table += '\\hline\n'
    latex_table += '\\textbf{Dataset} & \\textbf{Classifier} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\\n'
    latex_table += '\\hline\n'

    grouped = df.groupby('Dataset')
    for dataset, group in grouped:
        dataset_escaped = dataset.replace('_', '\\_')
        num_classifiers = len(group)
        first = True
        for _, row in group.iterrows():
            classifier = row['Classifier']
            accuracy = f"${row['Accuracy_mean']:.2f} \\pm {row['Accuracy_std']:.2f}$"
            precision = f"${row['Precision_mean']:.2f} \\pm {row['Precision_std']:.2f}$"
            recall = f"${row['Recall_mean']:.2f} \\pm {row['Recall_std']:.2f}$"
            f1 = f"${row['F1_mean']:.2f} \\pm {row['F1_std']:.2f}$"
            if first:
                latex_table += f'\\multirow{{{num_classifiers}}}{{*}}{{{dataset_escaped}}} & {classifier} & {accuracy} & {precision} & {recall} & {f1} \\\\\n'
                first = False
            else:
                latex_table += f' & {classifier} & {accuracy} & {precision} & {recall} & {f1} \\\\\n'
        latex_table += '\\hline\n'
    latex_table += '\\end{tabular}\n'
    latex_table += '\\end{table}\n'
    return latex_table

def main():
    # Load results
    results_dir = 'output_results'  # Adjust this path if needed
    df = load_results(results_dir)

    # Process results to get top and bottom models
    top_df, bottom_df = process_results(df)

    # Generate LaTeX tables
    top_table = generate_latex_table(top_df, 'Top Performing Models', 'tab:top_models')
    bottom_table = generate_latex_table(bottom_df, 'Bottom Performing Models', 'tab:bottom_models')

    # Save LaTeX tables to files
    with open('top_models_table.tex', 'w') as f:
        f.write(top_table)
    with open('bottom_models_table.tex', 'w') as f:
        f.write(bottom_table)

    print("LaTeX tables generated and saved to 'top_models_table.tex' and 'bottom_models_table.tex'.")

if __name__ == "__main__":
    main()
