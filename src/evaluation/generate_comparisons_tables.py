# src/evaluation/generate_comparison_tables.py

import os
import json
import pandas as pd
import numpy as np

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

def load_results(results_dir, method_name, dataset_sample_sizes):
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
                            # Get additional metrics if available
                            time_taken = metrics.get('elapsed_time_seconds', None)  # In seconds
                            memory_usage_mb = metrics.get('peak_memory_usage_mb', None)  # In MB
                            # Convert memory usage to GB
                            memory_usage_gb = memory_usage_mb / 1024 if memory_usage_mb is not None else None
                            # Get number of samples from mapping
                            num_samples = dataset_sample_sizes.get(dataset, None)
                            all_results.append({
                                'Dataset': dataset,
                                'Method': method_name,
                                'Classifier': classifier_family,
                                'Model': model_name,
                                'Accuracy': accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1': f1,
                                'Time_Taken': time_taken,
                                'Memory_Usage_GB': memory_usage_gb,
                                'Num_Samples': num_samples
                            })
    return pd.DataFrame(all_results)

def categorize_dataset_size(num_samples):
    """Categorize dataset size based on number of samples."""
    if num_samples is None:
        return 'Unknown'
    elif num_samples < 1000:
        return 'Small ($<$1K)'
    elif 1000 <= num_samples <= 10000:
        return 'Medium (1K-10K)'
    else:
        return 'Large ($>$10K)'

def generate_validation_comparison_table(df_holdout, df_cv):
    """Generate LaTeX table comparing validation methods."""
    # Compute average time and standard deviation in accuracy for each method
    avg_time_holdout = df_holdout['Time_Taken'].mean()
    std_accuracy_holdout = df_holdout['Accuracy'].std() * 100  # Convert to percentage

    avg_time_cv = df_cv['Time_Taken'].mean()
    std_accuracy_cv = df_cv['Accuracy'].std() * 100  # Convert to percentage

    # Create LaTeX table
    latex_table = '\\begin{table}[htbp]\n'
    latex_table += '\\centering\n'
    latex_table += '\\caption{Validation Method Comparison}\n'
    latex_table += '\\label{tab:validation_comparison}\n'
    latex_table += '\\begin{tabular}{@{}lll@{}}\n'
    latex_table += '\\toprule\n'
    latex_table += 'Method & Avg. Time (s) & Std. Dev. in Accuracy (\\%) \\\\\n'
    latex_table += '\\midrule\n'
    latex_table += f'Holdout & {avg_time_holdout:.1f} & {std_accuracy_holdout:.1f}\\% \\\\\n'
    latex_table += f'5-fold CV & {avg_time_cv:.1f} & {std_accuracy_cv:.1f}\\% \\\\\n'
    latex_table += '\\bottomrule\n'
    latex_table += '\\end{tabular}\n'
    latex_table += '\\end{table}\n'

    return latex_table

def generate_memory_usage_table(df_combined):
    """Generate LaTeX table for peak memory usage by dataset size."""
    # We need to compute the peak memory usage for each classifier and dataset size category
    # First, ensure that Num_Samples and Memory_Usage_GB are available
    df = df_combined.dropna(subset=['Memory_Usage_GB', 'Num_Samples'])
    if df.empty:
        print("No memory usage data available.")
        return ""

    # Categorize dataset sizes
    df['Dataset_Size'] = df['Num_Samples'].apply(categorize_dataset_size)

    # Pivot table to get peak memory usage
    pivot_table = df.pivot_table(
        index='Dataset_Size',
        columns='Classifier',
        values='Memory_Usage_GB',
        aggfunc='max'
    ).reindex(['Small ($<$1K)', 'Medium (1K-10K)', 'Large ($>$10K)'])

    # Ensure classifiers are in the desired order
    classifiers = ['kNN', 'Random Forest', 'SVM']
    # Include only existing classifiers in the pivot table
    existing_classifiers = [clf for clf in classifiers if clf in pivot_table.columns]
    pivot_table = pivot_table[existing_classifiers]

    # Create LaTeX table
    latex_table = '\\begin{table}[htbp]\n'
    latex_table += '\\centering\n'
    latex_table += '\\caption{Peak Memory Usage (GB) by Dataset Size}\n'
    latex_table += '\\label{tab:memory_usage}\n'
    latex_table += '\\begin{tabular}{@{}llll@{}}\n'
    latex_table += '\\toprule\n'
    latex_table += 'Dataset Size & ' + ' & '.join(existing_classifiers) + ' \\\\\n'
    latex_table += '\\midrule\n'
    for dataset_size in ['Small ($<$1K)', 'Medium (1K-10K)', 'Large ($>$10K)']:
        if dataset_size not in pivot_table.index:
            continue  # Skip if the dataset size category is not present
        row = pivot_table.loc[dataset_size]
        memory_values = []
        for clf in existing_classifiers:
            memory = row[clf]
            memory_str = f"{memory:.1f}" if pd.notnull(memory) else 'N/A'
            memory_values.append(memory_str)
        latex_table += f'{dataset_size} & ' + ' & '.join(memory_values) + ' \\\\\n'
    latex_table += '\\bottomrule\n'
    latex_table += '\\end{tabular}\n'
    latex_table += '\\end{table}\n'

    return latex_table

def main():
    # Directories containing the results
    holdout_results_dir = 'output_results_holdout'  # Adjust this path if needed
    cv_results_dir = 'output_results_cv'      # Adjust this path if needed

    # Define mapping from dataset names to number of samples

    dataset_sample_sizes = {
        'amazon_reviews': 750,         
        'congressional_voting': 218,     
        'traffic_prediction': 3168,     
        'wine_reviews': 150930,                 
    }

    # Load results
    df_holdout = load_results(holdout_results_dir, method_name='Holdout', dataset_sample_sizes=dataset_sample_sizes)
    df_cv = load_results(cv_results_dir, method_name='5-fold CV', dataset_sample_sizes=dataset_sample_sizes)

    # Combine dataframes for memory usage analysis
    df_combined = pd.concat([df_holdout, df_cv], ignore_index=True)

    # Generate Validation Method Comparison Table
    validation_table = generate_validation_comparison_table(df_holdout, df_cv)
    with open('validation_comparison_table.tex', 'w') as f:
        f.write(validation_table)
    print("Validation Method Comparison Table generated and saved to 'validation_comparison_table.tex'.")

    # Generate Peak Memory Usage Table
    memory_usage_table = generate_memory_usage_table(df_combined)
    if memory_usage_table:
        with open('memory_usage_table.tex', 'w') as f:
            f.write(memory_usage_table)
        print("Peak Memory Usage Table generated and saved to 'memory_usage_table.tex'.")
    else:
        print("No memory usage data available to generate the Peak Memory Usage Table.")

if __name__ == "__main__":
    main()
