import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_model_name(model_name):
    """
    Extracts the classifier family and hyperparameters from the model name.
    """
    # First, extract the classifier family and parameters
    pattern = r'^(.*?)\s*\((.*)\)$'
    match = re.match(pattern, model_name)
    if match:
        family = match.group(1).strip()
        params_str = match.group(2)
        # Now, extract the hyperparameters
        params_list = re.split(r',\s*', params_str)
        hyperparameters = {}
        for param in params_list:
            if '=' in param:
                key, value = param.split('=', 1)
                hyperparameters[key.strip()] = value.strip()
            else:
                # For positional parameters (e.g., KNN), store as positional arguments
                if 'positional' not in hyperparameters:
                    hyperparameters['positional'] = []
                hyperparameters['positional'].append(param.strip())
    else:
        # Model name without parameters
        family = model_name.strip()
        hyperparameters = {}
    return family, hyperparameters

def generate_abbreviated_model_name(model_name):
    """
    Generates an abbreviated model name based on the full model name.
    """
    family, hyperparameters = parse_model_name(model_name)

    # Now, generate abbreviation based on classifier
    if family == 'SVC':
        kernel = hyperparameters.get('kernel', '')
        C = hyperparameters.get('C', '')
        if kernel == 'poly':
            degree = hyperparameters.get('degree', '')
            abbrev = f"SVC-ply-d{degree}-C{C}"
        else:
            abbrev = f"SVC-{kernel[:3]}-C{C}"
    elif family == 'RF':
        n_estimators = hyperparameters.get('n_estimators', '')
        max_depth = hyperparameters.get('max_depth', '')
        abbrev = f"RF-n{n_estimators}-d{max_depth}"
    elif family == 'KNN':
        positional_params = hyperparameters.get('positional', [])
        if len(positional_params) >= 3:
            weights = positional_params[0][:1]  # First letter of weights
            algorithm = positional_params[1][:1]  # First letter of algorithm
            metric = positional_params[2][:1]  # First letter of metric
            k = hyperparameters.get('k', '')
            abbrev = f"KNN-{weights}{algorithm}{metric}-k{k}"
        else:
            abbrev = family
    else:
        abbrev = family
    return abbrev

def load_metrics(file_path):
    """
    Loads the metrics from a JSON file.
    """
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def collect_holdout_data(root_dir):
    data = []
    for dataset in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for model_folder in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_folder)
            if not os.path.isdir(model_path):
                continue
            metrics_file = os.path.join(model_path, f"{model_folder}_metrics.json")
            if os.path.exists(metrics_file):
                try:
                    metrics = load_metrics(metrics_file)
                    family, hyperparams = parse_model_name(model_folder)
                    entry = {
                        'Method': 'Holdout',
                        'Dataset': dataset,
                        'Classifier': family,
                        'Model_Name': model_folder,  # Include the full model name
                        'Hyperparameters': hyperparams,
                        'Accuracy': metrics['accuracy'],
                        'F1_Score': metrics['classification_report']['weighted avg']['f1-score'],
                        'Elapsed_Time': metrics['elapsed_time_seconds'],
                        'Memory_Usage': metrics['peak_memory_usage_mb']
                    }
                    data.append(entry)
                except Exception as e:
                    print(f"Error processing {metrics_file}: {e}")
    return data

def collect_crossval_data(root_dir):
    data = []
    for dataset in os.listdir(root_dir):
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for model_folder in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_folder)
            if not os.path.isdir(model_path):
                continue
            folds = []
            for fold in os.listdir(model_path):
                fold_path = os.path.join(model_path, fold)
                if os.path.isdir(fold_path):
                    metrics_file = os.path.join(fold_path, f"{model_folder}_metrics.json")
                    if os.path.exists(metrics_file):
                        try:
                            metrics = load_metrics(metrics_file)
                            folds.append(metrics)
                        except Exception as e:
                            print(f"Error processing {metrics_file}: {e}")
            if folds:
                accuracy = np.mean([f['accuracy'] for f in folds])
                f1_scores = np.mean([f['classification_report']['weighted avg']['f1-score'] for f in folds])
                elapsed_time = np.mean([f['elapsed_time_seconds'] for f in folds])
                memory_usage = np.mean([f['peak_memory_usage_mb'] for f in folds])
                family, hyperparams = parse_model_name(model_folder)
                entry = {
                    'Method': 'Cross-Validation',
                    'Dataset': dataset,
                    'Classifier': family,
                    'Model_Name': model_folder,  # Include the full model name
                    'Hyperparameters': hyperparams,
                    'Accuracy': accuracy,
                    'F1_Score': f1_scores,
                    'Elapsed_Time': elapsed_time,
                    'Memory_Usage': memory_usage
                }
                data.append(entry)
    return data

if __name__ == "__main__":
    # Paths to the results directories
    cross_val_dir = 'output_results_cross_val'
    holdout_dir = 'output_results_holdout'

    # Collect data from holdout results
    holdout_data = collect_holdout_data(holdout_dir)

    # Collect data from cross-validation results
    crossval_data = collect_crossval_data(cross_val_dir)

    # Combine all data
    all_data = holdout_data + crossval_data

    # Create a DataFrame
    df = pd.DataFrame(all_data)

    # Extract hyperparameters into separate columns
    hyperparams_df = pd.json_normalize(df['Hyperparameters'])
    df = pd.concat([df, hyperparams_df], axis=1).drop(columns=['Hyperparameters'])

    # Handle missing values if any
    df.fillna(method='ffill', inplace=True)

    # Generate abbreviated model names
    df['Abbrev_Model_Name'] = df['Model_Name'].apply(generate_abbreviated_model_name)

    # Group by Method, Dataset, Classifier
    summary = df.groupby(['Method', 'Dataset', 'Classifier']).agg(
        Mean_Accuracy=('Accuracy', 'mean'),
        Std_Accuracy=('Accuracy', 'std'),
        Mean_F1_Score=('F1_Score', 'mean'),
        Std_F1_Score=('F1_Score', 'std'),
        Mean_Elapsed_Time=('Elapsed_Time', 'mean'),
        Mean_Memory_Usage=('Memory_Usage', 'mean')
    ).reset_index()

    # Create output directory for plots
    output_plot_dir = 'plots'
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)

    # Plot 1: Box Plots of Accuracy across Datasets and Methods
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Dataset', y='Accuracy', hue='Method', data=df)
    plt.title('Accuracy Across Datasets and Methods')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.legend(title='Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, 'accuracy_boxplot.png'))
    plt.show()

    # Plot 1.5: Box Plots of F1_Score across Datasets and Methods
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Dataset', y='F1_Score', hue='Method', data=df)
    plt.title('F1_Score Across Datasets and Methods')
    plt.xlabel('Dataset')
    plt.ylabel('F1_Score')
    plt.legend(title='Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, 'F1_Score_boxplot.png'))
    plt.show()

    # Plot 2: F1-Score vs. Elapsed Time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Elapsed_Time', y='F1_Score', hue='Classifier', style='Method', data=df)
    plt.title('F1-Score vs. Elapsed Time')
    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('F1-Score')
    plt.legend(title='Classifier & Method')
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, 'f1_vs_time_scatter.png'))
    plt.show()

    # Plot 3: Memory Usage Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Dataset', y='Memory_Usage', hue='Classifier', data=df)
    plt.title('Memory Usage by Classifier and Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Memory Usage (MB)')
    plt.legend(title='Classifier')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, 'memory_usage_barplot.png'))
    plt.show()

    # Find the best hyperparameter settings for each combination of Method, Dataset, and Classifier
    best_models = df.loc[df.groupby(['Method', 'Dataset', 'Classifier'])['F1_Score'].idxmax()]

    # Plot 4: Best Hyperparameters Visualization for SVC
    svc_data = best_models[best_models['Classifier'] == 'SVC']
    plt.figure(figsize=(10, 6))
    sns.stripplot(x='Dataset', y='F1_Score', hue='C', data=svc_data, jitter=True)
    plt.title('SVC F1-Score vs. Hyperparameter C across Datasets')
    plt.xlabel('Dataset')
    plt.ylabel('F1-Score')
    plt.legend(title='C Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, 'svc_c_vs_f1.png'))
    plt.show()

    # Export summary table to CSV
    summary.to_csv('classifier_family_summary.csv', index=False)

    # Plot 1: Box Plot of Accuracy by Classifier Family
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Classifier', y='Accuracy', hue='Method', data=df)
    plt.title('Accuracy by Classifier Family and Method')
    plt.xlabel('Classifier Family')
    plt.ylabel('Accuracy')
    plt.legend(title='Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, 'accuracy_classifier_family_boxplot.png'))
    plt.show()

    # Plot 2: F1-Score vs. Elapsed Time by Classifier Family
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Elapsed_Time', y='F1_Score', hue='Classifier', style='Method', data=df)
    plt.title('F1-Score vs. Elapsed Time by Classifier Family')
    plt.xlabel('Elapsed Time (seconds)')
    plt.ylabel('F1-Score')
    plt.legend(title='Classifier & Method')
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, 'f1_vs_time_classifier_family_scatter.png'))
    plt.show()

    # Export Tables
    summary.to_csv('performance_summary.csv', index=False)
    best_models.to_latex('best_models.tex', index=False)

    # Additional Plot: Bar charts of top 3 and bottom 3 models per classifier family per dataset
    # Get list of datasets
    datasets = df['Dataset'].unique()
    datasets.sort()

    # Create 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    classifier_families = ['KNN', 'RF', 'SVC']

    for idx, dataset in enumerate(datasets):
        ax = axs.flatten()[idx]
        data = df[df['Dataset'] == dataset]
        selected_models = pd.DataFrame()

        # Define color palettes for each classifier family
        color_palettes = {
            'KNN': sns.color_palette('Blues', 6),
            'RF': sns.color_palette('Greens', 6),
            'SVC': sns.color_palette('Purples', 6)
        }

        for clf in classifier_families:
            # Get data for this classifier family
            clf_data = data[data['Classifier'] == clf]
            if len(clf_data) == 0:
                continue
            # Sort by F1_Score
            clf_data_sorted = clf_data.sort_values('F1_Score')
            # Get bottom 3 and top 3
            bottom3 = clf_data_sorted.head(3)
            top3 = clf_data_sorted.tail(3)
            selected = pd.concat([bottom3, top3])
            selected = selected.copy().reset_index(drop=True)
            # Assign colors
            selected['Color'] = color_palettes[clf]
            # Use abbreviated model names
            selected_models = pd.concat([selected_models, selected])

        # Now plot
        # Sort selected_models by F1_Score
        selected_models = selected_models.sort_values('F1_Score').reset_index(drop=True)

        x = np.arange(len(selected_models))
        y = selected_models['F1_Score']
        colors = selected_models['Color']
        labels = selected_models['Abbrev_Model_Name']

        ax.bar(x, y, color=colors)
        ax.set_title(f'Dataset: {dataset}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylabel('F1_Score')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, 'top_bottom_models_bar_charts.png'))
    plt.show()
