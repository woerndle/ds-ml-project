import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_model_name(model_name):
    """
    Extracts the classifier family from the model name.
    """
    model_name = model_name.strip()
    if model_name.startswith('SVC'):
        family = 'SVC'
    elif model_name.startswith('RF'):
        family = 'RF'
    elif model_name.startswith('KNN'):
        family = 'KNN'
    else:
        family = 'Other'
    return family

def generate_abbreviated_model_name(row):
    """
    Generates an abbreviated model name based on the classifier family and hyperparameters.
    """
    family = row['Classifier']
    hyperparameters = {key: row[key] for key in row.index if key in [
        'kernel', 'C', 'gamma', 'degree', 'coef0',
        'n_estimators', 'max_depth', 'min_samples_split',
        'min_samples_leaf', 'bootstrap', 'max_features',
        'criterion', 'class_weight',
        'n_neighbors', 'weights', 'algorithm', 'metric'
    ]}

    if family == 'SVC':
        # Retrieve all relevant hyperparameters
        kernel = hyperparameters.get('kernel', 'rbf')
        C = hyperparameters.get('C', '1.0')
        gamma = hyperparameters.get('gamma', 'scale')
        degree = hyperparameters.get('degree', '3')
        coef0 = hyperparameters.get('coef0', '0.0')

        # Shorten some parameter values for brevity
        kernel_abbr = kernel[:3]
        gamma_abbr = gamma if gamma in ['scale', 'auto'] else f"g{gamma}"
        C_abbr = f"C{C}"
        degree_abbr = f"d{degree}" if kernel == 'poly' else ''
        coef0_abbr = f"c{coef0}" if kernel in ['poly', 'sigmoid'] else ''

        abbrev = f"SVC-{kernel_abbr}{degree_abbr}{coef0_abbr}-{gamma_abbr}-{C_abbr}"
    elif family == 'RF':
        # Retrieve all relevant hyperparameters
        n_estimators = hyperparameters.get('n_estimators', '100')
        max_depth = hyperparameters.get('max_depth', 'None')
        min_samples_split = hyperparameters.get('min_samples_split', '2')
        min_samples_leaf = hyperparameters.get('min_samples_leaf', '1')
        bootstrap = hyperparameters.get('bootstrap', 'True')
        max_features = hyperparameters.get('max_features', 'auto')
        criterion = hyperparameters.get('criterion', 'gini')
        class_weight = hyperparameters.get('class_weight', 'None')

        # Abbreviate certain hyperparameter values
        criterion_abbr = {'gini': 'g', 'entropy': 'e'}.get(str(criterion), str(criterion)[:1])
        max_features_abbr = {'sqrt': 'sq', 'log2': 'l2', 'auto': 'au'}.get(str(max_features), str(max_features)[:2])
        bootstrap_abbr = 'b' if str(bootstrap) == 'True' else 'nb'
        class_weight_abbr = {'balanced': 'b', 'balanced_subsample': 'bs'}.get(str(class_weight), 'n')

        abbrev = (f"RF-n{n_estimators}-d{max_depth}-s{min_samples_split}-l{min_samples_leaf}-"
                  f"c{criterion_abbr}-mf{max_features_abbr}-{bootstrap_abbr}-cw{class_weight_abbr}")
    elif family == 'KNN':
        # Retrieve all relevant hyperparameters
        n_neighbors = hyperparameters.get('n_neighbors', '5')
        weights = hyperparameters.get('weights', 'uniform')
        algorithm = hyperparameters.get('algorithm', 'auto')
        metric = hyperparameters.get('metric', 'minkowski')

        # Abbreviate certain hyperparameter values
        weights_abbr = {'uniform': 'u', 'distance': 'd'}.get(str(weights), str(weights)[:1])
        algorithm_abbr = {'ball_tree': 'bt', 'kd_tree': 'kd', 'brute': 'br', 'auto': 'au'}.get(str(algorithm), str(algorithm)[:2])
        metric_abbr = {'euclidean': 'euc', 'cityblock': 'cit', 'manhattan': 'man'}.get(str(metric), str(metric)[:3])

        abbrev = f"KNN-{weights_abbr}{algorithm_abbr}{metric_abbr}-k{n_neighbors}"
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
                    family = parse_model_name(model_folder)
                    # Get hyperparameters from model_params
                    model_params = metrics.get('model_params', {})
                    # Define the hyperparameters that were changed during experiments
                    if family == 'SVC':
                        changed_hyperparams = ['kernel', 'C', 'gamma', 'degree', 'coef0']
                    elif family == 'RF':
                        changed_hyperparams = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                                               'bootstrap', 'max_features', 'criterion', 'class_weight']
                    elif family == 'KNN':
                        changed_hyperparams = ['n_neighbors', 'weights', 'algorithm', 'metric']
                    else:
                        changed_hyperparams = []
                    # Extract the values of the changed hyperparameters from model_params
                    hyperparameters = {param: model_params.get(param, None) for param in changed_hyperparams}
                    # Convert boolean values to strings
                    for key, value in hyperparameters.items():
                        if isinstance(value, bool):
                            hyperparameters[key] = str(value)
                    entry = {
                        'Method': 'Holdout',
                        'Dataset': dataset,
                        'Classifier': family,
                        'Model_Name': model_folder,  # Include the full model name
                        'Hyperparameters': hyperparameters,
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
                family = parse_model_name(model_folder)
                # Get hyperparameters from model_params
                model_params = folds[0].get('model_params', {})
                # Define the hyperparameters that were changed during experiments
                if family == 'SVC':
                    changed_hyperparams = ['kernel', 'C', 'gamma', 'degree', 'coef0']
                elif family == 'RF':
                    changed_hyperparams = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                                           'bootstrap', 'max_features', 'criterion', 'class_weight']
                elif family == 'KNN':
                    changed_hyperparams = ['n_neighbors', 'weights', 'algorithm', 'metric']
                else:
                    changed_hyperparams = []
                # Extract the values of the changed hyperparameters from model_params
                hyperparameters = {param: model_params.get(param, None) for param in changed_hyperparams}
                # Convert boolean values to strings
                for key, value in hyperparameters.items():
                    if isinstance(value, bool):
                        hyperparameters[key] = str(value)
                entry = {
                    'Method': 'Cross-Validation',
                    'Dataset': dataset,
                    'Classifier': family,
                    'Model_Name': model_folder,  # Include the full model name
                    'Hyperparameters': hyperparameters,
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
    df['Abbrev_Model_Name'] = df.apply(generate_abbreviated_model_name, axis=1)

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

    # Plot 5: Box Plot of Accuracy by Classifier Family
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

    # Plot 6: F1-Score vs. Elapsed Time by Classifier Family
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
    # Filter the DataFrame to only include holdout results
    df_holdout = df[df['Method'] == 'Holdout']

    # Get list of datasets
    datasets = df_holdout['Dataset'].unique()
    datasets.sort()

    # Determine the grid size based on the number of datasets
    num_datasets = len(datasets)
    num_cols = 2  # Adjust as needed
    num_rows = (num_datasets + num_cols - 1) // num_cols

    # Create subplot grid dynamically based on the number of datasets
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    classifier_families = ['KNN', 'RF', 'SVC']

    for idx, dataset in enumerate(datasets):
        ax = axs.flatten()[idx]
        data = df_holdout[df_holdout['Dataset'] == dataset]
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
            selected['Color'] = color_palettes[clf][:len(selected)]
            # Collect selected models
            selected_models = pd.concat([selected_models, selected])

        # Check if selected_models is empty
        if selected_models.empty:
            continue

        # Sort selected_models by F1_Score
        selected_models = selected_models.sort_values('F1_Score').reset_index(drop=True)

        x = np.arange(len(selected_models))
        y = selected_models['F1_Score']
        colors = selected_models['Color']
        # Use abbreviated model names as labels
        labels = selected_models['Abbrev_Model_Name']

        ax.bar(x, y, color=colors)
        ax.set_title(f'Dataset: {dataset}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel('F1 Score')

    # Remove empty subplots if any
    if num_datasets < num_rows * num_cols:
        for idx in range(num_datasets, num_rows * num_cols):
            fig.delaxes(axs.flatten()[idx])

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, 'top_bottom_models_bar_charts.png'))
    plt.show()
