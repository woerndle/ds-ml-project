# src/evaluation/visualisation.py

import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
# src/evaluation/visualisation.py

import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import numpy as np

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the data loading functions
from data_processing.preprocess import load_and_preprocess_data

# Import model-getting functions
from models.knn import get_knn_models
from models.rf import get_rf_models
from models.svm import get_svm_models

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

def get_classifier_family(model_name):
    """Extract the classifier family from the model name."""
    if model_name.startswith('KNN'):
        return 'KNN'
    elif model_name.startswith('RF'):
        return 'Random Forest'
    elif model_name.startswith('SVC') or model_name.startswith('Calibrated LinearSVC'):
        return 'SVM'
    else:
        return 'Other'

def abbreviate_model_name(model_name):
    """Abbreviate the model name to fit in the plot."""
    # Replace long parameter names with abbreviations
    abbreviations = {
        'n_estimators': 'n_est',
        'max_depth': 'max_d',
        'min_samples_split': 'min_ss',
        'bootstrap': 'bs',
        'kernel': 'k',
        'degree': 'deg',
        'gamma': 'g',
        'metric': 'm',
        'n_neighbors': 'k',
        'weights': 'w',
        'algorithm': 'alg',
        'C': 'C',
    }
    for long, short in abbreviations.items():
        model_name = model_name.replace(long, short)
    # Wrap text if it's still too long
    return '\n'.join(textwrap.wrap(model_name, width=30))

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
                    # Add classifier family
                    metrics['classifier_family'] = get_classifier_family(model)
                    # Accuracy
                    metrics['accuracy'] = metrics_data.get('accuracy', None)
                    # Parse classification report
                    report_str = metrics_data.get('classification_report', '')
                    report_metrics = parse_classification_report(report_str)
                    metrics.update(report_metrics)
                    results.append(metrics)
            else:
                print(f"Metrics file not found for model {model} in dataset {dataset}.")
    df = pd.DataFrame(results)
    # Abbreviate model names
    df['model_abbr'] = df['model'].apply(abbreviate_model_name)
    return df

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
    sns.set_theme(style='whitegrid')

    # Define color palettes for classifier families
    palettes = {
        'SVM': sns.color_palette("Blues", 6),
        'Random Forest': sns.color_palette("Greens", 6),
        'KNN': sns.color_palette("Reds", 6),
        'Other': sns.color_palette("Purples", 6)
    }

    for metric in metrics_to_plot:
        if metric not in df.columns:
            continue  # Skip metrics not present in the data

        # For each dataset
        datasets = df['dataset'].unique()

        for dataset in datasets:
            subset = df[df['dataset'] == dataset]
            # For each classifier family
            classifier_families = subset['classifier_family'].unique()
            frames = []
            for family in classifier_families:
                family_subset = subset[subset['classifier_family'] == family]
                family_subset = family_subset.sort_values(by=metric, ascending=False)
                top_3 = family_subset.head(3).copy()
                bottom_3 = family_subset.tail(3).copy()
                frames.append(top_3)
                frames.append(bottom_3)
            # Combine selected models
            plot_data = pd.concat(frames).drop_duplicates()

            # Create a FacetGrid
            g = sns.FacetGrid(
                plot_data,
                col='classifier_family',
                height=8,
                aspect=1,
                sharey=False,
            )

            def barplot(data, x, y, **kwargs):
                family = data['classifier_family'].iloc[0]
                sns.barplot(
                    data=data,
                    x=x,
                    y=y,
                    palette=palettes[family][:len(data)],
                    dodge=False,
                    **kwargs
                )

            g.map_dataframe(
                barplot,
                x='model_abbr',
                y=metric
            )

            g.set_xticklabels(rotation=90, horizontalalignment='center', fontsize=9)
            g.set_titles('{col_name}')
            g.set_axis_labels('Model', metric.capitalize())

            # Adjust layout
            g.fig.set_size_inches(5 * len(classifier_families), 10)
            plt.subplots_adjust(top=0.85)
            g.fig.suptitle(f'{metric} Comparison on {dataset}', fontsize=16)

            # Save the plot
            plot_file = os.path.join(plots_dir, f'{metric}_comparison_{dataset}.png')
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()
            print(f"Plot saved: {plot_file}")




def generate_decision_boundary_plots(models, X, y, dataset_name):
    """Generate decision boundary plots for selected models."""
    # Create output directory for decision boundary plots
    plots_dir = os.path.join('output_results', 'visualisations', 'decision_boundaries')
    os.makedirs(plots_dir, exist_ok=True)

    # Prepare data
    X = StandardScaler().fit_transform(X)
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.3, stratify=y, random_state=42)

    # Set figure size
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)

    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models):
        clf = make_pipeline(StandardScaler(), model)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot decision boundary
        DecisionBoundaryDisplay.from_estimator(
            clf, X_2d, ax=ax, alpha=0.5, eps=0.5, cmap='coolwarm', response_method='predict'
        )

        # Plot data points
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=20)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=50, alpha=0.6, marker='*')

        ax.set_xticks(())
        ax.set_yticks(())
        model_abbr_name = abbreviate_model_name(name)
        ax.set_title(f'{model_abbr_name}\nAccuracy: {score:.2f}', fontsize=10)

    plt.tight_layout()
    # Save the plot
    plot_file = os.path.join(plots_dir, f'decision_boundaries_{dataset_name}.png')
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"Decision boundary plot saved: {plot_file}")

def main():
    output_dir = 'output_results'
    # Collect metrics from JSON files
    df = collect_metrics(output_dir)
    if df.empty:
        print("No metrics found to plot.")
        return

    # Plot the metrics
    plot_metrics(df)

    # Generate decision boundary plots for selected models on actual datasets

    datasets = df['dataset'].unique()

    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        # Load the data
        X_full, _, y_full, _, label_encoder, _ = load_and_preprocess_data(dataset_name, eval_method='holdout')

        # Flatten X_full and y_full if they are split
        if isinstance(X_full, tuple):
            X_full = np.vstack((X_full[0], X_full[1]))
            y_full = np.hstack((y_full[0], y_full[1]))

        # Select top model per classifier family for the dataset
        dataset_df = df[df['dataset'] == dataset_name]
        decision_boundary_models = []
        for family in dataset_df['classifier_family'].unique():
            family_df = dataset_df[dataset_df['classifier_family'] == family]
            top_model_name = family_df.sort_values('accuracy', ascending=False).iloc[0]['model']
            # Recreate the model instance
            if family == 'KNN':
                models_list = get_knn_models()
            elif family == 'Random Forest':
                models_list = get_rf_models()
            elif family == 'SVM':
                models_list = get_svm_models()
            else:
                continue

            # Find the model instance matching the name
            for model_name, model_instance in models_list:
                if model_name == top_model_name:
                    decision_boundary_models.append((model_name, model_instance))
                    break

        if decision_boundary_models:
            # Generate decision boundary plots using the actual data
            generate_decision_boundary_plots(decision_boundary_models, X_full, y_full, dataset_name)

if __name__ == '__main__':
    main()
