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
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np

from sklearn.decomposition import PCA

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
        return 'RF'
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
        model_dirs = [m for m in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, m))]
        
        for model_dir in model_dirs:
            metrics_file = os.path.join(dataset_dir, model_dir, f'{model_dir}_metrics.json')
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                # Use model_name from metrics_data
                model_name = metrics_data.get('model_name', model_dir)
                
                # Extract metrics specific to the current model and dataset
                
                metrics = {
                    'dataset': dataset,
                    'model': model_name,
                    'classifier_family': get_classifier_family(model_name),
                    'accuracy': metrics_data.get('accuracy', None)
                }
                print(metrics)
                # Parse and add classification report metrics
                report_str = metrics_data.get('classification_report', '')
                report_metrics = parse_classification_report(report_str)
                metrics.update(report_metrics)
                
                # Append the model-specific metrics
                results.append(metrics)
            else:
                print(f"Metrics file not found for model {model_dir} in dataset {dataset}.")
    
    # Create a DataFrame from the collected results
    df = pd.DataFrame(results)
    
    # Add abbreviated model names
    #df['model_abbr'] = df['model'].apply(abbreviate_model_name)
    df['model_abbr'] = df['model'].apply(abbreviate_model_name)    
    return df


def plot_metrics(df):
    """Plot comparative bar charts for each metric across datasets and models."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import pandas as pd

    # Metrics to plot
    metrics_to_plot = ['accuracy', 'macro avg_precision', 'macro avg_recall',
                       'macro avg_f1-score', 'weighted avg_precision',
                       'weighted avg_recall', 'weighted avg_f1-score']
    
    # Create output directory for plots
    plots_dir = os.path.join('output_results', 'visualisations')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set seaborn style
    sns.set_theme(style='whitegrid')

    # Define color palettes for classifier families (reversed for full color first)
    palettes = {
        'SVM': sns.color_palette("Blues", 6)[::-1],
        'RF': sns.color_palette("Greens", 6)[::-1],
        'KNN': sns.color_palette("Purples", 6)[::-1],
        'Other': sns.color_palette("Reds", 6)[::-1]
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
            n_families = len(classifier_families)
            fig, axes = plt.subplots(1, n_families, figsize=(6 * n_families, 8), sharey=True)
            if n_families == 1:
                axes = [axes]
            for ax, family in zip(axes, classifier_families):
                family_subset = subset[subset['classifier_family'] == family]
                family_subset = family_subset.sort_values(by=metric, ascending=False)
                top_3 = family_subset.head(3).copy()
                bottom_3 = family_subset.tail(3).copy()
                plot_data = pd.concat([top_3, bottom_3]).drop_duplicates()
                sns.barplot(
                    data=plot_data,
                    x='model_abbr',
                    y=metric,
                    palette=palettes.get(family, 'gray')[:len(plot_data)],
                    ax=ax
                )
                ax.set_title(f'{family} Models')
                ax.set_xlabel('Model')
                ax.set_ylabel(metric.capitalize())
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                # Annotate bars with metric values
                for p in ax.patches:
                    height = p.get_height()
                    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 5),
                                textcoords='offset points')
            plt.tight_layout()
            plt.suptitle(f'{metric.capitalize()} Comparison on {dataset}', fontsize=16, y=1.02)

            # Save the plot
            plot_file = os.path.join(plots_dir, f'{metric}_comparison_{dataset}.png')
            plt.savefig(plot_file, bbox_inches='tight')
            plt.close()
            print(f"Plot saved: {plot_file}")


def generate_decision_boundary_plots(models, X, y, dataset_name):
    """Generate decision boundary plots for selected models."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import matplotlib.pyplot as plt
    from sklearn.inspection import DecisionBoundaryDisplay

    # Create output directory for decision boundary plots
    plots_dir = os.path.join('output_results', 'visualisations', 'decision_boundaries')
    os.makedirs(plots_dir, exist_ok=True)

    # Sample data to reduce computation time
    max_samples = 10000
    if X.shape[0] > max_samples:
        X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=42)
    else:
        X_sampled = X
        y_sampled = y

    # Handle sparse matrices in StandardScaler
    scaler = StandardScaler(with_mean=False)

    # Reduce to 2D using TruncatedSVD
    svd = TruncatedSVD(n_components=2, random_state=42)

    # Create a pipeline for scaling and dimensionality reduction
    dimensionality_reducer = make_pipeline(scaler, svd)

    X_2d = dimensionality_reducer.fit_transform(X_sampled)

    X_train, X_test, y_train, y_test = train_test_split(X_2d, y_sampled, test_size=0.3, stratify=y_sampled, random_state=42)

    # Set figure size
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)

    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models):
        clf = model
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
            X_full = sparse.vstack((X_full[0], X_full[1]))
            y_full = np.hstack((y_full[0], y_full[1]))

        # Select the top model for the dataset
        dataset_df = df[df['dataset'] == dataset_name]
        top_model_name = dataset_df.sort_values('accuracy', ascending=False).iloc[0]['model']

        # Recreate the model instance
        # Identify the classifier family of the top model
        classifier_family = get_classifier_family(top_model_name)
        if classifier_family == 'KNN':
            models_list = get_knn_models()
        elif classifier_family == 'RF':
            models_list = get_rf_models()
        elif classifier_family == 'SVM':
            models_list = get_svm_models()
        else:
            continue  # Skip if classifier family is not recognized

        # Find the model instance matching the name
        top_model = None
        for model_name, model_instance in models_list:
            if model_name == top_model_name:
                top_model = (model_name, model_instance)
                break

        if top_model:
            # Generate decision boundary plots using the actual data
            generate_decision_boundary_plots([top_model], X_full, y_full, dataset_name)
        else:
            print(f"Model instance not found for model: {top_model_name}")


if __name__ == "__main__":
    main()
