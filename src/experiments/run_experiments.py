## src/experiments/run_experiments.py
#
import sys
import os

# Append the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from scipy.sparse import issparse

# Import your custom modules
from src.data_processing.preprocess import load_and_preprocess_data
from src.models.svm import get_svm_models
from src.models.knn import get_knn_models
from src.models.rf import get_rf_models
from src.evaluation.metrics import save_metrics, evaluate_model

# Utility functions for plotting
def plot_confusion_matrix(y_true, y_pred, model_name, ax):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False, ax=ax,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    ax.set_title(f'Confusion Matrix\n{model_name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

def plot_roc_curve(y_true, y_scores, model_name, ax):
    classes = np.unique(y_true)
    n_classes = len(classes)
    y_true_binarized = label_binarize(y_true, classes=classes)

    if n_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true_binarized, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange',
                lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    else:
        # Multiclass classification
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Plot micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        ax.plot(fpr["micro"], tpr["micro"],
                label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
                color='deeppink', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve\n{model_name}')
    ax.legend(loc="lower right", fontsize='small')

def plot_precision_recall_curve(y_true, y_scores, model_name, ax):
    classes = np.unique(y_true)
    y_true_binarized = label_binarize(y_true, classes=classes)
    precision, recall, average_precision = {}, {}, {}

    for i in range(y_true_binarized.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_binarized[:, i], y_scores[:, i]
        )
        average_precision[i] = auc(recall[i], precision[i])

    # Plot PR curves for selected classes
    for i in range(y_true_binarized.shape[1]):
        if i % 10 == 0 or y_true_binarized.shape[1] <= 10:
            ax.plot(
                recall[i],
                precision[i],
                label='Class {0} (AP = {1:0.2f})'.format(classes[i], average_precision[i]),
            )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve\n{model_name}')
    ax.legend(loc='lower left', fontsize='small')

def plot_classification_report_heatmap(y_true, y_pred, model_name, ax):
    from sklearn.metrics import classification_report
    import pandas as pd
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    # Exclude 'accuracy', 'macro avg', 'weighted avg' rows
    report_df = report_df.iloc[:-3, :]
    sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap='Blues', fmt='.2f', ax=ax)
    ax.set_title(f'Classification Report Heatmap\n{model_name}')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Classes')

def plot_decision_boundary(model, X, y, model_name, ax):
    from matplotlib.colors import ListedColormap
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Reduce data to two dimensions
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    # Fit model on reduced data
    model.fit(X_scaled, y)

    # Create mesh grid
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    h = (x_max - x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cm_bright, edgecolor='k', s=20)
    ax.set_title(f'Decision Boundary\n{model_name}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run experiments on a dataset.')
    parser.add_argument('--dataset', type=str, default='traffic_prediction', help='Dataset name to use.')
    parser.add_argument('--model', type=str, default='knn', choices=['svm', 'knn', 'rf'], help='Model to use.')
    parser.add_argument('--subset', type=int, default=None, help='Use a subset of data for testing.')
    parser.add_argument('--eval_method', type=str, default='holdout', choices=['holdout', 'cross_val'], help='Evaluation method to use.')
    args = parser.parse_args()

    dataset_name = args.dataset
    model_type = args.model
    eval_method = args.eval_method

    # Load and preprocess data
    data = load_and_preprocess_data(
        dataset_name=dataset_name,
        data_size=1000,
        eval_method=eval_method
    )

    if eval_method == 'holdout':
        X_train, X_val, y_train, y_val, label_encoder, tfidf_vect = data
        
        # Check if X_train is a sparse matrix and convert to CSR format
        if issparse(X_train):
            X_train = X_train.tocsr()
            X_val = X_val.tocsr()

        # Convert DataFrames to NumPy arrays to avoid warnings
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.to_numpy()

        # Verify data shapes
        print(f"X_train shape: {X_train.shape}, should be (n_samples, n_features)")
        print(f"y_train shape: {y_train.shape}, should be (n_samples,)")
        print(f"X_val shape: {X_val.shape}, should be (n_samples, n_features)")
        print(f"y_val shape: {y_val.shape}, should be (n_samples,)")

        # Select models based on specified type
        if model_type == 'svm':
            models = get_svm_models()
        elif model_type == 'knn':
            models = get_knn_models()
        elif model_type == 'rf':
            models = get_rf_models()

        results = []

        # Main training and evaluation loop for holdout
        for model_name, model in tqdm(models, desc=f"Running {model_type} models"):
            try:
                # Train the model
                model.fit(X_train, y_train)

                # Evaluate the model
                metrics, y_test_decoded = evaluate_model(model, X_val, y_val, label_encoder)
                metrics['model_name'] = model_name
                results.append(metrics)

                # Generate and save plots using decoded labels
                y_scores = metrics['y_score']
                if y_scores is None:
                    print(f"Model {model_name} does not support probability estimates.")
                    continue

                # Create a directory for the model's output
                output_dir = os.path.join("output_results", dataset_name, model_name)
                os.makedirs(output_dir, exist_ok=True)

                # Initialize subplots
                fig, axes = plt.subplots(2, 3, figsize=(27, 10))

                # Plot Confusion Matrix
                plot_confusion_matrix(y_test_decoded, metrics['y_pred_decoded'], model_name, ax=axes[0, 0])

                # Plot ROC Curve
                plot_roc_curve(y_test_decoded, y_scores, model_name, ax=axes[0, 1])

                # Plot Precision-Recall Curve
                plot_precision_recall_curve(y_test_decoded, y_scores, model_name, ax=axes[0, 2])

                # Plot Classification Report Heatmap
                plot_classification_report_heatmap(y_test_decoded, metrics['y_pred_decoded'], model_name, ax=axes[1, 0])

                # Plot Decision Boundary (if applicable)
                try:
                    plot_decision_boundary(model, X_val, y_val, model_name, ax=axes[1, 1])
                except Exception as e:
                    print(f"Decision boundary plot not generated for {model_name}: {e}")
                    axes[1, 1].set_visible(False)

                # Hide any unused subplots
                axes[1, 2].set_visible(False)

                # Adjust layout and save the plot
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{model_name}_plots.png"))
                plt.close()

                # Save individual model results
                save_metrics(metrics, model_name=model_name, dataset_name=dataset_name)

            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                continue

        # Save overall results to JSON
        output_dir = os.path.join("output_results", dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f'results_{dataset_name}_{model_type}.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"All results saved to {results_file}")

    elif eval_method == 'cross_val':
        X, y, label_encoder, tfidf_vect, cv = data

        # Check if X is a sparse matrix and convert to CSR format
        if issparse(X):
            X = X.tocsr()

        # Convert DataFrames to NumPy arrays to avoid warnings
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Verify data shapes
        print(f"X shape: {X.shape}, should be (n_samples, n_features)")
        print(f"y shape: {y.shape}, should be (n_samples,)")

        # Select models based on specified type
        if model_type == 'svm':
            models = get_svm_models()
        elif model_type == 'knn':
            models = get_knn_models()
        elif model_type == 'rf':
            models = get_rf_models()

        results = []

        # Main training and evaluation loop for cross-validation
        for model_name, model in tqdm(models, desc=f"Running {model_type} models with cross-validation"):
            try:
                from sklearn.model_selection import cross_validate

                # Define scoring metrics
                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

                # Perform cross-validation
                cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

                # Compute mean and std of metrics
                metrics = {metric: cv_results['test_' + metric].mean() for metric in scoring}
                metrics_std = {metric + '_std': cv_results['test_' + metric].std() for metric in scoring}

                # Combine metrics and std
                metrics.update(metrics_std)
                metrics['model_name'] = model_name

                results.append(metrics)

                # Save individual model results
                save_metrics(metrics, model_name=model_name, dataset_name=dataset_name)

            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                continue

        # Save overall results to JSON
        output_dir = os.path.join("output_results", dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f'results_{dataset_name}_{model_type}_crossval.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"All cross-validation results saved to {results_file}")

    else:
        raise ValueError("Invalid evaluation method.")

if __name__ == "__main__":
    main()
