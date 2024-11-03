# src/experiments/run_experiments.py

import sys
import os
import time
from tqdm import tqdm

# Append the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import label_binarize, LabelEncoder

from src.data_processing.preprocess import load_and_preprocess_data
from src.models.svm import get_svm_models

def plot_confusion_matrix(y_true, y_pred, model_name, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=False, ax=ax)
    ax.set_title(f'Confusion Matrix\n{model_name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

def plot_roc_curve(y_true, y_scores, model_name, ax):
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_binarized.shape[1]

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve for the micro-average
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
        color='deeppink',
        linewidth=2,
    )

    # Plot ROC curves for selected classes
    for i in range(n_classes):
        if i % 10 == 0:
            ax.plot(
                fpr[i],
                tpr[i],
                label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]),
            )

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve\n{model_name}')
    ax.legend(loc="lower right", fontsize='small')

def plot_precision_recall_curve(y_true, y_scores, model_name, ax):
    y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
    precision, recall, average_precision = {}, {}, {}

    for i in range(y_true_binarized.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_binarized[:, i], y_scores[:, i]
        )
        average_precision[i] = auc(recall[i], precision[i])

    # Plot PR curves for selected classes
    for i in range(y_true_binarized.shape[1]):
        if i % 10 == 0:
            ax.plot(
                recall[i],
                precision[i],
                label='Class {0} (AP = {1:0.2f})'.format(i, average_precision[i]),
            )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve\n{model_name}')
    ax.legend(loc='lower left', fontsize='small')

def main():
    X_train, X_val, y_train, y_val = load_and_preprocess_data()

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)

    models = get_svm_models()

    # Initialize lists to store results
    model_names = []
    y_preds = []
    y_scores_list = []
    training_times = []
    roc_aucs = []

    for model_name, model in tqdm(models, desc="Training Models"):
        try:
            # Training the model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            training_times.append(training_time)

            # Predictions and evaluations
            y_pred = model.predict(X_val)
            print(f"\nTraining time for {model_name}: {training_time:.2f} seconds\n")
            print(f"{model_name}\n{classification_report(y_val, y_pred, zero_division=0)}")
            print(confusion_matrix(y_val, y_pred))

            # Storing results
            model_names.append(model_name)
            y_preds.append(y_pred)

            # Compute ROC AUC Score if possible
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_val)
                roc_auc = roc_auc_score(y_val, y_score, multi_class='ovr')
                roc_aucs.append(roc_auc)
                y_scores_list.append(y_score)
                print(f"ROC AUC Score for {model_name}: {roc_auc}\n")
            else:
                roc_aucs.append(None)
                y_scores_list.append(None)
                print(f"{model_name} does not support predict_proba.\n")

        except Exception as e:
            print(f"An error occurred during the experiment with {model_name}: {e}")
            continue

    # Create output directory if it doesn't exist
    output_dir = os.path.join(project_root, 'output_plots')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot confusion matrices
    num_models = len(model_names)
    fig_cm, axs_cm = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axs_cm = [axs_cm]
    for i in range(num_models):
        plot_confusion_matrix(y_val, y_preds[i], model_names[i], axs_cm[i])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()
    print("Saved confusion matrices plot.")

    # Plot ROC curves
    fig_roc, axs_roc = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axs_roc = [axs_roc]
    for i in range(num_models):
        if y_scores_list[i] is not None:
            plot_roc_curve(y_val, y_scores_list[i], model_names[i], axs_roc[i])
        else:
            axs_roc[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    print("Saved ROC curves plot.")

    # Plot Precision-Recall curves
    fig_pr, axs_pr = plt.subplots(1, num_models, figsize=(6 * num_models, 5))
    if num_models == 1:
        axs_pr = [axs_pr]
    for i in range(num_models):
        if y_scores_list[i] is not None:
            plot_precision_recall_curve(y_val, y_scores_list[i], model_names[i], axs_pr[i])
        else:
            axs_pr[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'))
    plt.close()
    print("Saved precision-recall curves plot.")

if __name__ == "__main__":
    main()
