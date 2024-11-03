# src/experiments/run_experiments.py
import sys
import os

# Append the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
from src.data_processing.preprocess import load_and_preprocess_data
from src.models.svm import get_svm_models
from src.evaluation.metrics import evaluate_model

def plot_decision_boundaries(models, X_train, y_train, X_test, y_test, dataset_name):
    plt.figure(figsize=(12, 10))

    for i, (title, model) in enumerate(models):
        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        report, matrix = evaluate_model(model, X_test, y_test)
        print(f"\n{title}")
        print(report)
        print(matrix)

        # Plot decision boundary only for datasets with two features
        if X_train.shape[1] == 2:
            plt.subplot(2, 2, i + 1)
            plt.title(title)

            x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
            y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                                 np.arange(y_min, y_max, 0.02))

            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

            # Plot data points
            scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolor="k", s=30)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
        else:
            # If not 2D, we can plot a different visualization
            plt.subplot(2, 2, i + 1)
            plt.title(title)
            plt.scatter(range(len(y_train)), y_train, alpha=0.5, label='Training data', color='blue')
            plt.scatter(range(len(y_test)), y_test, alpha=0.5, label='Testing data', color='orange')
            plt.xlabel('Sample index')
            plt.ylabel('Class')
            plt.legend()
            plt.title(f'Dataset: {dataset_name} - {title}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data for a specified dataset
    try:
        # Expecting 5 values to be returned
        X_train, X_val, y_train, y_val, X_test = load_and_preprocess_data()

        # Get models
        models = get_svm_models()

        # Plot decision boundaries for each model (You may need to adapt this for validation set)
        plot_decision_boundaries(models, X_train, y_train, X_val, y_val, 'amazon')

    except Exception as e:
        print(f"An error occurred during the experiment execution: {e}")
