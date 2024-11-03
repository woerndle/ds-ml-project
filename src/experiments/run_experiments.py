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

def plot_decision_boundaries(models, X_train, y_train, X_test, y_test, X, y, iris):
    plt.figure(figsize=(12, 10))

    for i, (title, model) in enumerate(models):
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        report, matrix = evaluate_model(model, X_test, y_test)
        print(f"\n{title}")
        print(report)
        print(matrix)
        
        # Plot decision boundary
        plt.subplot(2, 2, i + 1)
        plt.title(title)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor="k", s=30)
        plt.xlabel(iris.feature_names[0])
        plt.ylabel(iris.feature_names[1])

    # Legend for target classes
    for j, class_name in enumerate(iris.target_names):
        plt.scatter([], [], color=plt.cm.coolwarm(j / 2), label=class_name)
    plt.legend(title="Classes")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test, X, y, iris = load_and_preprocess_data()
    
    # Get models
    models = get_svm_models()
    
    # Plot decision boundaries for each model
    plot_decision_boundaries(models, X_train, y_train, X_test, y_test, X, y, iris)
