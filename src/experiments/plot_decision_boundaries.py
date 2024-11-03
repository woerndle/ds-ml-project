# src/experiments/plot_decision_boundaries.py

import sys
import os
# Append the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from src.data_processing.preprocess import load_and_preprocess_data

def main():
    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data()

    # Combine training and validation data
    X = np.vstack((X_train, X_val))
    y = np.hstack((y_train, y_val))

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Select top 3 classes
    class_counts = pd.Series(y_encoded).value_counts()
    top_classes = class_counts.index[:3]

    # Filter data
    indices = np.isin(y_encoded, top_classes)
    X_subset = X_reduced[indices]
    y_subset = y_encoded[indices]

    # Train SVM models
    C = 1.0
    svc = svm.SVC(kernel='linear', C=C).fit(X_subset, y_subset)
    lin_svc = svm.LinearSVC(C=C, max_iter=10000).fit(X_subset, y_subset)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_subset, y_subset)
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_subset, y_subset)

    models = [svc, lin_svc, rbf_svc, poly_svc]
    titles = [
        'SVC with linear kernel',
        'LinearSVC (linear kernel)',
        'SVC with RBF kernel',
        'SVC with polynomial (degree 3) kernel'
    ]

    # Create mesh
    h = .02
    x_min, x_max = X_subset[:, 0].min() - 1, X_subset[:, 0].max() + 1
    y_min, y_max = X_subset[:, 1].min() - 1, X_subset[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Plot decision boundaries
    for i, clf in enumerate(models):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y_subset, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(titles[i])

    plt.show()

if __name__ == "__main__":
    main()
