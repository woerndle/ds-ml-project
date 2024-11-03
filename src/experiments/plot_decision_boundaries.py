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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.data_processing.preprocess import load_and_preprocess_data

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, 'output_plots')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data("traffic_prediction")

    # Combine training and validation data
    X = np.vstack((X_train, X_val))
    y = np.hstack((y_train, y_val))

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Get feature names if available
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns
    else:
        feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scaled)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by PC1 and PC2: {explained_variance.sum():.2%}")
    print(f"PC1 explains {explained_variance[0]:.2%} of the variance")
    print(f"PC2 explains {explained_variance[1]:.2%} of the variance\n")

    # Get the loadings (coefficients)
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    print("Top features contributing to PC1:")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(10))
    print("\nTop features contributing to PC2:")
    print(loadings['PC2'].abs().sort_values(ascending=False).head(10))

    # Visualize the loadings using a biplot
    def biplot(score, coeff, labels=None):
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        plt.scatter(xs * scalex, ys * scaley, c=y_encoded, cmap=plt.cm.coolwarm, edgecolors='k')
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],
                      color='r', alpha=0.5)
            if labels is None:
                plt.text(coeff[i,0]*1.15, coeff[i,1]*1.15, f"Var{i+1}", color='g', ha='center', va='center')
            else:
                plt.text(coeff[i,0]*1.15, coeff[i,1]*1.15, labels[i], color='g', ha='center', va='center')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Biplot of Principal Components")

    plt.figure(figsize=(10, 8))
    biplot(X_reduced, pca.components_.T, labels=feature_names)
    plt.show()

    # Select top 3 classes
    class_counts = pd.Series(y_encoded).value_counts()
    top_classes = class_counts.index[:3]

    # Filter data for the top classes
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
    plt.figure(figsize=(12, 10))
    for i, clf in enumerate(models):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(
            X_subset[:, 0],
            X_subset[:, 1],
            c=y_subset,
            cmap=plt.cm.coolwarm,
            edgecolors='k'
        )
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(titles[i])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())

    plt.savefig(os.path.join(output_dir, 'decision_boundaries.png'))
    plt.close()

if __name__ == "__main__":
    main()
