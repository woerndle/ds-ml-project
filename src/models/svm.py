# src/models/svm.py
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def get_svm_models():
    # Define SVM models with probability estimates and various kernel types
    models = [
        # Linear kernel SVM
        ("SVC with linear kernel", SVC(kernel="linear", class_weight='balanced', probability=True)),
        
        # RBF kernel SVM
        ("SVC with RBF kernel", SVC(kernel="rbf", class_weight='balanced', probability=True)),
        
        # Polynomial kernel SVM (degree 3)
        ("SVC with polynomial (degree 3) kernel", SVC(kernel="poly", degree=3, class_weight='balanced', probability=True)),
        
        # Calibrated LinearSVC for probability estimates
        ("Calibrated LinearSVC (linear kernel)", CalibratedClassifierCV(LinearSVC(max_iter=100, class_weight='balanced'))),
        
        # Linear SVM with One-vs-All strategy explicitly defined
        ("SVC with linear kernel (OvA)", SVC(kernel="linear", class_weight='balanced', decision_function_shape="ovr", probability=True)),
        
        # Custom kernel example (if a custom kernel function is implemented separately)
        
        ("SVC with custom kernel", SVC(kernel=custom_kernel_function, class_weight='balanced', probability=True)),
    ]
    return models

def custom_kernel_function(X, Y):
    # Example: Gaussian-like kernel
    return np.exp(-0.5 * np.linalg.norm(X[:, None] - Y, axis=2) ** 2)
