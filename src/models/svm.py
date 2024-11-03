# src/models/svm.py
from sklearn.svm import SVC, LinearSVC

def get_svm_models():
    # Define SVM models with different kernels
    models = [
        ("SVC with linear kernel", SVC(kernel="linear")),
        ("LinearSVC (linear kernel)", LinearSVC(max_iter=10000)),
        ("SVC with RBF kernel", SVC(kernel="rbf", gamma=0.7)),
        ("SVC with polynomial (degree 3) kernel", SVC(kernel="poly", degree=3))
    ]
    return models
