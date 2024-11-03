# src/models/svm.py
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def get_svm_models():
    # Define SVM models with probability estimates
    models = [
        ("SVC with linear kernel", SVC(kernel="linear", probability=True)),
        ("Calibrated LinearSVC (linear kernel)", CalibratedClassifierCV(LinearSVC(max_iter=100))),
        ("SVC with RBF kernel", SVC(kernel="rbf", gamma=0.7, probability=True)),
        ("SVC with polynomial (degree 3) kernel", SVC(kernel="poly", degree=3, probability=True))
    ]
    return models