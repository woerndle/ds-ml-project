# src/models/svm.py
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def get_svm_models():
    # Define SVM models with probability estimates
    models = [
        #("Calibrated LinearSVC (linear kernel)", CalibratedClassifierCV(LinearSVC(max_iter=100))),
        ("SVC with linear kernel", SVC(kernel="linear",class_weight='balanced', probability=True)),
        ("SVC with RBF kernel", SVC(kernel="rbf",class_weight='balanced', probability=True)),
        ("SVC with polynomial (degree 3) kernel", SVC(kernel="poly", degree=3,class_weight='balanced', probability=True)),
        ]
    return models