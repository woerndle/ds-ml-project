# src/models/svm.py
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def get_svm_models():
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    degrees = [2, 3, 4, 6, 8]  # For polynomial kernel
    Cs = [0.1, 1, 10]  # Regularization parameter
    gammas = ["scale", "auto"]  # Kernel coefficient for RBF/poly/sigmoid
    models = []

    for kernel in kernels:
        for C in Cs:
            for gamma in gammas:
                if kernel == "poly":
                    for degree in degrees:
                        current_model = (
                            f"SVC (kernel={kernel}, C={C}, gamma={gamma}, degree={degree})",
                            SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, class_weight='balanced', probability=True),
                        )
                        models.append(current_model)
                else:
                    current_model = (
                        f"SVC (kernel={kernel}, C={C}, gamma={gamma})",
                        SVC(kernel=kernel, C=C, gamma=gamma, class_weight='balanced', probability=True),
                    )
                    models.append(current_model)

    # Add LinearSVC and calibrated LinearSVC separately
    models.append(
        ("Calibrated LinearSVC", CalibratedClassifierCV(LinearSVC(C=1, max_iter=1000, class_weight='balanced')))
    )

    return models
