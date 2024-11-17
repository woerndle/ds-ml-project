# src/models/random_forest.py
from sklearn.ensemble import RandomForestClassifier

def get_rf_models():
    # Define Random Forest models with different hyperparameters
    models = [
        ("Random Forest (10 estimators)", RandomForestClassifier(n_estimators=10, class_weight='balanced')),
        ("Random Forest (100 estimators)", RandomForestClassifier(n_estimators=100, class_weight='balanced')),
        ("Random Forest (max_depth=5)", RandomForestClassifier(max_depth=5, n_estimators=100, class_weight='balanced')),
        ("Random Forest (min_samples_split=10)", RandomForestClassifier(min_samples_split=10, n_estimators=100, class_weight='balanced')),
        ("Random Forest (bootstrap=False)", RandomForestClassifier(n_estimators=100, bootstrap=False, class_weight='balanced')),
    ]
    return models
