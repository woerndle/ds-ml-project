# src/models/random_forest.py
from sklearn.ensemble import RandomForestClassifier

def get_rf_models():
    n_estimators = [10, 50, 100, 200]
    max_depths = [None, 5, 10, 20]
    min_samples_splits = [2, 5, 10]
    bootstraps = [True, False]
    models = []

    for n in n_estimators:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for bootstrap in bootstraps:
                    current_model = (
                        f"RF (n_estimators={n}, max_depth={max_depth}, min_samples_split={min_samples_split}, bootstrap={bootstrap})",
                        RandomForestClassifier(
                            n_estimators=n,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            bootstrap=bootstrap,
                            class_weight='balanced',
                        ),
                    )
                    models.append(current_model)
    return models
