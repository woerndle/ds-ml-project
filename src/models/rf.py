# src/models/random_forest.py
from sklearn.ensemble import RandomForestClassifier


def get_rf_models():
    n_estimators = [20, 50, 100, 200]
    max_depths = [5, 10, 15]
    min_samples_splits = [2, 5, 10]
    min_samples_leaf = [1, 2, 4],
    bootstraps = [True, False]
    max_features = ['sqrt', 'log2'],
    criterions = ['gini', 'entropy'],
    class_weights = ['balanced', 'balanced_subsample'],
    models = []

    for n in n_estimators:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                for min_sample_leaf in min_samples_leaf:
                    for max_feature in max_features:
                        for criterion in criterions:
                            for class_weight in class_weights:
                                for bootstrap in bootstraps:
                                    current_model = (
                                        f"RF (n_estimators={n}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_sample_leaf} bootstrap={bootstrap},max_features={max_feature},criterions={criterion},class_weights={class_weight})",
                                        RandomForestClassifier(
                                            n_estimators=n,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            bootstrap=bootstrap,
                                            class_weight=class_weight,
                                            max_features=max_feature,
                                            criterion=criterion,
                                            min_samples_leaf=min_sample_leaf
                                        ),
                                    )
                                models.append(current_model)
    return models
