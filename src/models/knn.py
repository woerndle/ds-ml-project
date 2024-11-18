# src/models/knn.py
from sklearn.neighbors import KNeighborsClassifier


def get_knn_models():

    neighbors = [3, 5, 7, 10]
    weight_functions = ["uniform", "distance"]
    computation_algorithms = ["ball_tree", "kd_tree", "brute"]
    metrics = ["euclidean", "cityblock", "manhattan"]
    models = []

    # Define KNN models with different hyperparameters
    for weight_function in weight_functions:
        for computation_algorithm in computation_algorithms:
            for metric in metrics:
                for number_of_neigbors in neighbors:
                    current_model = (
                        f"KNN, ({weight_function}, {computation_algorithm}, {metric} ,k={number_of_neigbors})",
                        KNeighborsClassifier(
                            n_neighbors=number_of_neigbors,
                            algorithm=computation_algorithm,
                            metric=metric,
                            weights=weight_function,
                        ),
                    )
                    models.append(current_model)

    # print(models)
    # models = [
    #     ("KNN (k=3)", KNeighborsClassifier(n_neighbors=3)),
    #     ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
    #     ("KNN (k=10)", KNeighborsClassifier(n_neighbors=10)),
    #     (
    #         "KNN (distance-weighted, k=5)",
    #         KNeighborsClassifier(n_neighbors=5, weights="distance"),
    #     ),
    #     (
    #         "KNN (manhattan distance, k=5)",
    #         KNeighborsClassifier(n_neighbors=5, metric="manhattan"),
    #     ),
    # ]
    return models
