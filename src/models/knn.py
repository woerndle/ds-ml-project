# src/models/knn.py
from sklearn.neighbors import KNeighborsClassifier

def get_knn_models():
    # Define KNN models with different hyperparameters
    models = [
        ("KNN (k=3)", KNeighborsClassifier(n_neighbors=3)),
        ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
        ("KNN (k=10)", KNeighborsClassifier(n_neighbors=10)),
        ("KNN (distance-weighted, k=5)", KNeighborsClassifier(n_neighbors=5, weights='distance')),
        ("KNN (manhattan distance, k=5)", KNeighborsClassifier(n_neighbors=5, metric='manhattan')),
    ]
    return models
