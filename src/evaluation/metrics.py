# src/evaluation/metrics.py

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np
import json
import os

def evaluate_model(model, X_val, y_val, label_encoder):
    """Evaluate the model and return metrics."""
    # Predict on validation data
    y_pred = model.predict(X_val)
    
    # Get prediction probabilities or decision scores
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_val)
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_val)
        # If binary classification, reshape y_scores
        if len(y_scores.shape) == 1:
            y_scores = np.vstack([-y_scores, y_scores]).T
    else:
        y_scores = None  # Some models do not provide probabilities or scores

    # Decode labels if necessary
    y_test_decoded = label_encoder.inverse_transform(y_val)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'classification_report': classification_report(y_val, y_pred),
        'y_score': y_scores,
        'y_pred_decoded': y_pred_decoded,
    }
    return metrics, y_test_decoded

def save_metrics(metrics, model_name, dataset_name):
    output_dir = os.path.join("output_results", dataset_name, model_name)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{model_name}_metrics.json")

    # Convert any NumPy arrays to lists before saving
    for key in metrics:
        if isinstance(metrics[key], np.ndarray):
            metrics[key] = metrics[key].tolist()

    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {results_file}")
