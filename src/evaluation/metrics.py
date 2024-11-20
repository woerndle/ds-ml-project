from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import json
import os
import time
import tracemalloc 

from sklearn.base import BaseEstimator
import numpy as np

def serialize_params(params):
    serializable_params = {}
    for k, v in params.items():
        if isinstance(v, BaseEstimator):
            continue  # Skip model objects
        elif isinstance(v, dict):
            serializable_params[k] = serialize_params(v)
        elif isinstance(v, list):
            serializable_params[k] = [serialize_params(item) if isinstance(item, dict) else item for item in v]
        else:
            serializable_params[k] = v
    return serializable_params

def evaluate_model(model, X_val, y_val, label_encoder):
    """Evaluate the model and return metrics."""
    # Start tracking time and memory
    start_time = time.time()
    tracemalloc.start()
    
    # Predict on validation data
    y_pred = model.predict(X_val)
    
    # Get model parameters and exclude any model objects
    params = serialize_params(model.get_params())
    
    # Stop tracking memory and time
    current, peak = tracemalloc.get_traced_memory()  # Get current and peak memory usage
    tracemalloc.stop()
    elapsed_time = time.time() - start_time  # Time in seconds
    
    # Convert peak memory usage to MB
    peak_memory_mb = peak / (1024 * 1024)
    
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
        'elapsed_time_seconds': elapsed_time,  # Add elapsed time to metrics
        'peak_memory_usage_mb': peak_memory_mb,  # Add peak memory usage to metrics
        'model_params': params,  # Add model parameters to metrics
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
        elif isinstance(metrics[key], dict):
            # If any values in the dict are NumPy arrays, convert them
            for sub_key in metrics[key]:
                if isinstance(metrics[key][sub_key], np.ndarray):
                    metrics[key][sub_key] = metrics[key][sub_key].tolist()
        elif isinstance(metrics[key], list):
            # If any items in the list are NumPy arrays, convert them
            metrics[key] = [item.tolist() if isinstance(item, np.ndarray) else item for item in metrics[key]]

    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4)