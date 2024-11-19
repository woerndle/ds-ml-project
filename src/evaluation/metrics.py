from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import json
import os
import time
import tracemalloc  # Import tracemalloc for memory tracking

def evaluate_model(model, X_val, y_val, label_encoder):
    """Evaluate the model and return metrics."""
    # Start tracking time and memory
    start_time = time.time()
    tracemalloc.start()
    
    # Predict on validation data
    y_pred = model.predict(X_val)
    
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
