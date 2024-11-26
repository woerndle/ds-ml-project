#!/bin/bash
# Define the datasets, models, and evaluation methods
datasets=("wine_reviews" "amazon_reviews" "congressional_voting" "traffic_prediction")
models=("rf" "svm" "knn")
eval_methods=("holdout" "cross_val")

# Loop over each dataset, model, and evaluation method combination
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for eval_method in "${eval_methods[@]}"; do
      echo "Running experiments for dataset: $dataset, model: $model, evaluation method: $eval_method"
      python src/experiments/run_experiments.py --dataset "$dataset" --model "$model" --eval_method "$eval_method"
      echo "Completed experiments for dataset: $dataset, model: $model, evaluation method: $eval_method"     
      echo "------------------------------------------------------------"
    done
  done
done

# Run the visualization script after all experiments are completed
python src/evaluation/visualisation.py
python src/evaluation/process_results.py
echo "Visualization saved"
echo "All experiments completed."