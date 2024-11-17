#!/bin/bash

# Define the datasets and models
datasets=("wine_reviews" "amazon_reviews" "congressional_voting" "traffic_prediction")
models=("svm" "knn" "rf")

# Loop over each dataset and model combination
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "Running experiments for dataset: $dataset, model: $model"
    python src/experiments/run_experiments.py --dataset "$dataset" --model "$model"
    echo "Completed experiments for dataset: $dataset, model: $model"
    echo "------------------------------------------------------------"
  done
done

echo "All experiments completed."
