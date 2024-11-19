#!/bin/bash

# Define the datasets and models
datasets=("wine_reviews" "amazon_reviews" "congressional_voting" "traffic_prediction")
models=("rf" "svm" "knn")

# Loop over each dataset and model combination
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "Running experiments for dataset: $dataset, model: $model"
    python src/experiments/run_experiments.py --dataset "$dataset" --model "$model"
    echo "Completed experiments for dataset: $dataset, model: $model"
    echo "Visualisation saved"
    echo "------------------------------------------------------------"
  done
done
python src/evaluation/visualisation.py
echo "All experiments completed."
