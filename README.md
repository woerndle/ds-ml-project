# ds-ml-project

Data Science TU Wien Machine Learning Project

## Overview

This project is designed to automate and streamline the process of running machine learning experiments on various datasets using different classification algorithms and evaluation methods. It provides tools for data preprocessing, model training, evaluation, and visualization, facilitating comprehensive analysis and comparison of model performance across datasets.

## Table of Contents

- [Datasets](#datasets)
- [Models](#models)
- [Evaluation Methods](#evaluation-methods)
- [Features and Capabilities](#features-and-capabilities)
- [Repository Structure](#repository-structure)
- [How to Use the Repository](#how-to-use-the-repository)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Running Experiments](#running-experiments)
  - [Adjusting Data Size](#adjusting-data-size)
  - [Visualizing Results](#visualizing-results)
  - [Custom Experiments](#custom-experiments)
- [Detailed Description](#detailed-description)
  - [Data Processing](#data-processing)
  - [Models](#models-1)
  - [Experiment Running](#experiment-running)
  - [Evaluation and Metrics](#evaluation-and-metrics)
  - [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Datasets

The project includes the following datasets:

- **Wine Reviews**: Contains wine reviews with features like description, country, points, price, variety, and winery.
- **Amazon Reviews**: A dataset of Amazon product reviews with various attributes.
- **Congressional Voting**: Data on U.S. congressional voting patterns.
- **Traffic Prediction**: Traffic data including time, date, day of the week, and traffic situation.

## Models

Implemented machine learning models:

- **Support Vector Machines (SVM)**: Various kernels (linear, polynomial, RBF, sigmoid) and hyperparameters.
- **K-Nearest Neighbors (KNN)**: Different numbers of neighbors, weight functions, algorithms, and distance metrics.
- **Random Forests (RF)**: Varying numbers of trees, depths, splitting criteria, and other hyperparameters.

## Evaluation Methods

Supported evaluation methods:

- **Holdout**: Splits the dataset into training and validation sets.
- **Cross-Validation**: Performs stratified K-fold cross-validation.

## Features and Capabilities

- **Automated Experimentation**: Run experiments across combinations of datasets, models, and evaluation methods.
- **Data Preprocessing**: Load, clean, and preprocess data for model training.
- **Model Training and Evaluation**: Train models with specified hyperparameters and evaluate using various metrics.
- **Visualization**: Generate plots for model performance, including confusion matrices and ROC curves.
- **Result Aggregation**: Collect and aggregate results from experiments.
- **Performance Metrics**: Measure and report execution time and memory usage.

## Repository Structure

ds-ml-project/ ├── data/ │ ├── raw/ │ │ ├── amazon-reviews/ │ │ ├── congressional-voting/ │ │ ├── traffic-data/ │ │ └── wine-reviews.arff │ └── processed/ │ └── wine_reviews_processed.csv ├── output_results_holdout/ ├── output_results_cross_val/ ├── plots/ ├── run_all_experiments.sh ├── src/ │ ├── data_processing/ │ │ └── preprocess.py │ ├── evaluation/ │ │ ├── metrics.py │ │ └── visualisation.py │ ├── experiments/ │ │ └── run_experiments.py │ └── models/ │ ├── knn.py │ ├── random_forest.py │ └── svm.py └── README.md




- **data/**: Contains raw and processed datasets.
- **output_results_holdout/** and **output_results_cross_val/**: Directories where experiment results are saved.
- **plots/**: Directory for generated plots.
- **run_all_experiments.sh**: Shell script to automate running all experiments.
- **src/**: Source code directory.
  - **data_processing/preprocess.py**: Functions for loading and preprocessing data.
  - **evaluation/metrics.py**: Functions for evaluating models and saving metrics.
  - **evaluation/visualisation.py**: Scripts for generating visualizations.
  - **experiments/run_experiments.py**: Main script to run experiments.
  - **models/**: Contains model definitions for KNN, Random Forest, and SVM.

## How to Use the Repository

### Prerequisites

- **Python 3.x**
- Required Python packages (see `requirements.txt` if available)
- **NLTK data packages**: WordNet, stopwords, etc.

### Setup

1. **Clone the repository**:

   ```
   git clone https://github.com/yourusername/ds-ml-project.git
   cd ds-ml-project
Install the required Python packages:

It's recommended to use a virtual environment.



pip install -r requirements.txt
Download NLTK data:

The scripts automatically download NLTK data if not present. Ensure internet connectivity for the first run.

Running Experiments
To reproduce the results, execute the run_all_experiments.sh script. This script runs experiments for all combinations of datasets, models, and evaluation methods.

Navigate to the root directory in your terminal and execute:



./run_all_experiments.sh
Note: Running all experiments can be time-consuming, especially with large datasets. To reduce execution time, you can limit the amount of data used by adjusting the data_size parameter in run_experiments.py.

Adjusting Data Size
To use a reduced amount of data, modify the data_size parameter in the main function in run_experiments.py. For example, set data_size=10000 to use 10,000 samples per dataset.

Visualizing Results
After running experiments, results are saved in the output_results_holdout/ and output_results_cross_val/ directories. To generate summary visualizations, run the visualization script:



python src/evaluation/visualisation.py
This will create plots and summary tables in the plots/ directory.

Custom Experiments
You can run custom experiments by directly calling run_experiments.py with specific arguments.

Example:



python src/experiments/run_experiments.py --dataset wine_reviews --model svm --eval_method cross_val
Available arguments:

--dataset: Name of the dataset (wine_reviews, amazon_reviews, congressional_voting, traffic_prediction)
--model: Model to use (svm, knn, rf)
--eval_method: Evaluation method (holdout, cross_val)
--subset: Use a subset of data for testing (e.g., --subset 1000)
Detailed Description
Data Processing
The preprocess.py script in src/data_processing/ handles loading and preprocessing of datasets. This includes:

Loading Datasets: From various formats like CSV and ARFF.
Handling Missing Values: Data imputation and cleaning.
Text Preprocessing: Tokenization, lemmatization, and stopword removal for textual data.
Feature Encoding: Label encoding and one-hot encoding for categorical features.
Feature Scaling: Standardization of numerical features.
Dimensionality Reduction: Using PCA for high-dimensional data.
Data Splitting: Into training and validation sets or preparing for cross-validation.
Models
The models are defined in the src/models/ directory.

SVM (svm.py)
Defines SVM models with various kernels and hyperparameters.

KNN (knn.py)
Defines KNN models with different configurations.

Random Forest (random_forest.py)
Defines Random Forest models with varying hyperparameters.

Experiment Running
The run_experiments.py script orchestrates the running of experiments.

Argument Parsing: Determines dataset, model, evaluation method, and subset size.
Data Loading: Calls preprocessing functions to load and prepare data.
Model Retrieval: Gets the list of models based on the specified type.
Training and Evaluation: Trains models and evaluates them using the specified evaluation method.
Result Saving: Saves metrics and generates plots for each model.
Evaluation and Metrics
The metrics.py script in src/evaluation/ handles model evaluation.

Metrics Calculation: Accuracy, F1-score, confusion matrix, ROC curve, etc.
Performance Tracking: Measures elapsed time and memory usage.
Result Serialization: Saves metrics to JSON files for analysis.
Visualization
The visualisation.py script in src/evaluation/ generates visualizations.

Data Collection: Gathers metrics from result directories.
Plot Generation: Creates plots like box plots, scatter plots, and bar charts.
Summary Tables: Generates tables summarizing model performance.
Output: Saves plots and tables in the plots/ directory.
Contributing
If you wish to contribute to this project, please fork the repository and submit a pull request.