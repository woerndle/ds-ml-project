import os

# Define the directory structure and files to be created
structure = {
    "data": ["raw/README.md", "processed/README.md"],
    "notebooks": ["eda.ipynb"],
    "src": [
        "__init__.py",
        "data_processing/__init__.py",
        "data_processing/preprocess.py",
        "models/__init__.py",
        "models/svm.py",
        "models/random_forest.py",
        "models/nn.py",
        "experiments/__init__.py",
        "experiments/run_experiments.py",
        "evaluation/__init__.py",
        "evaluation/metrics.py",
        "utils/__init__.py",
        "utils/helper_functions.py",
    ],
    "config": ["config.yaml"],
    "tests": [
        "__init__.py",
        "test_data_processing.py",
        "test_models.py",
        "test_evaluation.py",
    ],
}

# Define the contents of each file
file_contents = {
    "README.md": "# Project Name\n\nThis project runs machine learning experiments on multiple datasets using different classifiers (SVM, Random Forest, Neural Network). The results are evaluated based on metrics like precision, recall, F1 score, and accuracy.\n\n## Structure\n\n- `data/`: Raw and processed datasets.\n- `notebooks/`: Jupyter notebooks for EDA.\n- `src/`: Source code for preprocessing, model training, and evaluation.\n- `config/`: Configuration file for dataset paths and model settings.\n- `tests/`: Unit tests for modules.\n\n## Setup\n\n1. Install dependencies:\n   ```bash\n   pip install -r requirements.txt\n   ```\n\n2. Run experiments:\n   ```bash\n   python src/experiments/run_experiments.py\n   ```\n\n## License\n\n[MIT License](LICENSE)\n",
    "src/data_processing/preprocess.py": '"""\nData Preprocessing Module\n\nThis module contains functions for preprocessing datasets used in the experiments.\n"""\n\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\n\ndef load_data(file_path):\n    """\n    Loads data from a specified file path.\n\n    Parameters:\n    file_path (str): Path to the data file.\n\n    Returns:\n    pd.DataFrame: Loaded data.\n    """\n    return pd.read_csv(file_path)\n\ndef preprocess_data(df):\n    """\n    Preprocesses the dataset.\n\n    Parameters:\n    df (pd.DataFrame): The raw data.\n\n    Returns:\n    pd.DataFrame: Processed data.\n    """\n    df = df.dropna()\n    return df\n\ndef split_data(df, test_size=0.3, random_state=42):\n    """\n    Splits data into training and testing sets.\n\n    Parameters:\n    df (pd.DataFrame): The preprocessed data.\n    test_size (float): Fraction of data to reserve for testing.\n    random_state (int): Seed for reproducibility.\n\n    Returns:\n    tuple: (X_train, X_test, y_train, y_test)\n    """\n    X = df.drop("target", axis=1)\n    y = df["target"]\n    return train_test_split(X, y, test_size=test_size, random_state=random_state)\n',
    "src/models/svm.py": '"""\nSVM Model\n\nThis module provides functions to train and evaluate an SVM classifier.\n"""\n\nfrom sklearn.svm import SVC\n\ndef train_svm(X_train, y_train, kernel=\'linear\'):\n    """\n    Trains an SVM classifier.\n\n    Parameters:\n    X_train (pd.DataFrame): Training features.\n    y_train (pd.Series): Training labels.\n    kernel (str): Kernel type for SVM.\n\n    Returns:\n    SVC: Trained SVM model.\n    """\n    model = SVC(kernel=kernel)\n    model.fit(X_train, y_train)\n    return model\n\ndef predict_svm(model, X_test):\n    """\n    Predicts labels using the trained SVM model.\n\n    Parameters:\n    model (SVC): Trained SVM model.\n    X_test (pd.DataFrame): Test features.\n\n    Returns:\n    np.ndarray: Predicted labels.\n    """\n    return model.predict(X_test)\n',
    "src/experiments/run_experiments.py": '"""\nExperiment Runner\n\nThis script orchestrates the entire experiment pipeline.\n"""\n\nimport os\nfrom src.data_processing import preprocess\nfrom src.models import svm, random_forest, nn\nfrom src.evaluation import metrics\nimport yaml\n\ndef load_config(config_path):\n    with open(config_path, \'r\') as file:\n        return yaml.safe_load(file)\n\ndef main():\n    config = load_config("config/config.yaml")\n    \n    for dataset_name, dataset_path in config[\'datasets\'].items():\n        print(f"\\nRunning experiments on {dataset_name}...")\n        df = preprocess.load_data(dataset_path)\n        df = preprocess.preprocess_data(df)\n        X_train, X_test, y_train, y_test = preprocess.split_data(df)\n        for model_name in config[\'models\']:\n            if model_name == "SVM":\n                model = svm.train_svm(X_train, y_train, kernel=config[\'svm_kernel\'])\n            elif model_name == "RandomForest":\n                model = random_forest.train_rf(X_train, y_train)\n            elif model_name == "NN":\n                model = nn.train_nn(X_train, y_train)\n            y_pred = model.predict(X_test)\n            print(f"\\nResults for {model_name} on {dataset_name}:")\n            metrics.evaluate(y_test, y_pred)\n\nif __name__ == "__main__":\n    main()\n',
    "src/evaluation/metrics.py": '"""\nEvaluation Metrics\n\nThis module provides functions to evaluate model performance.\n"""\n\nfrom sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score\n\ndef evaluate(y_true, y_pred):\n    """\n    Prints the evaluation metrics for model predictions.\n\n    Parameters:\n    y_true (np.ndarray): True labels.\n    y_pred (np.ndarray): Predicted labels.\n    """\n    print("Precision:", precision_score(y_true, y_pred, average=\'macro\'))\n    print("Recall:", recall_score(y_true, y_pred, average=\'macro\'))\n    print("F1 Score:", f1_score(y_true, y_pred, average=\'macro\'))\n    print("Accuracy:", accuracy_score(y_true, y_pred))\n',
    "config/config.yaml": 'datasets:\n  wine_reviews: "data/raw/wine_reviews.csv"\n  traffic: "data/raw/traffic_data.csv"\n  amazon_reviews: "data/raw/amazon_reviews.csv"\n  congressional_voting: "data/raw/congressional_voting.csv"\n\nmodels:\n  - SVM\n  - RandomForest\n  - NN\n\nsvm_kernel: "linear"\n',
    "requirements.txt": "pandas\nscikit-learn\npyyaml\nmatplotlib\n",
}



# Function to create directory structure
def create_structure(base_path, structure):
    for key, files in structure.items():
        dir_path = os.path.join(base_path, key)
        
        # Create the main directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Create each file in the directory
        for file in files:
            file_path = os.path.join(dir_path, file)
            # Ensure that all intermediate directories exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create the file if it doesn't already exist
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    pass  # Just create an empty file

# Run the function
base_path = os.getcwd()  # Set this to your repository root if different
create_structure(base_path, structure)

