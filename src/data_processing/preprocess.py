import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    # Load the training data
    train_data = pd.read_csv('data/raw/amazon-reviews/amazon_review_ID.shuf.lrn.csv')
    # Load the test data without labels
    test_data = pd.read_csv('data/raw/amazon-reviews/amazon_review_ID.shuf.tes.csv')

    # Check if 'Class' column exists in train data
    if 'Class' not in train_data.columns:
        raise ValueError("The 'Class' column is not found in the training dataset.")

    # Separate features and labels
    X = train_data.drop(columns=['ID', 'Class'])  # Drop ID and Class columns from features
    y = train_data['Class']  # Class labels from training data
    
    # Convert categorical features to numerical if necessary
    X = pd.get_dummies(X)

    # Check class distribution
    print("Original class distribution in y:")
    print(y.value_counts())

    # Ensure there is more than one class in the dataset
    if y.nunique() < 2:
        raise ValueError("The training set contains only one class.")

    # Train-test split (30% for validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Load test data features only, dropping the ID column
    X_test = test_data.drop(columns=['ID'])
    X_test = pd.get_dummies(X_test)  # Ensure test features match training features
    
    # Ensure test data has the same feature columns as training data
    X_test = X_test.reindex(columns=X.columns, fill_value=0)

    return X_train, X_val, y_train, y_val, X_test
