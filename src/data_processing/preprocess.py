# src/data_processing/preprocess.py
import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import re
import pandas as pd
def input_data_check(file_path):
    """Checks to determine data integrety
    """
    # Check if the file exists
    

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    print(f"file_path:  {file_path} exists")

def load_arff_data(file_path, columns, target_column=None, drop_columns=None):
    """Load and preprocess ARFF file into a DataFrame."""

    
    # Open and read the file lines
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Check if the @data tag is present in the file
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == '@data':
            data_start_idx = i + 1
            break
    if data_start_idx is None:
        raise ValueError("The ARFF file does not contain a '@data' section.")
    
    # Extract data lines
    data_lines = lines[data_start_idx:]
    print(f"Data lines start from line index {data_start_idx}.")

    # Regular expression to handle ARFF data lines with commas and quotes correctly
    pattern = re.compile(r"""('(?:\\.|[^'])*'|[^,]+)""")
    
    # Parse data rows
    data = []
    for line in data_lines:
        row = [x.strip().strip("'") for x in pattern.findall(line.strip())]
        if len(row) != len(columns):
            print(f"Skipping malformed row: {row}")  # Log rows that don't match expected column length
            continue
        data.append(row)
    
    print(f"Parsed {len(data)} rows successfully.")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Convert data types
    df = df.apply(pd.to_numeric, errors='coerce', axis=0)

    # Drop rows with missing target if specified
    if target_column:
        df = df.dropna(subset=[target_column])
    
    # Drop specified columns
    if drop_columns:
        df = df.drop(columns=drop_columns)
    
    print("DataFrame loaded with shape:", df.shape)
    return df


def preprocess_categorical(df, categorical_features, top_n_limit=20):
    """Limit high-cardinality categories in categorical features."""
    for col in categorical_features:
        top_n = min(df[col].nunique(), top_n_limit)
        top_categories = df[col].value_counts().nlargest(top_n).index
        df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
    return pd.get_dummies(df, drop_first=True)

def standardize_features(X_train, X_val, sparse=False):
    """Standardize features, optionally handling sparse data."""
    scaler = StandardScaler(with_mean=not sparse)
    return scaler.fit_transform(X_train), scaler.transform(X_val)

def load_and_split_data(X, y, test_size=0.3, stratify=None):
    """Split data into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=42)

def load_wine_review_data():
    file_path = 'data/raw/wine-reviews.arff'
    columns = ['country', 'description', 'designation', 'points', 'price', 'province', 'region_1', 'region_2', 'variety', 'winery']
    wine_df = load_arff_data(file_path, columns, target_column='points', drop_columns=['description'])

    wine_df['price'].fillna(wine_df['price'].median(), inplace=True)
    wine_df.fillna({'country': 'Unknown', 'designation': 'Unknown', 'province': 'Unknown', 'region_1': 'Unknown', 'region_2': 'Unknown', 'variety': 'Unknown', 'winery': 'Unknown'}, inplace=True)

    X, y = wine_df.drop(columns=['points']), wine_df['points']
    X = preprocess_categorical(X, ['country', 'designation', 'province', 'region_1', 'region_2', 'variety', 'winery'])
    X_train, X_val, y_train, y_val = load_and_split_data(X, y)
    X_train, X_val = standardize_features(X_train, X_val, sparse=True)
    
    return X_train, X_val, y_train, y_val

def load_amazon_review_data():
    train_data = pd.read_csv('data/raw/amazon-reviews/amazon_review_ID.shuf.lrn.csv')
    if 'Class' not in train_data:
        raise ValueError("The 'Class' column is missing from the Amazon reviews dataset.")

    X, y = train_data.drop(columns=['ID', 'Class']), train_data['Class']
    X = pd.get_dummies(X)
    X_train, X_val, y_train, y_val = load_and_split_data(X, y, stratify=y)
    X_train, X_val = standardize_features(X_train, X_val)

    return X_train, X_val, y_train, y_val

def load_congressional_voting_data():
    data = pd.read_csv('data/raw/congressional-voting/CongressionalVotingID.shuf.lrn.csv')
    if 'class' not in data:
        raise ValueError("The 'class' column is missing from the Congressional voting dataset.")
    
    X = data.drop(columns=['ID', 'class']).replace({'y': 1, 'n': 0, 'unknown': np.nan}).fillna(method='ffill')
    y = data['class']
    X_train, X_val, y_train, y_val = load_and_split_data(X, y, stratify=y)
    X_train, X_val = standardize_features(X_train, X_val)

    return X_train, X_val, y_train, y_val

def load_traffic_data():
    data_dir = 'data/raw/traffic-data/'
    files = [os.path.join(data_dir, f) for f in ['Traffic.csv', 'TrafficTwoMonth.csv']]
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    if 'Time' in data:
        data['Time'] = pd.to_datetime(data['Time'], format='%I:%M:%S %p', errors='coerce').dt.hour.fillna(method='ffill')
    if 'Date' in data:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['DayOfWeek'], data['Month'], data['Day'] = data['Date'].dt.dayofweek, data['Date'].dt.month, data['Date'].dt.day

    if 'Traffic Situation' not in data:
        raise ValueError("'Traffic Situation' column is missing from the traffic dataset.")
    
    X, y = data.drop(columns=['Traffic Situation', 'Date']), data['Traffic Situation']
    X.fillna(X.mean(numeric_only=True), inplace=True)
    y.fillna(y.mode()[0], inplace=True)

    X_train, X_val, y_train, y_val = load_and_split_data(X, y, stratify=y)
    X_train, X_val = standardize_features(X_train, X_val)
    
    if y_train.dtype == 'object':
        le = LabelEncoder()
        y_train, y_val = le.fit_transform(y_train), le.transform(y_val)

    return X_train, X_val, y_train, y_val

def load_and_preprocess_data(dataset_name):
    loaders = {
        'wine_reviews': load_wine_review_data,
        'amazon_reviews': load_amazon_review_data,
        'congressional_voting': load_congressional_voting_data,
        'traffic_prediction': load_traffic_data,
    }
    if dataset_name not in loaders:
        raise ValueError("Invalid dataset name provided.")
    
    return loaders[dataset_name]()