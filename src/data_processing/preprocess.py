# src/data_processing/preprocess.py

import os
import re
import pandas as pd
import numpy as np

from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# Ensure NLTK resources are available
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize global NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def input_data_check(file_path):
    """Checks to determine data integrity."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    print(f"file_path: {file_path} exists")

def load_arff_data(file_path, columns, target_column=None, drop_columns=None):
    """Load and preprocess ARFF file into a DataFrame."""
    input_data_check(file_path)
    
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
        # Replace '?' with np.nan to handle missing values
        row = [np.nan if x == '?' else x for x in row]
        data.append(row)
    
    print(f"Parsed {len(data)} rows successfully.")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Print a sample of the DataFrame to verify parsing
    print("Sample of the parsed DataFrame:")
    print(df.head())
    
    # Drop rows with missing target if specified
    if target_column:
        df = df.dropna(subset=[target_column])
    
    # Drop specified columns
    if drop_columns:
        df = df.drop(columns=drop_columns)
    
    print("DataFrame loaded with shape:", df.shape)
    return df

def preprocess_text(tokens):
    """Apply stopword removal, non-alpha filtering, and lemmatization to tokenized text."""
    processed = []
    for word, tag in pos_tag(tokens):
        word = re.sub(r'[^a-z]', '', word)  # Keep only alphabetic characters
        if word and word not in stop_words:  # Remove stop words
            pos = penn2morphy(tag)
            processed.append(lemmatizer.lemmatize(word, pos))
    return ' '.join(processed)

def penn2morphy(penntag):
    """Convert Penn Treebank tags to WordNet tags for lemmatization."""
    morphy_tag = {'N': 'n', 'J': 'a', 'V': 'v', 'R': 'r'}
    return morphy_tag.get(penntag[0], 'n')

def standardize_features(X_train, X_val):
    """Standardize numerical features."""
    scaler = StandardScaler()
    if isinstance(X_train, pd.DataFrame):
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    else:
        # Assume X_train and X_val are NumPy arrays
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    return X_train, X_val


def load_and_split_data(X, y, test_size=0.3, stratify=None):
    """Split data into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=42)

def load_wine_review_data():
    """Load and preprocess the wine review dataset."""
    processed_file_path = 'data/processed/wine_reviews_processed.csv'

    # Check if processed data exists
    if os.path.exists(processed_file_path):
        print("Loading processed data...")
        wine_df = pd.read_csv(processed_file_path)
    else:
        print("Processing raw data...")
        file_path = 'data/raw/wine-reviews.arff'
        columns = ['country', 'description', 'designation', 'points', 'price', 'province',
                   'region_1', 'region_2', 'variety', 'winery']
        
        # Load raw data and drop unnecessary columns
        wine_df = load_arff_data(
            file_path,
            columns,
            drop_columns=['designation', 'points', 'price', 'province', 'region_1',
                          'region_2', 'variety', 'winery']
        )

        # Drop rows with missing 'country' or 'description'
        wine_df.dropna(subset=['country', 'description'], inplace=True)

        # Preprocess text data
        print("Preprocessing text data...")
        wine_df['description'] = wine_df['description'].str.lower()
        wine_df['description'] = [word_tokenize(text) for text in tqdm(wine_df['description'], desc="Tokenizing")]
        wine_df['description'] = [
            preprocess_text(tokens) for tokens in tqdm(wine_df['description'], desc="Cleaning and lemmatizing")
        ]

        # Save processed data for future use
        os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
        wine_df.to_csv(processed_file_path, index=False)
        print("Processed data saved for future use.")

    # Filter out classes with fewer than 2 instances
    country_counts = wine_df['country'].value_counts()
    common_countries = country_counts[country_counts > 1].index
    wine_df = wine_df[wine_df['country'].isin(common_countries)]
    print(f"Data shape after removing single-instance countries: {wine_df.shape}")

    # Encode target variable (country)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(wine_df['country'])

    # TF-IDF vectorization on the processed text data
    tfidf_vect = TfidfVectorizer()
    X_tfidf = tfidf_vect.fit_transform(wine_df['description'])

    # Split the data into training and validation sets
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_tfidf, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_val, y_train, y_val, label_encoder, tfidf_vect

def load_amazon_review_data():
    """Load and preprocess the Amazon review dataset with PCA."""
    train_data = pd.read_csv('data/raw/amazon-reviews/amazon_review_ID.shuf.lrn.csv')
    if 'Class' not in train_data:
        raise ValueError("The 'Class' column is missing from the Amazon reviews dataset.")

    X = train_data.drop(columns=['ID', 'Class'])
    y = train_data['Class']

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_val, y_train, y_val = load_and_split_data(X, y_encoded, stratify=y_encoded)

    # Standardize features
    X_train, X_val = standardize_features(X_train, X_val)

    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)  # You can adjust this number
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)

    return X_train, X_val, y_train, y_val, label_encoder, None



def load_congressional_voting_data():
    """Load and preprocess the Congressional voting dataset."""
    data = pd.read_csv('data/raw/congressional-voting/CongressionalVotingID.shuf.lrn.csv')
    if 'class' not in data:
        raise ValueError("The 'class' column is missing from the Congressional voting dataset.")
    
    X = data.drop(columns=['ID', 'class']).replace({'y': 1, 'n': 0, 'unknown': np.nan})
    y = data['class']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(X)
    
    # Convert back to DataFrame with original columns
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Encode labels
    label_encoder = LabelEncoder()
    #y_encoded = label_encoder.fit_transform(y)
    # Ensure y is a 1D array
    y_encoded = label_encoder.fit_transform(y)#.ravel()

    X_train, X_val, y_train, y_val = load_and_split_data(X_imputed, y_encoded, stratify=y_encoded)
    
    # Standardize features
    X_train, X_val = standardize_features(X_train, X_val)
    
    return X_train, X_val, y_train, y_val, label_encoder, None

def load_traffic_data():
    """Load and preprocess the traffic dataset."""
    data_dir = 'data/raw/traffic-data/'
    files = ['Traffic.csv', 'TrafficTwoMonth.csv']
    data = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in files], ignore_index=True)
    
    # Convert 'Time' to hour
    if 'Time' in data:
        data['Time'] = pd.to_datetime(data['Time'], format='%I:%M:%S %p', errors='coerce').dt.hour
        data['Time'].fillna(data['Time'].mode()[0], inplace=True)
    
    # Convert 'Date' to datetime and extract features
    if 'Date' in data:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['DayOfWeek'] = data['Date'].dt.day_name()
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
    
    if 'Traffic Situation' not in data:
        raise ValueError("'Traffic Situation' column is missing from the traffic dataset.")
    
    X = data.drop(columns=['Traffic Situation', 'Date'])
    y = data['Traffic Situation']
    
    # Handle missing values in X
    X.fillna(X.mode().iloc[0], inplace=True)
    
    # Encode categorical features
    categorical_features = ['DayOfWeek']
    X = pd.get_dummies(X, columns=categorical_features)

        # Identify all categorical features
    categorical_features = ['Day of the week', 'Time']

    # Convert 'Time' to categorical if it's still in string format
    if 'Time' in X and X['Time'].dtype == object:
        categorical_features.append('Time')

    # Use one-hot encoding
    X = pd.get_dummies(X, columns=categorical_features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_val, y_train, y_val = load_and_split_data(X, y_encoded, stratify=y_encoded)
    
    # Feature scaling
    X_train, X_val = standardize_features(X_train, X_val)
    
    return X_train, X_val, y_train, y_val, label_encoder, None

def load_and_preprocess_data(dataset_name):
    """Load and preprocess data based on the dataset name."""
    loaders = {
        'wine_reviews': load_wine_review_data,
        'amazon_reviews': load_amazon_review_data,
        'congressional_voting': load_congressional_voting_data,
        'traffic_prediction': load_traffic_data,
    }
    if dataset_name not in loaders:
        raise ValueError("Invalid dataset name provided.")
    return loaders[dataset_name]()
