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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

from scipy.sparse import hstack, csr_matrix

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

def load_and_split_data(X, y, test_size=0.3, stratify=None, data_size=None):
    """Split data into training and validation sets."""
    # Adjust data size if specified
    if data_size is not None and data_size < len(X):
        # Use stratify to maintain class proportions
        X, _, y, _ = train_test_split(X, y, train_size=data_size, stratify=y, random_state=42)
    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=42)

def load_wine_review_data(data_size=None, eval_method="cross_val"):
    """Load and preprocess the wine review dataset."""
    # Ensure NLTK resources are available
    nltk.download('punkt', quiet=True)

    processed_file_path = 'data/processed/wine_reviews_processed.csv'
    required_columns = ['country', 'description', 'points', 'price', 'variety', 'winery']

    # Check if processed data exists
    if os.path.exists(processed_file_path):
        print("Loading processed data...")
        wine_df = pd.read_csv(processed_file_path)

        # Check if required columns are present
        missing_columns = [col for col in required_columns if col not in wine_df.columns]
        if missing_columns:
            print(f"Processed data is missing columns: {missing_columns}. Reprocessing raw data...")
            os.remove(processed_file_path)
            return load_wine_review_data(data_size)  # Call the function again to reprocess
    else:
        print("Processing raw data...")
        file_path = 'data/raw/wine-reviews.arff'
        columns = ['country', 'description', 'designation', 'points', 'price', 'province',
                   'region_1', 'region_2', 'variety', 'winery']

        # Load raw data and keep all features
        wine_df = load_arff_data(
            file_path,
            columns,
            drop_columns=[]  # Keep all columns
        )

        # Drop rows with missing 'country' or 'description'
        wine_df.dropna(subset=['country', 'description'], inplace=True)

        # Preprocess text data
        print("Preprocessing text data...")
        wine_df['description'] = wine_df['description'].str.lower()
        wine_df['description'] = [
            word_tokenize(text) for text in tqdm(wine_df['description'], desc="Tokenizing")
        ]
        wine_df['description'] = [
            preprocess_text(tokens) for tokens in tqdm(wine_df['description'], desc="Cleaning and lemmatizing")
        ]

        # Save processed data for future use
        os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
        wine_df.to_csv(processed_file_path, index=False)
        print("Processed data saved for future use.")

    # Include Additional Features
    # Handle numerical features: 'points', 'price'
    numerical_features = ['points', 'price']
    for col in numerical_features.copy():
        if col in wine_df.columns:
            wine_df[col] = pd.to_numeric(wine_df[col], errors='coerce')
            wine_df[col].fillna(wine_df[col].median(), inplace=True)
        else:
            print(f"Warning: Column '{col}' not found in data. It will be excluded.")
            numerical_features.remove(col)

    # Handle categorical features: 'variety', 'winery'
    categorical_features = ['variety', 'winery']
    for col in categorical_features.copy():
        if col in wine_df.columns:
            wine_df[col] = wine_df[col].fillna('Unknown')
        else:
            print(f"Warning: Column '{col}' not found in data. It will be excluded.")
            categorical_features.remove(col)

    # Filter Classes with Sufficient Samples
    min_samples_per_class = 2
    country_counts = wine_df['country'].value_counts()
    sufficient_countries = country_counts[country_counts >= min_samples_per_class].index
    wine_df = wine_df[wine_df['country'].isin(sufficient_countries)]

    # Limit to Top N Countries if Desired
    top_N_countries = 10  # Adjust as needed
    top_countries = wine_df['country'].value_counts().nlargest(top_N_countries).index
    wine_df = wine_df[wine_df['country'].isin(top_countries)]
    print(f"Data shape after selecting top {top_N_countries} countries: {wine_df.shape}")

    if data_size is not None and data_size < len(wine_df):
        # Calculate samples per class
        total_classes = wine_df['country'].nunique()
        samples_per_class = max(2, data_size // total_classes)
        
        # Sample data
        wine_df = wine_df.groupby('country', group_keys=False).apply(
            lambda x: x.sample(min(len(x), samples_per_class), random_state=42)
        ).reset_index(drop=True)
        
        # After sampling, remove any classes with less than min_samples_per_class samples
        country_counts = wine_df['country'].value_counts()
        sufficient_countries = country_counts[country_counts >= min_samples_per_class].index
        wine_df = wine_df[wine_df['country'].isin(sufficient_countries)]

    # Encode Target Variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(wine_df['country'])

    # Process Categorical Features
    # For 'variety' and 'winery', encode only the most frequent categories
    for col in categorical_features:
        top_categories = wine_df[col].value_counts().nlargest(10).index  # Keep top 10 categories
        wine_df[col] = wine_df[col].apply(lambda x: x if x in top_categories else 'Other')

    # One-hot encode the categorical features
    wine_df = pd.get_dummies(wine_df, columns=categorical_features, drop_first=True)

    # TF-IDF Vectorization
    tfidf_vect = TfidfVectorizer()
    X_tfidf = tfidf_vect.fit_transform(wine_df['description'])

    # Remove 'description' and 'country' from wine_df as they're already processed
    columns_to_drop = ['description', 'country']
    wine_df.drop(columns=[col for col in columns_to_drop if col in wine_df.columns], inplace=True)

    # Drop any remaining unprocessed object-type columns
    unprocessed_cols = wine_df.select_dtypes(include=['object']).columns.tolist()
    if unprocessed_cols:
        print(f"Warning: The following columns are of object type and will be dropped: {unprocessed_cols}")
        wine_df.drop(columns=unprocessed_cols, inplace=True)

    # Standardize numerical features
    if numerical_features:
        scaler = StandardScaler()
        numerical_data = scaler.fit_transform(wine_df[numerical_features])
        numerical_sparse = csr_matrix(numerical_data)
    else:
        numerical_sparse = None

    # Remaining columns are one-hot encoded categorical features
    categorical_cols = [col for col in wine_df.columns if col not in numerical_features]
    if categorical_cols:
        categorical_data = wine_df[categorical_cols].values.astype(np.float64)
        categorical_sparse = csr_matrix(categorical_data)
    else:
        categorical_sparse = None

    # Combine features
    feature_list = [X_tfidf]
    if numerical_sparse is not None:
        feature_list.append(numerical_sparse)
    if categorical_sparse is not None:
        feature_list.append(categorical_sparse)
    X = hstack(feature_list)

    print(f"Final X shape before splitting: {X.shape}")
    print(f"Final y shape before splitting: {y.shape}")

    if eval_method == 'cross_val':
        return X, y, label_encoder, tfidf_vect
    else:
        # Split Data into Training and Validation Sets
        print("Splitting data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        return X_train, X_val, y_train, y_val, label_encoder, tfidf_vect


def load_amazon_review_data(data_size=None, eval_method="holdout"):
    """Load and preprocess the Amazon review dataset."""
    file_path = 'data/raw/amazon-reviews/amazon_review_ID.shuf.lrn.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")

    data = pd.read_csv(file_path)
    if 'Class' not in data.columns:
        raise ValueError("The 'Class' column is missing from the Amazon reviews dataset.")

    X = data.drop(columns=['ID', 'Class'])
    y = data['Class']

    # Adjust data size if specified
    if data_size is not None and data_size < len(X):
        X = X.sample(n=data_size, random_state=42)
        y = y.loc[X.index]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize features
    X = StandardScaler().fit_transform(X)

    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X)

    if eval_method == 'holdout':
        # Split data
        X_train, X_val, y_train, y_val = load_and_split_data(X_pca, y_encoded, stratify=y_encoded)
        return X_train, X_val, y_train, y_val, label_encoder, None
    elif eval_method == 'cross_val':
        # Return full dataset for cross-validation
        return X_pca, None, y_encoded, None, label_encoder, None
    else:
        raise ValueError("Invalid evaluation method. Choose 'holdout' or 'cross_val'.")

def load_congressional_voting_data(data_size=None, eval_method="holdout"):
    """Load and preprocess the Congressional voting dataset."""
    file_path = 'data/raw/congressional-voting/CongressionalVotingID.shuf.lrn.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")

    data = pd.read_csv(file_path)
    if 'class' not in data.columns:
        raise ValueError("The 'class' column is missing from the Congressional voting dataset.")

    X = data.drop(columns=['ID', 'class']).replace({'y': 1, 'n': 0, 'unknown': np.nan})
    y = data['class']

    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(X)

    # Adjust data size if specified
    if data_size is not None and data_size < len(X_imputed):
        X_imputed = X_imputed[:data_size]
        y = y.iloc[:data_size]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize features
    X_standardized = StandardScaler().fit_transform(X_imputed)

    if eval_method == 'holdout':
        # Split data
        X_train, X_val, y_train, y_val = load_and_split_data(X_standardized, y_encoded, stratify=y_encoded)
        return X_train, X_val, y_train, y_val, label_encoder, None
    elif eval_method == 'cross_val':
        # Return full dataset for cross-validation
        return X_standardized, None, y_encoded, None, label_encoder, None
    else:
        raise ValueError("Invalid evaluation method. Choose 'holdout' or 'cross_val'.")

def load_traffic_data(data_size=None, eval_method="holdout"):
    """Load and preprocess the traffic dataset."""
    data_dir = 'data/raw/traffic-data/'
    files = ['Traffic.csv', 'TrafficTwoMonth.csv']

    # Load and combine datasets
    data_frames = []
    for file in files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")
        df = pd.read_csv(file_path)
        data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)
    data.drop_duplicates(subset=['Time', 'Date', 'Day of the week'], inplace=True)

    # Process 'Time' column
    if 'Time' in data.columns:
        data['Time'] = pd.to_datetime(data['Time'], format='%I:%M:%S %p', errors='coerce').dt.hour
        data['Time'].fillna(data['Time'].mode()[0], inplace=True)

    # Process 'Day of the week' column
    if 'Day of the week' in data.columns:
        data['Day of the week'] = data['Day of the week'].astype(str)

    # Prepare feature and target sets
    if 'Traffic Situation' not in data.columns:
        raise ValueError("'Traffic Situation' column is missing from the traffic dataset.")
    X = data.drop(columns=['Traffic Situation', 'Date'])
    y = data['Traffic Situation']

    # Handle missing values in X
    X.fillna(X.mode().iloc[0], inplace=True)

    # One-hot encode categorical features
    categorical_features = ['Day of the week']
    X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # Adjust data size if specified
    if data_size is not None and data_size < len(X):
        X = X.sample(n=data_size, random_state=42)
        y = y.loc[X.index]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Standardize features
    X_standardized = StandardScaler().fit_transform(X)

    if eval_method == 'holdout':
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_standardized, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        return X_train, X_val, y_train, y_val, label_encoder, None
    elif eval_method == 'cross_val':
        # Return full dataset for cross-validation
        return X_standardized, None, y_encoded, None, label_encoder, None
    else:
        raise ValueError("Invalid evaluation method. Choose 'holdout' or 'cross_val'.")

def load_and_preprocess_data(dataset_name, data_size=None, eval_method='holdout', n_splits=5):
    """Load and preprocess data based on the dataset name.

    Parameters:
        dataset_name (str): Name of the dataset to load.
        data_size (int, optional): Desired number of samples for the dataset.
        eval_method (str, optional): Evaluation method ('holdout' or 'cross_val').
        n_splits (int, optional): Number of splits for cross-validation.

    Returns:
        Depending on eval_method:
            If 'holdout': X_train, X_val, y_train, y_val, label_encoder, tfidf_vect
            If 'cross_val': X, y, label_encoder, tfidf_vect, cv (cross-validator)
    """
    loaders = {
        'wine_reviews': load_wine_review_data,
        'amazon_reviews': load_amazon_review_data,
        'congressional_voting': load_congressional_voting_data,
        'traffic_prediction': load_traffic_data,
    }
    if dataset_name not in loaders:
        raise ValueError("Invalid dataset name provided.")

    loader = loaders[dataset_name]
    if eval_method == 'holdout':
        X_train, X_val, y_train, y_val, label_encoder, tfidf_vect = loader(
            data_size=data_size,
            eval_method=eval_method
        )
        return X_train, X_val, y_train, y_val, label_encoder, tfidf_vect
    elif eval_method == 'cross_val':
        if loader == load_wine_review_data:
            X, y, label_encoder, tfidf_vect = loader(
                data_size=data_size,
                eval_method=eval_method
            )
        else:
            X_train, X_val, y_train, y_val, label_encoder, tfidf_vect = loader(data_size=data_size)
            if isinstance(X_train, pd.DataFrame):
                X = pd.concat([X_train, X_val], ignore_index=True)
            else:
                X = np.vstack((X_train, X_val))
            y = np.hstack((y_train, y_val))

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return X, y, label_encoder, tfidf_vect, cv
    else:
        raise ValueError("Invalid evaluation method. Choose 'holdout' or 'cross_val'.")



def get_dataset_sample_sizes():
    """Determine the number of samples for each dataset by reading the raw data files."""
    sample_sizes = {}
    
    # wine_reviews
    wine_file = 'data/raw/wine-reviews.arff'
    if os.path.exists(wine_file):
        with open(wine_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data_start = False
        count = 0
        for line in lines:
            if line.strip().lower() == '@data':
                data_start = True
                continue
            if data_start:
                if line.strip():  # Ignore empty lines
                    count += 1
        sample_sizes['wine_reviews'] = count
    else:
        print(f"File {wine_file} does not exist.")
        sample_sizes['wine_reviews'] = None
    
    # amazon_reviews
    amazon_file = 'data/raw/amazon-reviews/amazon_review_ID.shuf.lrn.csv'
    if os.path.exists(amazon_file):
        df = pd.read_csv(amazon_file)
        sample_sizes['amazon_reviews'] = len(df)
    else:
        print(f"File {amazon_file} does not exist.")
        sample_sizes['amazon_reviews'] = None
    
    # congressional_voting
    voting_file = 'data/raw/congressional-voting/CongressionalVotingID.shuf.lrn.csv'
    if os.path.exists(voting_file):
        df = pd.read_csv(voting_file)
        sample_sizes['congressional_voting'] = len(df)
    else:
        print(f"File {voting_file} does not exist.")
        sample_sizes['congressional_voting'] = None
    
    # traffic_prediction
    traffic_dir = 'data/raw/traffic-data/'
    traffic_files = ['Traffic.csv', 'TrafficTwoMonth.csv']
    traffic_data = pd.DataFrame()
    for file in traffic_files:
        file_path = os.path.join(traffic_dir, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            traffic_data = pd.concat([traffic_data, df], ignore_index=True)
        else:
            print(f"File {file_path} does not exist.")
    if not traffic_data.empty:
        # Remove duplicates based on ['Time', 'Date', 'Day of the week']
        traffic_data = traffic_data.drop_duplicates(subset=['Time', 'Date', 'Day of the week'])
        sample_sizes['traffic_prediction'] = len(traffic_data)
    else:
        print(f"No traffic data found in {traffic_dir}.")
        sample_sizes['traffic_prediction'] = None
    
    return sample_sizes

def main():
    """Main function to determine and display dataset sample sizes."""
    sample_sizes = get_dataset_sample_sizes()
    print("\nDataset Sample Sizes:")
    print("---------------------")
    for dataset, size in sample_sizes.items():
        if size is not None:
            print(f"Dataset '{dataset}': {size} samples")
        else:
            print(f"Dataset '{dataset}': Sample size could not be determined (file missing)")

if __name__ == "__main__":
    main()
