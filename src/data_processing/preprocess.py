# src/data_processing/preprocess.py
import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(dataset_name):
    if dataset_name == 'amazon_reviews':
        # Load Amazon Reviews dataset
        train_data = pd.read_csv('data/raw/amazon-reviews/amazon_review_ID.shuf.lrn.csv')
        test_data = pd.read_csv('data/raw/amazon-reviews/amazon_review_ID.shuf.tes.csv')

        if 'Class' not in train_data.columns:
            raise ValueError("The 'Class' column is not found in the training dataset.")

        X = train_data.drop(columns=['ID', 'Class'])
        y = train_data['Class']
        X = pd.get_dummies(X)

        print("Original class distribution in y:")
        print(y.value_counts())

        if y.nunique() < 2:
            raise ValueError("The training set contains only one class.")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)


        result = (X_train, X_val, y_train, y_val)
        print(f"Number of items being returned: {len(result)}")
        return result

    elif dataset_name == 'congressional_voting':
        # Load Congressional Voting dataset
        data = pd.read_csv('data/raw/congressional-voting/CongressionalVotingID.shuf.lrn.csv')

        if 'class' not in data.columns:
            raise ValueError("The 'Class' column is not found in the dataset.")

        X = data.drop(columns=['ID', 'class'])
        y = data['class']

        # Convert categorical features to numerical
        X = X.replace({'y': 1, 'n': 0, 'unknown': np.nan})
        X = X.fillna(X.mode().iloc[0])  # Fill missing values with the mode

        print("Original class distribution in y:")
        print(y.value_counts())

        if y.nunique() < 2:
            raise ValueError("The dataset contains only one class.")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        result = (X_train, X_val, y_train, y_val)
        print(f"Number of items being returned: {len(result)}")
        return result

    elif dataset_name == 'wine_reviews':
        # **Load the Wine Reviews dataset manually**
        file_path = 'data/raw/wine-reviews.arff'
        columns = ['country', 'description', 'designation', 'points', 'price',
                   'province', 'region_1', 'region_2', 'variety', 'winery']
        
        # Read the ARFF file
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Find the start of data
        data_start_idx = next(i for i, line in enumerate(lines) if line.strip().lower() == '@data') + 1
        data_lines = lines[data_start_idx:]
        
        data = []
        
        # Regular expression to handle ARFF data lines with commas and quotes correctly
        pattern = re.compile(r"""('(?:\\.|[^'])*'|[^,]+)""")
        
        for line in data_lines:
            # Split the line using the regular expression
            row = [x.strip().strip("'") for x in pattern.findall(line.strip())]
            data.append(row)
        
        # Create DataFrame
        wine_df = pd.DataFrame(data, columns=columns)
        
         # Convert data types
        wine_df['points'] = pd.to_numeric(wine_df['points'], errors='coerce')
        wine_df['price'] = pd.to_numeric(wine_df['price'], errors='coerce')

        # Drop rows with missing target variable 'points'
        wine_df = wine_df.dropna(subset=['points'])

        # Fill missing values in features
        wine_df = wine_df.fillna({
            'country': 'Unknown',
            'designation': 'Unknown',
            'price': wine_df['price'].median(),
            'province': 'Unknown',
            'region_1': 'Unknown',
            'region_2': 'Unknown',
            'variety': 'Unknown',
            'winery': 'Unknown'
        })

        # Drop 'description' column as it may be too text-heavy
        wine_df = wine_df.drop(columns=['description'])

        # **Limit categories for high-cardinality features**
        categorical_features = ['country', 'designation', 'province', 'region_1', 'region_2', 'variety', 'winery']

        for col in categorical_features:
            # Determine the top N categories
            top_n = 20 if wine_df[col].nunique() > 30 else wine_df[col].nunique()
            top_categories = wine_df[col].value_counts().nlargest(top_n).index
            # Replace less frequent categories with 'Other'
            wine_df[col] = wine_df[col].apply(lambda x: x if x in top_categories else 'Other')

        # Features and target
        X = wine_df.drop(columns=['points'])
        y = wine_df['points']

        # One-hot encode categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Standardize features
        scaler = StandardScaler(with_mean=False)  # Set with_mean=False for sparse data
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        result = (X_train, X_val, y_train, y_val)
        print(f"Number of items being returned: {len(result)}")
        return result

    elif dataset_name == 'traffic_prediction':
        # Load the manually downloaded dataset
        data_dir = 'data/raw/traffic-data/'
        file1 = os.path.join(data_dir, 'Traffic.csv')
        file2 = os.path.join(data_dir, 'TrafficTwoMonth.csv')  # Ensure the filename matches

        # Load the datasets
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        # Combine the datasets
        data = pd.concat([df1, df2], ignore_index=True)

        # Parse 'Time' column to extract hour
        if 'Time' in data.columns:
            data['Time'] = pd.to_datetime(data['Time'], format='%I:%M:%S %p', errors='coerce').dt.hour
            data['Time'].fillna(method='ffill', inplace=True)
        else:
            pass

        # Parse 'Date' column and extract features
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data['DayOfWeek'] = data['Date'].dt.dayofweek  # Numerical day of the week (0=Monday)
            data['Month'] = data['Date'].dt.month
            data['Day'] = data['Date'].dt.day
        else:
            pass

        # **Handle categorical variables**
        categorical_columns = ['Day of the week']  # Include 'Day of the week' as a categorical column

        # Encode 'Day of the week' using label encoding
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype('category').cat.codes

        # Define features and target
        if 'Traffic Situation' in data.columns:
            X = data.drop(columns=['Traffic Situation', 'Date'])  # Drop 'Date' column
            y = data['Traffic Situation']
        else:
            raise ValueError("'Traffic Situation' column not found in the dataset.")

        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))
        y = y.fillna(y.mode()[0])

        # **Ensure all columns are numeric**
        print("Data types after preprocessing:")
        print(X.dtypes)

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        # **Encode target variable if necessary**
        if y_train.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_val = le.transform(y_val)
        else:
            pass

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        result = (X_train, X_val, y_train, y_val)
        print(f"Number of items being returned: {len(result)}")
        return result

    else:
        raise ValueError("Invalid dataset name provided.")