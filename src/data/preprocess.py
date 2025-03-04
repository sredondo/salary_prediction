import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(df):
    """
    Cleans the DataFrame by handling missing values and duplicates.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    # Handle missing values (example: fill numerical NaN with mean, categorical with mode)
    for column in df.columns:
        if df[column].isnull().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                df[column].fillna(df[column].mode()[0], inplace=True) # mode()[0] to handle potential multiple modes

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    print("Data cleaning completed.")
    return df

def engineer_features(df):
    """
    Engineers new features from existing columns.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    pandas.DataFrame: DataFrame with engineered features.
    """
    # Example: Create experience level from Job Title (this is a placeholder, more sophisticated engineering needed)
    df['experience_level'] = df['Job Title'].apply(lambda title: 'Senior' if 'senior' in title.lower() else 'Mid' if 'sr' not in title.lower() and 'lead' not in title.lower() and  'manager' not in title.lower() else 'Entry')

    print("Feature engineering completed.")
    return df

def split_data(df, target_column='avg_salary', test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    target_column (str): Name of the target column.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.

    Returns:
    tuple: (X_train, X_test, y_train, y_test) - Splitted DataFrames and Series.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("Data splitting completed.")
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Example usage
    filepath = 'salary_prediction/data/salary_data.csv'
    df = pd.read_csv(filepath) # Directly load for testing purposes

    cleaned_df = clean_data(df.copy()) # Use copy to avoid modifying original df during testing
    engineered_df = engineer_features(cleaned_df.copy())
    X_train, X_test, y_train, y_test = split_data(engineered_df.copy())

    print("\nCleaned Data Head:")
    print(cleaned_df.head())
    print("\nEngineered Data Head:")
    print(engineered_df.head())
    print("\nShapes of split data:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")