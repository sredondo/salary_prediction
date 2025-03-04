import pandas as pd

def load_data(filepath):
    """
    Loads data from a csv file into a pandas DataFrame.

    Parameters:
    filepath (str): Path to the csv file.

    Returns:
    pandas.DataFrame: DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data successfully loaded from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def verify_data_integrity(df):
    """
    Verifies the integrity of the loaded DataFrame.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    bool: True if data integrity is verified, False otherwise.
    """
    if df is None:
        print("Dataframe is None, cannot verify integrity.")
        return False
    if not isinstance(df, pd.DataFrame):
        print("Loaded data is not a Pandas DataFrame.")
        return False
    if df.empty:
        print("DataFrame is empty.")
        return False
    print("Data integrity verified.")
    return True

def analyze_dataset_structure(df):
    """
    Performs initial analysis of the dataset structure.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    """
    if df is not None and verify_data_integrity(df):
        print("\nDataset Structure Analysis:")
        print("---------------------------")
        print(f"Dataset shape: {df.shape}")
        print("\nColumn Data Types:")
        print(df.dtypes)
        print("\nMissing Values per Column:")
        print(df.isnull().sum())
        print("\nDescriptive Statistics:")
        print(df.describe())

if __name__ == '__main__':
    # Example usage
    filepath = 'salary-prediction-challenge/data/raw/salary_data.csv'
    data = load_data(filepath)
    if verify_data_integrity(data):
        analyze_dataset_structure(data)