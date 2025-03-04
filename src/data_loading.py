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

if __name__ == '__main__':
    # Example usage
    filepath = 'salary_prediction/data/salary_data.csv'
    data = load_data(filepath)
    if data is not None:
        print(data.head())