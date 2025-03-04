import pandas as pd

def build_experience_level(df):
    """
    Engineers experience level feature from Job Title.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    pandas.DataFrame: DataFrame with experience level feature.
    """
    df['experience_level'] = df['Job Title'].apply(lambda title: 'Senior' if 'senior' in title.lower() else 'Mid' if 'sr' not in title.lower() and 'lead' not in title.lower() and  'manager' not in title.lower() else 'Entry')
    print("Experience level feature built.")
    return df

# Additional feature engineering functions can be added here

if __name__ == '__main__':
    # Example usage (assuming data is loaded)
    from src.data.make_dataset import load_data
    filepath = 'salary-prediction-challenge/data/raw/salary_data.csv'
    df = load_data(filepath)
    if df is not None:
        df_with_experience = build_experience_level(df.copy())
        print(df_with_experience[['Job Title', 'experience_level']].head())