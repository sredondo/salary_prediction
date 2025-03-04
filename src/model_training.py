import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

def train_random_forest_model(X_train, y_train):
    """
    Trains a RandomForestRegressor model.

    Parameters:
    X_train (pandas.DataFrame): Training features.
    y_train (pandas.Series): Training target.

    Returns:
    sklearn.ensemble.RandomForestRegressor: Trained RandomForestRegressor model.
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest model training completed.")
    return model

def create_dummy_model(X_train, y_train):
    """
    Creates a baseline DummyRegressor model.

    Parameters:
    X_train (pandas.DataFrame): Training features.
    y_train (pandas.Series): Training target.

    Returns:
    sklearn.dummy.DummyRegressor: Trained DummyRegressor model.
    """
    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(X_train, y_train)
    print("Dummy model training completed.")
    return dummy_model

if __name__ == '__main__':
    # Example usage (assuming data_preprocessing and data_loading are done)
    from data_loading import load_data
    from data_preprocessing import clean_data, engineer_features, split_data

    filepath = 'salary_prediction/data/salary_data.csv'
    df = load_data(filepath)
    if df is not None:
        cleaned_df = clean_data(df.copy())
        engineered_df = engineer_features(cleaned_df.copy())
        X_train, X_test, y_train, y_test = split_data(engineered_df.copy())

        rf_model = train_random_forest_model(X_train, y_train)
        dummy_model = create_dummy_model(X_train, y_train)

        print("\nRandom Forest Model:", rf_model)
        print("\nDummy Model:", dummy_model)