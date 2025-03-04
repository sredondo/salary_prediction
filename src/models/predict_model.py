import pandas as pd
import joblib

def load_model(model_path):
    """
    Loads a trained model from the specified path.

    Parameters:
    model_path (str): Path to the trained model file.

    Returns:
    model: Loaded machine learning model.
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None

def predict_salary(model, features):
    """
    Predicts salary using the loaded model.

    Parameters:
    model: Trained machine learning model.
    features (pandas.DataFrame): DataFrame of features for prediction.

    Returns:
    numpy.ndarray: Predicted salaries.
    """
    try:
        predictions = model.predict(features)
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

if __name__ == '__main__':
    # Example usage (assuming model is trained and saved)
    from src.data.make_dataset import load_data
    from src.data.preprocess import clean_data, engineer_features, split_data
    from src.models.train_model import train_random_forest_model

    # Load data, preprocess, split and train model (similar to notebook)
    filepath = 'salary-prediction-challenge/data/raw/salary_data.csv'
    df = load_data(filepath)
    if df is not None:
        cleaned_df = clean_data(df.copy())
        engineered_df = engineer_features(cleaned_df.copy())
        X_train, X_test, y_train, y_test = split_data(engineered_df.copy())
        rf_model = train_random_forest_model(X_train, y_train)

        # Example: Save model (for demonstration, you might save it after hyperparameter tuning etc.)
        model_path = 'salary_prediction_model.joblib'
        joblib.dump(rf_model, model_path)
        print(f"Trained model saved to {model_path}")

        # Load model
        loaded_model = load_model(model_path)
        if loaded_model:
            # Example prediction (using first 5 rows of test set for demonstration)
            sample_features = X_test.head()
            predictions = predict_salary(loaded_model, sample_features)
            if predictions is not None:
                print("\nSample Predictions:")
                print(predictions)