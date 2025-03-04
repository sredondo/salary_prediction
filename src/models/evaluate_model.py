import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import scipy.stats as st

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a model and prints performance metrics with confidence intervals.

    Parameters:
    model: Trained model.
    X_test (pandas.DataFrame): Test features.
    y_test (pandas.Series): Test target.

    Returns:
    dict: Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Calculate confidence intervals (example for RMSE and R2)
    rmse_ci = confidence_interval(y_test, y_pred, metric_func=lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_ci = confidence_interval(y_test, y_pred, metric_func=r2_score)

    print(f"Model Evaluation Metrics:")
    print(f"  RMSE: {rmse:.2f} ({rmse_ci})")
    print(f"  R^2: {r2:.2f} ({r2_ci})")

    return {
        'rmse': rmse,
        'rmse_ci': rmse_ci,
        'r2': r2,
        'r2_ci': r2_ci
    }

def confidence_interval(y_true, y_pred, metric_func, confidence_level=0.95, n_bootstrap=1000):
    """
    Calculates confidence interval for a given metric using bootstrapping.

    Parameters:
    y_true (pandas.Series or numpy.array): True target values.
    y_pred (pandas.Series or numpy.array): Predicted target values.
    metric_func (callable): Metric function to evaluate.
    confidence_level (float): Confidence level for CI.
    n_bootstrap (int): Number of bootstrap samples.

    Returns:
    str: Confidence interval as a string.
    """
    bootstrap_values = []
    n_samples = len(y_true)
    rng = np.random.RandomState(42) # for reproducibility

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n_samples, n_samples) # resample with replacement
        bootstrap_y_true = y_true.iloc[indices] if isinstance(y_true, pd.Series) else y_true[indices]
        bootstrap_y_pred = y_pred[indices]
        bootstrap_values.append(metric_func(bootstrap_y_true, bootstrap_y_pred))

    alpha = (1 - confidence_level)
    ci = st.t.interval(confidence_level, len(bootstrap_values)-1, 
                       loc=np.mean(bootstrap_values), 
                       scale=st.sem(bootstrap_values)) # Using t-interval for more robust CI with smaller samples

    return f"CI95%: [{ci[0]:.2f}, {ci[1]:.2f}]"


if __name__ == '__main__':
    # Example usage (assuming models and split data are available)
    from data_loading import load_data
    from data_preprocessing import clean_data, engineer_features, split_data
    from model_training import train_random_forest_model, create_dummy_model

    filepath = 'salary_prediction/data/salary_data.csv'
    df = load_data(filepath)
    if df is not None:
        cleaned_df = clean_data(df.copy())
        engineered_df = engineer_features(cleaned_df.copy())
        X_train, X_test, y_train, y_test = split_data(engineered_df.copy())

        rf_model = train_random_forest_model(X_train, y_train)
        dummy_model = create_dummy_model(X_train, y_train)

        print("\nEvaluating Random Forest Model:")
        rf_metrics = evaluate_model(rf_model, X_test, y_test)

        print("\nEvaluating Dummy Model:")
        dummy_metrics = evaluate_model(dummy_model, X_test, y_test)