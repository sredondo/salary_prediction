import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def train_random_forest_model(X_train, y_train):
    """
    Entrena un modelo RandomForestRegressor con búsqueda de hiperparámetros.

    Parameters:
    X_train (pandas.DataFrame): Features de entrenamiento
    y_train (pandas.Series): Variable objetivo de entrenamiento

    Returns:
    sklearn.ensemble.RandomForestRegressor: Modelo entrenado
    """
    # Validar tipos de datos
    if not all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("Todas las características deben ser numéricas. Verifique el preprocesamiento de datos.")
    if not np.issubdtype(y_train.dtype, np.number):
        raise ValueError("La variable objetivo debe ser numérica. Verifique el preprocesamiento de datos.")

    # Definir grid de hiperparámetros
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Inicializar modelo base
    rf_base = RandomForestRegressor(random_state=42)

    # Realizar búsqueda de hiperparámetros con validación cruzada
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1
    )

    # Entrenar modelo
    grid_search.fit(X_train, y_train)
    
    print("\nMejores hiperparámetros encontrados:")
    print(grid_search.best_params_)
    print(f"\nMejor score de validación cruzada: {np.sqrt(-grid_search.best_score_):.2f} RMSE")

    # Retornar el mejor modelo
    return grid_search.best_estimator_

def create_dummy_model(X_train, y_train):
    """
    Crea y entrena un modelo base (DummyRegressor) que predice la media.

    Parameters:
    X_train (pandas.DataFrame): Features de entrenamiento
    y_train (pandas.Series): Variable objetivo de entrenamiento

    Returns:
    sklearn.dummy.DummyRegressor: Modelo base entrenado
    """
    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(X_train, y_train)
    print("Modelo base (DummyRegressor) entrenado")
    return dummy_model

def save_model(model, filepath):
    """
    Guarda el modelo entrenado en disco.

    Parameters:
    model: Modelo entrenado
    filepath (str): Ruta donde guardar el modelo
    """
    try:
        joblib.dump(model, filepath)
        print(f"Modelo guardado en: {filepath}")
    except Exception as e:
        print(f"Error guardando el modelo: {e}")

def load_model(filepath):
    """
    Carga un modelo guardado desde disco.

    Parameters:
    filepath (str): Ruta del modelo guardado

    Returns:
    El modelo cargado
    """
    try:
        model = joblib.load(filepath)
        print(f"Modelo cargado desde: {filepath}")
        return model
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return None

if __name__ == '__main__':
    # Ejemplo de uso
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generar datos de ejemplo
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelos
    print("\nEntrenando modelo base...")
    dummy_model = create_dummy_model(X_train, y_train)
    
    print("\nEntrenando Random Forest...")
    rf_model = train_random_forest_model(X_train, y_train)
    
    # Guardar modelos
    save_model(dummy_model, 'dummy_model.joblib')
    save_model(rf_model, 'rf_model.joblib')