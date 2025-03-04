import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as st
from sklearn.utils import resample

def calculate_confidence_interval(y_true, y_pred, metric_func, confidence=0.95, n_bootstrap=1000):
    """
    Calcula intervalos de confianza para una métrica usando bootstrapping.

    Parameters:
    y_true (array-like): Valores reales
    y_pred (array-like): Valores predichos
    metric_func (callable): Función que calcula la métrica
    confidence (float): Nivel de confianza (default: 0.95)
    n_bootstrap (int): Número de muestras bootstrap

    Returns:
    tuple: (valor inferior del IC, valor superior del IC)
    """
    # Convertir a arrays de numpy
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    
    bootstrap_metrics = []
    n_samples = len(y_true_array)
    
    for _ in range(n_bootstrap):
        # Generar índices para bootstrap
        indices = resample(range(n_samples), replace=True)
        
        # Calcular métrica para esta muestra
        sample_metric = metric_func(
            y_true_array[indices],
            y_pred_array[indices]
        )
        bootstrap_metrics.append(sample_metric)
    
    # Calcular intervalos de confianza
    alpha = (1 - confidence)
    ci_lower = np.percentile(bootstrap_metrics, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_metrics, (1 - alpha/2) * 100)
    
    return ci_lower, ci_upper

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo y calcula métricas con intervalos de confianza.

    Parameters:
    model: Modelo entrenado
    X_test (pandas.DataFrame): Features de prueba
    y_test (pandas.Series): Variable objetivo de prueba

    Returns:
    dict: Diccionario con métricas e intervalos de confianza
    """
    # Convertir a arrays de numpy
    y_test_array = np.array(y_test)
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas base
    mse = mean_squared_error(y_test_array, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_array, y_pred)
    
    # Calcular intervalos de confianza para RMSE
    rmse_ci = calculate_confidence_interval(
        y_test_array,
        y_pred,
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    )
    
    # Calcular intervalos de confianza para R²
    r2_ci = calculate_confidence_interval(
        y_test_array,
        y_pred,
        r2_score
    )
    
    # Calcular error porcentual medio absoluto
    mape = np.mean(np.abs((y_test_array - y_pred) / y_test_array)) * 100
    mape_ci = calculate_confidence_interval(
        y_test_array,
        y_pred,
        lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    )
    
    # Imprimir resultados detallados
    print("\nResultados de la Evaluación:")
    print(f"RMSE: {rmse:.2f} (IC 95%: [{rmse_ci[0]:.2f}, {rmse_ci[1]:.2f}])")
    print(f"R²: {r2:.3f} (IC 95%: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}])")
    print(f"MAPE: {mape:.2f}% (IC 95%: [{mape_ci[0]:.2f}%, {mape_ci[1]:.2f}%])")
    
    return {
        'rmse': rmse,
        'rmse_ci': f"[{rmse_ci[0]:.2f}, {rmse_ci[1]:.2f}]",
        'r2': r2,
        'r2_ci': f"[{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]",
        'mape': mape,
        'mape_ci': f"[{mape_ci[0]:.2f}%, {mape_ci[1]:.2f}%]"
    }

if __name__ == '__main__':
    # Ejemplo de uso
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    
    # Generar datos de ejemplo
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    metrics = evaluate_model(model, X_test, y_test)
    print("\nMétricas calculadas:", metrics)