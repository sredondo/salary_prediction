import pandas as pd
from src.features.build_features import preprocess_and_engineer_features
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Limpia el DataFrame manejando valores faltantes, duplicados y tipos de datos.
    Aplica conversiones específicas para columnas numéricas y categóricas.
    """
    df_cleaned = df.copy()
    
    # Definir columnas numéricas y sus tipos de datos esperados
    numeric_columns = {
        'rating': 'float64',
        'hourly': 'int64',
        'employer_provided': 'int64',
        'min_salary': 'float64',
        'max_salary': 'float64',
        'avg_salary': 'float64',
        'same_state': 'int64',
        'age': 'float64'
    }
    
    # Convertir y validar columnas numéricas
    for col, dtype in numeric_columns.items():
        if col in df_cleaned.columns:
            try:
                # Primero convertir a numérico permitiendo NaN
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                
                # Para columnas que deben ser enteros, redondear antes de convertir
                if dtype == 'int64':
                    df_cleaned[col] = df_cleaned[col].round().astype('float64')
                
                # Validar rangos para valores específicos
                if col == 'rating':
                    df_cleaned.loc[df_cleaned[col] < 0, col] = np.nan
                    df_cleaned.loc[df_cleaned[col] > 5, col] = 5.0
                elif col in ['hourly', 'employer_provided', 'same_state']:
                    df_cleaned.loc[~df_cleaned[col].isin([0, 1]), col] = np.nan
                elif col == 'age':
                    df_cleaned.loc[df_cleaned[col] < 0, col] = np.nan
                elif 'salary' in col:
                    df_cleaned.loc[df_cleaned[col] < 0, col] = np.nan
            except Exception as e:
                print(f"Error al procesar columna {col}: {e}")
                df_cleaned[col] = np.nan
    
    # Definir y procesar columnas categóricas
    categorical_columns = {
        'job_title': str,
        'company_name': str,
        'location': str,
        'size': str,
        'type_of_ownership': str,
        'industry': str,
        'sector': str,
        'company_txt': str,
        'job_state': str
    }
    
    # Convertir y limpiar columnas categóricas
    for col, dtype in categorical_columns.items():
        if col in df_cleaned.columns:
            # Convertir a string y limpiar espacios
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            # Reemplazar valores vacíos o 'nan' por NaN
            df_cleaned[col] = df_cleaned[col].replace(['', 'nan', 'None', 'NaN'], np.nan)
    
    # Manejar valores faltantes
    for column in df_cleaned.columns:
        if df_cleaned[column].isnull().any():
            if column in numeric_columns:
                # Para columnas numéricas, usar la mediana en lugar de la media
                df_cleaned[column].fillna(df_cleaned[column].median(), inplace=True)
            else:
                # Para categóricas, usar el modo
                df_cleaned[column].fillna(df_cleaned[column].mode()[0], inplace=True)
    
    # Eliminar duplicados
    df_cleaned.drop_duplicates(inplace=True)
    
    # Asegurar tipos de datos finales
    for col, dtype in numeric_columns.items():
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(dtype)
    
    return df_cleaned

def encode_categorical_variables(df):
    """
    Codifica variables categóricas usando one-hot encoding.
    """
    df_encoded = df.copy()
    
    # Identificar columnas categóricas
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    
    # Aplicar one-hot encoding
    for column in categorical_columns:
        dummies = pd.get_dummies(df_encoded[column], prefix=column, dummy_na=False)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        df_encoded.drop(column, axis=1, inplace=True)
    
    return df_encoded

def scale_numerical_features(df, exclude_columns=None):
    """
    Escala características numéricas usando StandardScaler.
    """
    if exclude_columns is None:
        exclude_columns = []
    
    df_scaled = df.copy()
    
    # Identificar columnas numéricas
    numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns
    columns_to_scale = [col for col in numeric_columns if col not in exclude_columns]
    
    if columns_to_scale:
        scaler = StandardScaler()
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
    
    return df_scaled

def split_data(df, target_column='avg_salary', test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    print(f"\nSplitting data con columnas: {df.columns.tolist()}")
    print(f"Target column: {target_column}")
    
    if target_column not in df.columns:
        raise ValueError(f"Columna objetivo '{target_column}' no encontrada en el DataFrame")
    
    # Aplicar codificación y escalado antes del split
    df_processed = encode_categorical_variables(df.copy())
    df_processed = scale_numerical_features(df_processed, exclude_columns=[target_column])
    
    # Separar features y target
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]
    
    # Convertir y a array de numpy
    y = y.values
    
    # Split los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Shapes después del split:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Ejemplo de uso
    from make_dataset import load_data
    
    # Cargar datos
    df = load_data('salary-prediction-challenge/data/raw/salary_data.csv')
    if df is not None:
        # Procesar datos
        df_processed = preprocess_and_engineer_features(df.copy()) # Pass a copy to avoid modifying original df
        
        
        
        # Split datos
        X_train, X_test, y_train, y_test = split_data(df_processed)
        
        print("\nProcesamiento completado:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")