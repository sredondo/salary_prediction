import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def normalize_column_names(df):
    """
    Normaliza los nombres de las columnas del DataFrame.
    - Elimina espacios extra
    - Convierte a lowercase
    - Reemplaza espacios con _
    - Remueve caracteres especiales
    """
    print("\n=== Normalizando nombres de columnas ===")
    print("Nombres originales:", df.columns.tolist())
    
    # Crear un mapa de nombres originales a normalizados
    normalized_names = {
        col: col.strip().lower().replace(' ', '_').replace('\n', '')
        for col in df.columns
    }
    
    df = df.rename(columns=normalized_names)
    print("Nombres normalizados:", df.columns.tolist())
    return df

def print_debug_info(df, step=""):
    """Función auxiliar para debug"""
    print(f"\n=== Debug Info ({step}) ===")
    print("Columnas disponibles:")
    print(df.columns.tolist())
    print(f"Shape: {df.shape}")

def build_experience_features(df):
    """
    Construye características basadas en el título del trabajo y la experiencia.
    """
    print_debug_info(df, "build_experience_features - inicio")
    
    # Buscar la columna correcta para el título del trabajo
    possible_title_columns = ['job_title', 'jobtitle', 'job']
    title_column = None
    
    for col in possible_title_columns: # Iterate through possible title columns
        if col in df.columns: # Check if the column exists in the dataframe
            title_column = col          # If it exists, assign it to title_column
            break # Exit loop once title column is found
    print(f"Possible title columns: {possible_title_columns}") # DEBUG
    print(f"Available columns: {df.columns.tolist()}") # DEBUG
    
    if title_column is None:
        print("WARNING: No se encontró columna de título de trabajo")
        print("Columnas disponibles:", df.columns.tolist())
        df['experience_level'] = 'Unknown'
        return df
    
    # Categorías de experiencia basadas en el título
    print(f"Columns in build_experience_features: {df.columns.tolist()}") # DEBUG
    print(f"title_column in build_experience_features: {title_column}") # DEBUG

    df['experience_level'] = df[title_column].astype(str).apply(
        lambda x: 'Senior' if any(term in x.lower() for term in ['senior', 'sr.', 'sr', 'lead', 'principal'])
        else 'Junior' if any(term in x.lower() for term in ['junior', 'jr.', 'jr', 'associate'])
        else 'Mid'
    )
    
    print_debug_info(df, "build_experience_features - fin")
    return df

def build_company_features(df):
    """
    Construye características basadas en información de la compañía.
    """
    print_debug_info(df, "build_company_features - inicio")
    
    # Mapa de posibles nombres de columnas
    column_maps = {
        'size': ['size', 'company_size'],
        'ownership': ['type_of_ownership', 'ownership_type', 'company_type']
    }
    
    # Procesar tamaño de la empresa
    size_col = next((col for col in column_maps['size'] if col in df.columns), None)
    if size_col:
        size_order = {
            '1 to 50 employees': 1,
            '51 to 200 employees': 2,
            '201 to 500 employees': 3,
            '501 to 1000 employees': 4,
            '1001 to 5000 employees': 5,
            '5001 to 10000 employees': 6,
            '10000+ employees': 7
        }
        df['company_size_code'] = df[size_col].map(size_order)
    else:
        print("WARNING: No se encontró columna de tamaño de empresa")
        df['company_size_code'] = -1

    # Procesar tipo de propiedad
    ownership_col = next((col for col in column_maps['ownership'] if col in df.columns), None)
    if ownership_col:
        le = LabelEncoder()
        df['ownership_code'] = le.fit_transform(df[ownership_col].fillna('Unknown'))
    else:
        print("WARNING: No se encontró columna de tipo de propiedad")
        df['ownership_code'] = -1
    
    print_debug_info(df, "build_company_features - fin")
    return df

def build_location_features(df):
    """
    Construye características basadas en la ubicación.
    """
    print_debug_info(df, "build_location_features - inicio")
    
    # Buscar la columna de ubicación
    location_cols = ['location', 'company_location']
    location_col = next((col for col in location_cols if col in df.columns), None)
    
    if location_col:
        df['state'] = df[location_col].astype(str).apply(
            lambda x: x.split(', ')[-1] if ', ' in x else x
        )
        le = LabelEncoder()
        df['state_code'] = le.fit_transform(df['state'])
    else:
        print("WARNING: No se encontró columna de ubicación")
        df['state'] = 'Unknown'
        df['state_code'] = -1
    
    print_debug_info(df, "build_location_features - fin")
    return df

def build_salary_features(df):
    """
    Construye características basadas en información salarial.
    """
    print_debug_info(df, "build_salary_features - inicio")
    
    salary_columns = {
        'min': ['min_salary', 'MinSalary', 'min salary'],
        'max': ['max_salary', 'MaxSalary', 'max salary'],
        'avg': ['avg_salary', 'AvgSalary', 'avg salary']
    }
    
    # Encontrar las columnas correctas
    found_cols = {}
    for key, possible_cols in salary_columns.items():
        found_cols[key] = next((col for col in possible_cols if col in df.columns), None)
    
    if all(found_cols.values()):
        df['salary_range'] = df[found_cols['max']] - df[found_cols['min']]
        df['salary_category'] = pd.qcut(
            df[found_cols['avg']], 
            q=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
    else:
        print("WARNING: No se encontraron todas las columnas de salario necesarias")
        df['salary_range'] = 0
        df['salary_category'] = 'Unknown'
    
    print_debug_info(df, "build_salary_features - fin")
    return df

def preprocess_and_engineer_features(df):
    """
    Aplica todas las transformaciones de features al DataFrame.
    """
    print("\nIniciando preprocesamiento con shape:", df.shape)
    print("Columnas iniciales:", df.columns.tolist())
    
    # Normalizar nombres de columnas
    df = normalize_column_names(df)
    
    print("\n=== Columnas luego de normalizar en preprocess_and_engineer_features ===")
    print("Columnas disponibles:")
    print(df.columns.tolist())

    df_processed = df.copy()
    
    # Aplicar transformaciones
    df_processed = build_experience_features(df_processed)
    df_processed = build_company_features(df_processed)
    df_processed = build_location_features(df_processed)
    df_processed = build_salary_features(df_processed)
    
    # Convertir variables categóricas a dummies
    categorical_cols = ['experience_level', 'salary_category']
    for col in categorical_cols:
        if col in df_processed.columns:
            dummies = pd.get_dummies(df_processed[col], prefix=col)
            # Asegurar que las columnas dummy sean numéricas
            for dummy_col in dummies.columns:
                dummies[dummy_col] = dummies[dummy_col].astype(int)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed = df_processed.drop(columns=[col])
    
    # Codificar la columna de compañía si existe
    company_cols = ['company_name', 'company']
    company_col = next((col for col in company_cols if col in df_processed.columns), None)
    if company_col:
        le = LabelEncoder()
        df_processed['company_code'] = le.fit_transform(df_processed[company_col].fillna('Unknown'))
    
    # Eliminar columnas originales
    columns_to_drop = [
        col for col in df_processed.columns 
        if any(term in col.lower() for term in ['job_title', 'location', 'size', 'ownership', 'industry', 'sector', 'company_name', 'company'])
        or col in ['state', 'experience_level', 'salary_category']
    ]
    
    df_processed = df_processed.drop(columns=[
        col for col in columns_to_drop if col in df_processed.columns
    ])
    
    # Ensure all remaining columns are numeric
    non_numeric_columns = []
    for col in df_processed.columns:
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            print(f"Converting non-numeric column {col} to numeric...")
            try:
                df_processed[col] = pd.to_numeric(df_processed[col])
            except ValueError:
                print(f"Warning: Could not convert {col} to numeric. Column will be dropped.")
                non_numeric_columns.append(col)
    
    # Drop all non-numeric columns that couldn't be converted
    if non_numeric_columns:
        print(f"Dropping non-numeric columns: {non_numeric_columns}")
        df_processed = df_processed.drop(columns=non_numeric_columns)
    
    print("\nPreprocesamiento completado")
    print("Columnas finales:", df_processed.columns.tolist())
    print("Shape final:", df_processed.shape)
    
    # Verify all columns are numeric
    non_numeric_cols = df_processed.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        raise ValueError(f"Las siguientes columnas aún no son numéricas: {non_numeric_cols.tolist()}")
    
    return df_processed

if __name__ == '__main__':
    # Ejemplo de uso
    data = pd.read_csv('../data/raw/salary_data.csv')
    processed_data = preprocess_and_engineer_features(data)
    print("\nShape final:", processed_data.shape)
