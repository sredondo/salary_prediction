# Instrucciones Detalladas para el Desafío de Predicción de Salarios

## Objetivo Principal
Desarrollar un modelo predictivo en Python que pronostique el salario de individuos basado en un conjunto de datos proporcionado, cumpliendo con **todos** los requisitos obligatorios y al menos una característica opcional.

## 1. Configuración Inicial del Proyecto

### 1.1. Creación del Repositorio
- Crear un repositorio público en GitHub con un nombre descriptivo (ej: `salary-prediction-challenge`)
- Inicializar con un archivo README.md básico y un archivo .gitignore para Python
- Configurar el repositorio siguiendo las mejores prácticas de Git

### 1.2. Estructura de Directorios
Implementar la siguiente estructura de directorios:
```
salary-prediction-challenge/
├── data/                      # Directorio para datos
│   ├── raw/                   # Datos originales sin procesar
│   └── processed/             # Datos procesados
├── notebooks/                 # Jupyter notebooks
│   └── main_notebook.ipynb    # Notebook principal con resultados finales
├── src/                       # Código fuente modularizado
│   ├── __init__.py
│   ├── data/                  # Módulos para procesamiento de datos
│   │   ├── __init__.py
│   │   ├── make_dataset.py    # Carga y preprocesamiento inicial
│   │   └── preprocess.py      # Transformaciones y feature engineering
│   ├── features/              # Módulos para ingeniería de características
│   │   ├── __init__.py
│   │   └── build_features.py  # Construcción de características
│   ├── models/                # Módulos para modelos
│   │   ├── __init__.py
│   │   ├── train_model.py     # Entrenamiento de modelos
│   │   ├── predict_model.py   # Predicción con modelos entrenados
│   │   └── evaluate_model.py  # Evaluación de modelos
│   └── visualization/         # Módulos para visualización
│       ├── __init__.py
│       └── visualize.py       # Funciones de visualización
├── tests/                     # Pruebas unitarias (si se implementan)
│   └── __init__.py
├── .gitignore                 # Especifica archivos a ignorar por Git
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación principal del proyecto
```

### 1.3. Gestión de Dependencias
- Crear un archivo `requirements.txt` con todas las dependencias necesarias
- Especificar versiones exactas para asegurar reproducibilidad
- Dependencias mínimas requeridas:
  ```
  pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
ipykernel>=6.0.0
scipy>=1.9.0
  ```

## 2. Procesamiento y Exploración de Datos

### 2.1. Carga de Datos
En `src/data/make_dataset.py`:
- Implementar función para cargar el dataset CSV proporcionado
- Verificar la integridad de los datos cargados
- Realizar un análisis inicial de la estructura del dataset

### 2.2. Análisis Exploratorio de Datos (EDA)
En `notebooks/eda.ipynb` (opcional) o `notebooks/main_notebook.ipynb`:
- Examinar la estructura del dataset (tipos de datos, valores nulos, estadísticas descriptivas)
- Analizar la distribución de la variable objetivo (salario)
- Explorar relaciones entre variables predictoras y el salario
- Identificar outliers y valores atípicos
- Generar visualizaciones para entender los datos:
  - Distribución de salarios
  - Relación entre experiencia y salario
  - Distribución de salarios por nivel educativo y género
  - Análisis de correlación entre variables

### 2.3. Preprocesamiento de Datos
En `src/data/preprocess.py`:
- Implementar funciones para:
  - Manejo de valores faltantes
  - Codificación de variables categóricas (one-hot, label encoding)
  - Procesamiento de texto de títulos de trabajo y descripciones
  - Normalización/estandarización de variables numéricas
  - Detección y tratamiento de outliers

## 3. Ingeniería de Características

### 3.1. Creación de Características
En `src/features/build_features.py`:
- Desarrollar funciones para extraer características útiles:
  - Extracción de información de textos usando técnicas de NLP
  - Creación de características derivadas basadas en la experiencia
  - Transformación de variables educativas en rangos numéricos
  - Generación de interacciones entre características relevantes
  - Creación de variables binarias para aspectos importantes del trabajo

### 3.2. Selección de Características
En `src/features/build_features.py`:
- Implementar métodos para seleccionar características relevantes:
  - Eliminación de características con alta correlación
  - Selección basada en importancia de características
  - Selección basada en análisis estadístico

## 4. Desarrollo de Modelos

### 4.1. Modelo Base
En `src/models/train_model.py`:
- Implementar un modelo base (DummyRegressor):
  - Usar la estrategia "mean" o "median"
  - Entrenar y evaluar como línea base de comparación
  - Calcular intervalos de confianza para las métricas

### 4.2. Modelo Principal
En `src/models/train_model.py`:
- Implementar al menos un modelo de regresión avanzado (ej: RandomForest, Gradient Boosting)
- Aplicar validación cruzada para evaluar el rendimiento
- Incluir cálculo de intervalos de confianza para las métricas:
  - Usar bootstrapping o técnicas similares
  - Reportar intervalos para RMSE, MAE, R²

### 4.3. Evaluación de Modelos
En `src/models/evaluate_model.py`:
- Implementar funciones para:
  - Calcular múltiples métricas de rendimiento (RMSE, MAE, R², etc.)
  - Generar intervalos de confianza para todas las métricas
  - Comparar el modelo con la línea base
  - Validar las suposiciones del modelo (según corresponda)

## 5. Características Opcionales (Implementar al menos UNA)

### 5.1. Explicación de Suposiciones del Modelo
- Documentar las suposiciones del modelo seleccionado
- Implementar pruebas para validar si se cumplen las suposiciones
- Interpretar los resultados de las pruebas y sus implicaciones

### 5.2. Ajuste de Hiperparámetros
- Implementar optimización de hiperparámetros usando Optuna o métodos similares
- Definir espacio de búsqueda adecuado
- Visualizar resultados del proceso de optimización

### 5.3. Validación Cruzada Avanzada
- Implementar técnicas de validación cruzada (k-fold, stratified, etc.)
- Analizar la estabilidad del modelo a través de diferentes particiones
- Reportar intervalos de confianza en las métricas de validación cruzada

### 5.4. Visualización de Relaciones
- Crear visualizaciones avanzadas para explorar relaciones entre características y salario
- Implementar gráficos de dependencia parcial
- Visualizar interacciones entre características

### 5.5. Interpretación del Modelo (SHAP)
- Implementar análisis SHAP para explicar la contribución de cada característica
- Visualizar valores SHAP globales y locales
- Interpretar los resultados en el contexto del problema

### 5.6. Modelos Avanzados
- Implementar modelos más complejos (Random Forest, Neural Networks)
- Comparar rendimiento con modelos más simples
- Analizar ventajas/desventajas de la complejidad adicional

### 5.7. Ensemble de Modelos
- Implementar técnicas de ensemble (bagging, boosting, stacking)
- Combinar predicciones de múltiples modelos
- Evaluar la mejora en rendimiento

### 5.8. Bloqueo de Dependencias
- Configurar un sistema de gestión de dependencias (pipenv, poetry, pdm)
- Garantizar la reproducibilidad exacta del entorno

### 5.9. API REST
- Desarrollar una API con FastAPI para servir predicciones
- Implementar endpoints para predicción individual y por lotes
- Documentar el uso de la API

### 5.10. Interfaz de Usuario
- Crear una interfaz web con Streamlit o Gradio
- Permitir al usuario ingresar características y obtener predicciones
- Visualizar la interpretación de los resultados

### 5.11. Almacenamiento en Base de Datos
- Implementar almacenamiento de predicciones en SQLite
- Crear esquema adecuado para almacenar entradas y resultados
- Desarrollar funciones para consultar predicciones históricas

### 5.12. Seguimiento de Experimentos
- Configurar MLflow, W&B o herramienta similar
- Registrar experimentos, parámetros y métricas
- Comparar diferentes experimentos visualmente

### 5.13. Detección de Drift
- Implementar técnicas de detección de drift de datos
- Simular un escenario de drift y mostrar la detección
- Documentar estrategias para manejar el drift

### 5.14. Control de Versiones para Datasets
- Configurar DVC para control de versiones de datos
- Documentar diferentes versiones de datos procesados
- Mostrar la reproducibilidad con diferentes versiones

### 5.15. Datasets Sintéticos
- Crear versiones anónimas/sintéticas del dataset
- Evaluar la calidad de los datos sintéticos
- Comparar rendimiento del modelo con datos reales vs. sintéticos

### 5.16. Visualización Interactiva
- Implementar visualizaciones interactivas con Altair
- Crear gráficos exploratorios interactivos
- Visualizar resultados del modelo de forma interactiva

### 5.17. Pruebas Unitarias
- Desarrollar pruebas unitarias con pytest
- Cubrir funcionalidades críticas del procesamiento de datos y modelado
- Implementar integración continua si es posible

### 5.18. DataFrames Tipados
- Implementar validación de esquemas con Pandera
- Definir esquemas para datos crudos y procesados
- Integrar validación a lo largo del flujo de procesamiento

### 5.19. Calibración del Modelo
- Implementar técnicas de calibración para intervalos de predicción
- Evaluar la calidad de calibración
- Visualizar resultados antes y después de la calibración

## 6. Jupyter Notebook Principal

### 6.1. Estructura del Notebook
Crear un notebook (`notebooks/main_notebook.ipynb`) con la siguiente estructura:
- Tabla de contenidos generada automáticamente
- Secciones claramente definidas con encabezados markdown
- Formato consistente y profesional
- Código mínimo que importe funcionalidades de los módulos

### 6.2. Contenido del Notebook
El notebook debe incluir:
- Introducción al problema y objetivos
- Breve descripción de los datos
- Resumen del proceso de exploración de datos (con visualizaciones)
- Explicación del preprocesamiento aplicado
- Descripción de las características creadas
- Implementación y evaluación del modelo base
- Implementación y evaluación del modelo principal
- Comparación entre modelo base y modelo principal
- Resultados con intervalos de confianza
- Visualización de resultados clave
- Implementación y resultados de características opcionales
- Conclusiones y recomendaciones

## 7. Documentación

### 7.1. README Completo
Crear un archivo README.md detallado con:
- Descripción clara del proyecto y objetivos
- Estructura del repositorio y explicación de cada componente
- Instrucciones completas para configurar el entorno
- Guía paso a paso para ejecutar el código
- Explicación de las características implementadas
- Resumen de los resultados principales
- Descripción de decisiones técnicas importantes
- Referencias y enlaces relevantes

### 7.2. Docstrings
- Incluir docstrings detallados en todas las funciones y clases
- Seguir convenciones de NumPy o Google para docstrings
- Documentar parámetros, tipos, valores de retorno y excepciones

### 7.3. Comentarios en el Código
- Añadir comentarios explicativos para lógica compleja
- Mantener un estilo consistente en los comentarios
- Evitar comentarios obvios o redundantes

## 8. Buenas Prácticas de Desarrollo

### 8.1. Estilo de Código
- Seguir PEP 8 para el estilo de código Python
- Mantener consistencia en la estructura y formato
- Usar nombres descriptivos para variables, funciones y clases

### 8.2. Control de Versiones
- Realizar commits frecuentes y atómicos
- Escribir mensajes de commit descriptivos y consistentes
- Mantener un historial de commits limpio que refleje el progreso

### 8.3. Gestión de Errores
- Implementar manejo adecuado de excepciones
- Validar entradas en funciones críticas
- Proporcionar mensajes de error informativos

## 9. Entrega Final

### 9.1. Verificación de Requisitos
Comprobar que se han implementado todos los requisitos obligatorios:
- [X] Carga y preprocesamiento de datos
- [X] Transformaciones de características
- [X] Modelo predictivo con scikit-learn o similar
- [X] Evaluación con métricas apropiadas
- [X] Comparación con modelo base
- [X] Reporte de métricas con intervalos
- [X] Código estructurado en módulos
- [X] Jupyter Notebook final con formato adecuado
- [X] Al menos una característica opcional

### 9.2. Prueba de Funcionalidad
- Verificar que todo el código se ejecuta sin errores
- Comprobar que el flujo completo funciona correctamente
- Asegurar que el Jupyter Notebook se ejecuta de principio a fin

### 9.3. Entrega
- Realizar commit final con todos los cambios
- Asegurar que el README está completo y actualizado
- Compartir el URL del repositorio público de GitHub