# Salary Prediction Model

## Overview
This project develops a machine learning model to predict employee salaries based on job-related features. It includes data loading, preprocessing, feature engineering, model training, and evaluation, structured as a technical challenge for a Data Scientist role.

## Project Structure
```
salary_prediction/
├── data/
│   └── salary_data.csv         # Dataset
├── notebooks/
│   └── salary_prediction_notebook.ipynb # Jupyter notebook for analysis and presentation
├── src/
│   ├── data_loading.py         # Script for data loading
│   ├── data_preprocessing.py   # Script for data cleaning and preprocessing
│   ├── model_training.py       # Script for model training
│   └── model_evaluation.py     # Script for model evaluation
└── README.md                   # Project overview and setup instructions
```

## Getting Started
1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd salary_prediction
   ```
2. **Explore the Jupyter Notebook**
   - Open `notebooks/salary_prediction_notebook.ipynb` to walk through the data analysis, model development, and evaluation steps.

## Dependencies
- pandas
- scikit-learn
- numpy
- scipy

## Usage
Run the Jupyter notebook `salary_prediction_notebook.ipynb` to execute the end-to-end salary prediction workflow. The notebook is structured to be self-contained and imports necessary functions from the `src` directory.

## Model Development
- **Data Loading**: `src/data_loading.py` handles loading the dataset from `data/salary_data.csv`.
- **Data Preprocessing**: `src/data_preprocessing.py` includes functions for cleaning data, handling missing values, and splitting data into training and testing sets.
- **Feature Engineering**:  `src/data_preprocessing.py` engineers relevant features from the existing dataset to improve model accuracy. 
- **Model Training**: `src/model_training.py` trains a `RandomForestRegressor` model and a `DummyRegressor` for baseline comparison.
- **Model Evaluation**: `src/model_evaluation.py` evaluates model performance using metrics such as RMSE and R^2, including confidence intervals for robust evaluation.

## Results
The Jupyter notebook `salary_prediction_notebook.ipynb` presents the detailed results of model evaluation, including performance metrics and comparisons between the trained models.

## License
[Specify License, e.g., MIT License]