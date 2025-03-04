# Salary Prediction Challenge

## Overview
This project implements a machine learning model to predict employee salaries based on various job-related features. The implementation includes comprehensive data preprocessing, feature engineering, model training, and evaluation components.

## Project Structure
```
salary-prediction-challenge/
├── data/
│   ├── raw/         # Original dataset
│   └── processed/    # Preprocessed data
├── notebooks/
│   └── main_notebook.ipynb  # Main analysis notebook
├── src/
│   ├── data/
│   │   ├── make_dataset.py  # Data loading utilities
│   │   └── preprocess.py    # Data cleaning and preprocessing
│   ├── features/
│   │   └── build_features.py # Feature engineering
│   ├── models/
│   │   ├── train_model.py    # Model training
│   │   ├── predict_model.py  # Model prediction
│   │   └── evaluate_model.py # Model evaluation
│   ├── tests/               # Unit tests
│   └── visualization/       # Data visualization
└── requirements.txt        # Project dependencies
```

## Features

### Data Preprocessing
- Normalizes column names for consistency
- Handles missing values and duplicates
- Converts data types appropriately
- Scales numerical features

### Feature Engineering
- Experience level extraction from job titles
- Company-related features (size, ownership type)
- Location-based features
- Salary-related features and categories
- Categorical variable encoding

### Model Development
- Implements data splitting with proper validation
- Includes baseline model (DummyRegressor)
- Uses Random Forest as the main model
- Comprehensive model evaluation metrics

## Getting Started

1. **Set up the environment**
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Data Preparation**
- Place your salary dataset in `data/raw/`
- The dataset should include fields like job title, company information, location, and salary details

3. **Run the Analysis**
- Open and run `notebooks/main_notebook.ipynb`
- The notebook provides a step-by-step walkthrough of the entire process

## Key Components

### Data Processing (`src/data/`)
- `make_dataset.py`: Handles data loading and initial validation
- `preprocess.py`: Implements data cleaning and preprocessing functions

### Feature Engineering (`src/features/`)
- `build_features.py`: Contains all feature engineering functions:
  - Experience level extraction
  - Company features processing
  - Location features
  - Salary categorization

### Model Training and Evaluation (`src/models/`)
- Training pipeline with cross-validation
- Model performance evaluation
- Prediction functionality

## Dependencies
- pandas
- numpy
- scikit-learn
- jupyter

## Development

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.