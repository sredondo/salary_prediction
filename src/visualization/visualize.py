import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_salary_distribution(df, salary_column='avg_salary'):
    """
    Plots the distribution of salaries.

    Parameters:
    df (pandas.DataFrame): DataFrame containing salary data.
    salary_column (str): Name of the salary column.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[salary_column], kde=True)
    plt.title('Distribution of Salaries')
    plt.xlabel('Average Salary')
    plt.ylabel('Frequency')
    plt.show()

def plot_experience_salary_relation(df, experience_col='age', salary_col='avg_salary'):
    """
    Plots the relationship between experience and salary.

    Parameters:
    df (pandas.DataFrame): DataFrame with experience and salary data.
    experience_col (str): Name of the experience column.
    salary_col (str): Name of the salary column.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=experience_col, y=salary_col, data=df)
    plt.title('Relationship between Experience and Salary')
    plt.xlabel('Age (Proxy for Experience)')
    plt.ylabel('Average Salary')
    plt.show()

def plot_salary_by_category(df, category_col, salary_col='avg_salary'):
    """
    Plots the distribution of salaries by a categorical variable.

    Parameters:
    df (pandas.DataFrame): DataFrame containing salary and categorical data.
    category_col (str): Name of the categorical column.
    salary_col (str): Name of the salary column.
    """
    plt.figure(figsize=(12, 7))
    sns.boxplot(x=category_col, y=salary_col, data=df)
    plt.title(f'Salary Distribution by {category_col}')
    plt.xlabel(category_col)
    plt.ylabel('Average Salary')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Correlation analysis
def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap for numeric columns in the DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing numeric data.
    """
    plt.figure(figsize=(12, 8))
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_columns].corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix))
    
    # Plot heatmap with improved readability
    sns.heatmap(correlation_matrix,
                mask=mask,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                square=True,
                linewidths=0.5)
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

# Add more visualization functions as needed, e.g., correlation heatmap, etc.

if __name__ == '__main__':
    # Example usage (assuming data is loaded)
    from src.data.make_dataset import load_data
    filepath = 'salary-prediction-challenge/data/raw/salary_data.csv'
    df = load_data(filepath)
    if df is not None:
        plot_salary_distribution(df)
        plot_experience_salary_relation(df)
        plot_salary_by_category(df, 'Sector') # Example category