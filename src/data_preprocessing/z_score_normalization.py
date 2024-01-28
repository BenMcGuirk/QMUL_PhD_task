"""
Algorithm for feature scaling
"""
# Import necessary libraries
import pandas as pd

def normalize_data(data):
    """
    Perform Z-score normalization on a pandas DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the dataset.

    Returns:
    - pd.DataFrame: DataFrame with features Z-score normalized.
    """

    # Ensure the input is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Separate target variable (y) and features (X)
    y_column = 'case'  # Target variable
    X = data.drop(y_column, axis=1)
    y = data[y_column]

    # Calculate mean and standard deviation for each column
    mean_values = X.mean()
    std_values = X.std()

    # Z-score normalize each column
    normalized_X = (X - mean_values) / std_values

    # Add target variable back to the DataFrame
    normalized_data = pd.concat([y, normalized_X], axis=1)

    return normalized_data