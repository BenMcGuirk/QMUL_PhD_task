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

    # Calculate mean and standard deviation for each column
    mean_values = data.mean()
    std_values = data.std()

    # Z-score normalize each column
    normalized_data = (data - mean_values) / std_values

    return normalized_data