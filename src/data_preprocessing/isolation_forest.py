"""
Algorithm for outlier detection
"""
# Import necessary libraries
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import pandas as pd

def remove_outliers(data, contamination=0.05, random_state=None):
    """
    Remove outliers from a dataset using the Isolation Forest algorithm.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the dataset.
    - contamination (float): The proportion of outliers in the dataset. Default is 0.05 (5%).
    - random_state (int or None): Random seed for reproducibility. Default is None.

    Returns:
    - pd.DataFrame: DataFrame with outliers removed (inliers).
    - pd.DataFrame: DataFrame with the removed outliers.
    """
    # Replace missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Create an Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)

    # Fit the model and predict outliers
    outlier_preds = iso_forest.fit_predict(data_imputed)

    # Identify outliers and inliers
    outliers = data[outlier_preds == -1]
    inliers = data[outlier_preds == 1]

    # Return both the DataFrame with outliers removed (inliers) and the removed outliers
    return inliers, outliers