"""
Function to split the data into training, cv and test sets
"""
from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.2, cv_size=0.2, random_state=None):
    """
    Split the dataset into training, cross-validation, and test sets.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing the dataset.
    - test_size (float): Proportion of the dataset to include in the test split.
    - cv_size (float): Proportion of the dataset to include in the cross-validation split.
    - random_state (int or None): Random seed for reproducibility.

    Returns:
    - tuple: (train_data, cv_data, test_data)
    """
    # Split into training and temporary set
    train_data, temp_data = train_test_split(data, test_size=(test_size + cv_size), random_state=random_state)

    # Split the temporary set into CV and test sets
    cv_data, test_data = train_test_split(temp_data, test_size=cv_size/(test_size + cv_size), random_state=random_state)

    return train_data, cv_data, test_data
