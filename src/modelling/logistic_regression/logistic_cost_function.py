"""
Cost function for logistic regression model
"""
import numpy as np

def logistic_cost_function(y, h_theta_x):
    m = len(y)
    cost = -(1/m) * np.sum(y * np.log(h_theta_x) + (1 - y) * np.log(1 - h_theta_x))
    return cost