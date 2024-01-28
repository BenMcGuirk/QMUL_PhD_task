"""
Gradient descent for logistic regression model
"""
import numpy as np
from src.modelling.logistic_regression.logistic_hypothesis_function import sigmoid

def logistic_gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    for _ in range(num_iterations):
        h_theta_x = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (h_theta_x - y)) / m
        theta -= alpha * gradient
    return theta
