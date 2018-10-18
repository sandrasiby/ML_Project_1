import numpy as np

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    a = tx.T.dot(tx) + ( 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1]))
    b = tx.T.dot(y)
    w_star = np.linalg.solve(a, b)
    return w_star

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w_star = np.linalg.solve(a, b)
    return w_star

def compute_mse(y, tx, w):
    error = y - np.dot(tx, w)
    mse = np.dot(np.transpose(error), error)/(2 * len(y))
    return mse