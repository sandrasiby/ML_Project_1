import numpy as np

def compute_mse(y, tx, w):
    error = y - np.dot(tx, w)
    mse = np.dot(np.transpose(error), error)/(2 * len(y))
    return mse

def least_squares(y, tx):
    """calculate the least squares solution."""
    lhs = tx.T.dot(tx)
    rhs = tx.T.dot(y)
    w_star = np.linalg.solve(lhs, rhs)
    loss = compute_mse(y, tx, w_star)
    return w_star, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lhs = tx.T.dot(tx) + ( 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1]))
    rhs = tx.T.dot(y)
    w_star = np.linalg.solve(lhs, rhs)
    loss = compute_mse(y, tx, w_star)
    return w_star, loss