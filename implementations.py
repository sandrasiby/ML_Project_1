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

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    return -tx.T.dot(err) / len(err)

def compute_loss(y, tx, w):
    """Calculate the loss using mse"""
    e = y - tx.dot(w)
    return calculate_mse(e)

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y,tx,w)
        # gradient w by descent update
        w = w - gamma * grad

    return loss, w


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx,batch_size, 1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
    return loss, w
