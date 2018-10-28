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


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

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

    return w, loss

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
    return w, loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
       <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]