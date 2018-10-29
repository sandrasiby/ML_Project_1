import numpy as np
# *********************** LINEAR REGRESSION FUNCTIONS ************************************** #
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

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
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


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic Gradient descent algorithm."""
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
    
# *********************** LOGISTIC REGRESSION FUNCTIONS ************************************** #

def sigmoid(t):
    ''' Obtain the log function for a given tx . w '''
    logfn = np.divide(1,1+np.exp(-t))    
    return logfn

	
def calculate_loss_logistic(y, tx, w):
    '''Calculate the loss function using negative log likelihood '''
    y = np.reshape(y,(len(y),)) # Make sure the sizes are (n,) and not (n,1)
	
    ''' IMPORTANT: The loss function has been change to account for very large values of tx,w 
    This happens during Newton's method when the weights can tend to be quite high i.e. 1e04 ish '''
    vec_tx_w = np.dot(tx,w)
	
	# Separate the tx dot w vector into components that are < 100 and those that are much > 100
    vec_tx_w_normal = vec_tx_w[vec_tx_w < 100]
    vec_tx_w_large = vec_tx_w[vec_tx_w >= 100] # For very large values, log (1 + e^x) ~= log(e^x) = x
    loss_2 = np.sum(np.log(1+np.exp(vec_tx_w_normal))) + np.sum(vec_tx_w_large) 
    loss = -np.dot(np.transpose(y),np.dot(tx,w)) + loss_2
    
    return loss


def calculate_gradient_logistic(y, tx, w):
    ''' Calculage the gradient for logistic regression'''
    # Reshape all vectors (just in case)
    y = np.reshape(y,(len(y),))
    w = np.reshape(w,(len(w),))
  
	# Get the sigmoid function and reshape it just in case
    logfn = sigmoid(np.dot(tx,w))
    logfn = np.reshape(logfn,(len(logfn),))
    
    gradient = np.dot(np.transpose(tx), logfn-y)
    return gradient


def learning_by_gradient_descent(y, tx, w, gamma, lambda_):
    '''Peform one iteration of the gradient descent for logistic regression'''
    # Get the loss function
    loss = calculate_loss_logistic(y,tx,w) 
    
    # Get the gradient of the loss function at the current point
    gradient = calculate_gradient_logistic(y,tx,w)
	
    # If regularization is used, lambda_ is non-zero
    loss += lambda_ * w.T.dot(w)
    gradient += 2 * lambda_ * w

    # Update the weight vector using a step size gamma and the direction provided by the gradient
    w = w - gamma*gradient 
    return loss, w


def calculate_hessian(y, tx, w):
    '''Return the Hessian for logistic regression using Newton's method	'''
    # Get the sigmoid function for each sample
    logfn = sigmoid(np.dot(tx,w))
    
    # Get the diagonal vector containing (sigma_n * (1-sigma_n)) for each n
    S_vector = np.transpose(np.array([np.multiply(logfn,1-logfn)]))
    
    # Compute the Hessian using X'SX to get a DxD matrix
    hessian = np.dot(np.transpose(tx), S_vector*tx)	
    return hessian
   

def learning_by_newton_method(y, tx, w,gamma, lambda_):
    '''Perform one iteration of Newton's method'''
	# Get the loss, gradient and hessian at the current point w
    loss = calculate_loss_logistic(y,tx,w)
    gradient = calculate_gradient_logistic(y,tx,w)
    hessian = calculate_hessian(y,tx,w)
    
    if lambda_ > -1:
        loss += lambda_ * w.T.dot(w)
        gradient += 2 * lambda_ * w
        hessian += np.identity(len(hessian))*lambda_*2
    
    """ Note that instead of computing the inverse of the hessian and then multiplying by the gradient,
    I solve a linear system -> lambda = inv(H)*gradient => H*lambda = gradient.
    Hence I can calculate lambda and use it for the update : w = w - lambda """
    
    lam_ = np.linalg.solve(hessian,gradient)
    
    w = w - gamma*lam_     
    return loss, w


def logistic_regression(y,tx,init_w,max_iters,gamma):
    '''Logistic Regression without regularization using GD'''
    # First convert the data to a 0-1 scale
    y[np.where(y == -1)] = 0
    lambda_ = 0
	# init parameters
    threshold = 1e-7
    losses = []
    isNewton = 0
    
    # Initialize guess weights
    w = np.reshape(init_w,(len(init_w),))

    # Start the logistic regression
    for iter in range(max_iters):
        # Use the Gradient Descent method  to get the update
        loss, w = learning_by_gradient_descent(y, tx, w, gamma, lambda_)
        print(loss)
        losses.append(loss)        
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-1]) < threshold:
            break
        
    w_star = w 
    return w_star, loss

 
def logistic_regression_newton(y,tx,init_w,max_iters,gamma):
    '''Logistic Regression using Newton's method without regularization'''
    # First convert the data to a 0-1 scale
    y[np.where(y == -1)] = 0
    lambda_ = 0
	# init parameters
    threshold = 1e-7
    losses = []
        
    # Initialize guess weights
    w = np.reshape(init_w,(len(init_w),))

    # Start the logistic regression
    for iter in range(max_iters):
        # Get the updated w using the Newton's method
        loss, w = learning_by_newton_method(y, tx, w,gamma, lambda_)
        # print(loss)
        losses.append(loss)        
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-1]) < threshold:
            break
        
    w_star = w 
    return w_star, loss

	
def reg_logistic_regression(y,tx,lambda_,init_w,max_iters,gamma):
    '''Logistic Regression with regularization using GD'''
    # First convert the data to a 0-1 scale
    y[np.where(y == -1)] = 0
	# init parameters
    threshold = 1e-7
    losses = []
    
    # Initialize guess weights
    w = np.reshape(init_w,(len(init_w),))

    # Start the logistic regression
    for iter in range(max_iters):
        # Get the updated w using Gradient Descent
        loss, w = learning_by_gradient_descent(y, tx, w, gamma, lambda_)
        print(loss)
        losses.append(loss)        
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-1]) < threshold:
            break
        
    w_star = w 
    return w_star, loss


def reg_logistic_regression_newton(y,tx,lambda_,init_w,max_iters,gamma):
    '''Logistic Regression with regularization using Newton's method (Final Submission)	'''
    # First convert the data to a 0-1 scale
    y[np.where(y == -1)] = 0
	# init parameters
    threshold = 1e-7
    losses = []
    
    # Initialize guess weights
    w = np.reshape(init_w,(len(init_w),))

    # Start the logistic regression
    for iter in range(max_iters):
        # Get the updated w using the Newton's method
        loss, w = learning_by_newton_method(y, tx, w,gamma, lambda_)
        losses.append(loss)        
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-1]) < threshold:
            break
        
    w_star = w 
    return w_star, loss
