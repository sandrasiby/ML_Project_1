import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    logfn = np.divide(1,1+np.exp(-t))    
    return logfn

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    y = np.reshape(y,(len(y),)) # Make sure the sizes are (n,) and not (n,1)
	
	# IMPORTANT: The loss function has been change to account for very large values of tx,w 
    vec_tx_w = np.dot(tx,w)
	# Separate the tx dot w vector into components that are < 100 and those that are much > 100
    vec_tx_w_normal = vec_tx_w[vec_tx_w < 100]
    vec_tx_w_large = vec_tx_w[vec_tx_w >= 100] # For very large values, log (1 + e^x) ~= log(e^x) = x
    loss_2 = np.sum(np.log(1+np.exp(vec_tx_w_normal))) + np.sum(vec_tx_w_large) 
    loss = -np.dot(np.transpose(y),np.dot(tx,w)) + loss_2
    # loss = -np.dot(np.transpose(y),np.dot(tx,w)) + np.sum(np.log(1+np.exp(np.dot(tx,w))))
    return loss
    
def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    # Reshape all vectors
    y = np.reshape(y,(len(y),))
    w = np.reshape(w,(len(w),))
  
	# Get the sigmoid function and reshape it just in case
    logfn = sigmoid(np.dot(tx,w))
    logfn = np.reshape(logfn,(len(logfn),))
  
    gradient = np.dot(np.transpose(tx), logfn-y)
    return gradient
    
def learning_by_gradient_descent(y, tx, w, gamma, lambda_):
    # ************ Peforms one iteration of the gradient descent ****************
    
    # Get the loss function
    loss = calculate_loss(y,tx,w) 
    
    # Get the gradient of the loss function at the current point
    gradient = calculate_gradient(y,tx,w)
    if lambda_ > -1:
        loss += lambda_ * w.T.dot(w)
        gradient += 2 * lambda_ * w

    # Update the weight vector using a step size gamma and the direction provided by the gradient
    w = w - gamma*gradient 
    return loss, w

def calculate_hessian(y, tx, w):
    """.Return the Hessian of the given function """
    # Get the sigmoid function for each sample
    logfn = sigmoid(np.dot(tx,w))
    
    # Get the diagonal vector containing (sigma_n * (1-sigma_n)) for each n
    # S = np.diag(np.multiply(logfn,1-logfn))
    S_vector = np.transpose(np.array([np.multiply(logfn,1-logfn)]))
    
    # Compute the Hessian using X'SX to get a DxD matrix
    # hessian = np.dot(np.transpose(tx), np.dot(S,tx))
    hessian = np.dot(np.transpose(tx), S_vector*tx)	
    return hessian
    
def learning_by_newton_method(y, tx, w,gamma, lambda_):
    """ Perform one iteration of Newton's method """
    
    # Get the loss, gradient and hessian at the current point w
    loss = calculate_loss(y,tx,w)
    gradient = calculate_gradient(y,tx,w)
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

# def logistic_regression_SGD(y, tx, stepsize, max_iters):
    # """Stochastic gradient descent algorithm."""
    # # ***************************************************
    # # INSERT YOUR CODE HERE
    # # TODO: implement stochastic gradient descent.
    # # ***************************************************
    # # Initialize guess weights
    # w = np.zeros((tx.shape[1], ))
    # gamma = stepsize
    # threshold = 1e-7

    # losses = []
    # batch_size = 100
    # for n_iter in range(max_iters):
        # # y_n, tx_n = batch_iter(y, tx, batch_size)
        # for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # loss = calculate_loss(y, tx, w)
            # gradient = calculate_gradient(minibatch_y, minibatch_tx, w)
            # w = w - gamma * gradient
            # print('The iteration is ', n_iter, ' The norm of the gradient is =', np.linalg.norm(gradient), ' and the loss is = ', loss  )
            # # print('The loss is', loss)
            # # store w and loss
            # losses.append(loss)		
            # if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-1]) < threshold: break
              
    # w_star = w
    # return w_star
	
def logistic_regression(y,tx,isNewton,stepSize,max_iter, lambda_ = 0):
    # First convert the data to a 0-1 scale
    y[np.where(y == -1)] = 0
	# init parameters
    threshold = 1e-7
    losses = []
    gamma = stepSize

    # Initialize guess weights
    w = np.zeros((tx.shape[1], ))

    # Start the logistic regression
    for iter in range(max_iter):
        # print('ITERATION NUMBER:',iter)
        if(isNewton): # Get the updated w using the Newton's method
            loss, w = learning_by_newton_method(y, tx, w,gamma, lambda_)
        else: # Otherwise, use the Gradient Descent method 
            loss, w = learning_by_gradient_descent(y, tx, w, gamma, lambda_)
        # print('Loss = ', loss)
        losses.append(loss)        
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-1]) < threshold:
            break
        
    w_star = w 
    return w_star
	
# def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    # """
    # Generate a minibatch iterator for a dataset.
    # Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    # Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    # Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    # Example of use :
    # for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        # <DO-SOMETHING>
    # """
    # data_size = len(y)

    # if shuffle:
        # shuffle_indices = np.random.permutation(np.arange(data_size))
        # shuffled_y = y[shuffle_indices]
        # shuffled_tx = tx[shuffle_indices]
    # else:
        # shuffled_y = y
        # shuffled_tx = tx
    # for batch_num in range(num_batches):
        # start_index = batch_num * batch_size
        # end_index = min((batch_num + 1) * batch_size, data_size)
        # if start_index != end_index:
            # yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]