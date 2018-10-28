import numpy as np

# Obtain the log function for a given tx . w
def sigmoid(t):
    logfn = np.divide(1,1+np.exp(-t))    
    return logfn

# Calculate the loss function using negative log likelihood	
def calculate_loss_logistic(y, tx, w):
    y = np.reshape(y,(len(y),)) # Make sure the sizes are (n,) and not (n,1)
	
	# IMPORTANT: The loss function has been change to account for very large values of tx,w 
	# This happens during Newton's method when the weights can tend to be quite high i.e. 1e04 ish
    vec_tx_w = np.dot(tx,w)
	
	# Separate the tx dot w vector into components that are < 100 and those that are much > 100
    vec_tx_w_normal = vec_tx_w[vec_tx_w < 100]
    vec_tx_w_large = vec_tx_w[vec_tx_w >= 100] # For very large values, log (1 + e^x) ~= log(e^x) = x
    loss_2 = np.sum(np.log(1+np.exp(vec_tx_w_normal))) + np.sum(vec_tx_w_large) 
    loss = -np.dot(np.transpose(y),np.dot(tx,w)) + loss_2
    
    return loss

# Calculage the gradient for logistic regression
def calculate_gradient_logistic(y, tx, w):
    
    # Reshape all vectors (just in case)
    y = np.reshape(y,(len(y),))
    w = np.reshape(w,(len(w),))
  
	# Get the sigmoid function and reshape it just in case
    logfn = sigmoid(np.dot(tx,w))
    logfn = np.reshape(logfn,(len(logfn),))
    
    gradient = np.dot(np.transpose(tx), logfn-y)
    return gradient

# Peform one iteration of the gradient descent for logistic regression
def learning_by_gradient_descent(y, tx, w, gamma, lambda_):
  
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

# Return the Hessian for logistic regression using Newton's method
def calculate_hessian(y, tx, w):
    
    # Get the sigmoid function for each sample
    logfn = sigmoid(np.dot(tx,w))
    
    # Get the diagonal vector containing (sigma_n * (1-sigma_n)) for each n
    S_vector = np.transpose(np.array([np.multiply(logfn,1-logfn)]))
    
    # Compute the Hessian using X'SX to get a DxD matrix
    hessian = np.dot(np.transpose(tx), S_vector*tx)	
    return hessian
   
# Perform one iteration of Newton's method
def learning_by_newton_method(y, tx, w,gamma, lambda_):
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

# Logistic Regression without regularization
def logistic_regression(y,tx,init_w,max_iters,gamma):
    # First convert the data to a 0-1 scale
    y[np.where(y == -1)] = 0
    lambda_ = 0
	# init parameters
    threshold = 1e-7
    losses = []
    isNewton = 0
    
    # Initialize guess weights
    w = init_w

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

# Logistic Regression using Newton's method 
def logistic_regression_newton(y,tx,init_w,max_iters,gamma):
    # First convert the data to a 0-1 scale
    y[np.where(y == -1)] = 0
    lambda_ = 0
	# init parameters
    threshold = 1e-7
    losses = []
        
    # Initialize guess weights
    w = init_w

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

# Logistic Regression with regularization	
def reg_logistic_regression(y,tx,lambda_,init_w,max_iters,gamma):
    # First convert the data to a 0-1 scale
    y[np.where(y == -1)] = 0
	# init parameters
    threshold = 1e-7
    losses = []
    
    # Initialize guess weights
    w = np.zeros((tx.shape[1], ))

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

# Logistic Regression with regularization using Newton's method (Final Submission)	
def reg_logistic_regression_newton(y,tx,lambda_,init_w,max_iters,gamma):
    # First convert the data to a 0-1 scale
    y[np.where(y == -1)] = 0
	# init parameters
    threshold = 1e-7
    losses = []
    
    # Initialize guess weights
    w = init_w

    # Start the logistic regression
    for iter in range(max_iters):
        # Get the updated w using the Newton's method
        loss, w = learning_by_newton_method(y, tx, w,gamma, lambda_)
        losses.append(loss)        
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-1]) < threshold:
            break
        
    w_star = w 
    return w_star, loss
