import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    # ***************************************************
    # print('Shape of t is', np.shape(t))
    # logfn = np.divide(np.exp(t),1+np.exp(t))
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
    # print('tx,w is : ', np.dot(tx,w))
    # print('norm of the original tx,w is : ', np.linalg.norm(np.dot(tx,w)))	
    # Get the sigmoid function and reshape it just in case
    logfn = sigmoid(np.dot(tx,w))
    # print('Sigmoid function in calculate_gradient is:', logfn)
    logfn = np.reshape(logfn,(len(logfn),))
    # print('Sigmoid function after reshape in calculate_gradient is:', logfn)
	# print('shape of logfn is : ', np.shape(logfn))
    # print('norm of logfn w is : ', np.linalg.norm(logfn))	
    # Obtain the gradient : delta =  X'(sigma - y)
    gradient = np.dot(np.transpose(tx), logfn-y)
    return gradient
    
def learning_by_gradient_descent(y, tx, w, gamma):
    # ************ Peforms one iteration of the gradient descent ****************
    
    # Get the loss function
    loss = calculate_loss(y,tx,w) 
    
    # Get the gradient of the loss function at the current point
    gradient = calculate_gradient(y,tx,w) 
    # print('shape of the original w is : ', np.shape(w))
    # print('norm of the original w is : ', np.linalg.norm(w))	
    # print('the shape of the gradient is: ', np.shape(gradient))
    # print('norm of the gradient is : ', np.linalg.norm(gradient))	
    # Update the weight vector using a step size gamma and the direction provided by the gradient
    w = w - gamma*gradient 
    # print('gamma*gradient is : ', gamma*gradient)
    # print('norm of the final w is : ', np.linalg.norm(w))	
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
    print('Gradient before regularization is', np.linalg.norm(gradient))
    if lambda_ > 0:
        loss += lambda_ * np.squeeze(w.T.dot(w))
        gradient += 2 * lambda_ * w
        hessian += lambda_*2 	
    print('Gradient after regularization is', np.linalg.norm(gradient))
    print('The loss is', loss)
    # print('size of the gradient is', np.shape(gradient))
    # print('size of the hessian is', np.shape(hessian))
    """ Note that instead of computing the inverse of the hessian and then multiplying by the gradient,
    I solve a linear system -> lambda = inv(H)*gradient => H*lambda = gradient.
    Hence I can calculate lambda and use it for the update : w = w - lambda """
    
    lam_ = np.linalg.solve(hessian,gradient)
    # print('size of the lambda_ is', np.shape(lambda_))
    # w = w - gamma*lambda_     
    w = w - gamma*lam_     
    return loss, w

def logistic_regression(y,tx,isNewton,stepSize,max_iter, lambda_ = 0):
    # First convert the data to a 0-1 scale
    y[np.where(y == -1)] = 0
	# init parameters
    # max_iter = 500
    threshold = 1e-6
#     lambda_ = 0.1
    losses = []
    gamma = stepSize

    # Initialize guess weights
    w = np.zeros((tx.shape[1], ))

    # Start the logistic regression
    for iter in range(max_iter):
        print('ITERATION NUMBER:',iter)
        if(isNewton): # Get the updated w using the Newton's method
            loss, w = learning_by_newton_method(y, tx, w,gamma, lambda_)
        else: # Otherwise, use the Gradient Descent method 
            loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        print('The loss is:', loss)
        losses.append(loss)        
        if len(losses) > 1 and (np.abs(losses[-1] - losses[-2]))/np.abs(losses[-1]) < threshold:
            break
        
    w_star = w 
    return w_star