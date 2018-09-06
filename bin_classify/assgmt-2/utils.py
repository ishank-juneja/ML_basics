import numpy as np
 
####################Loss Functions########################
#All the loss functions take input as (target,output) 
#and return the loss
def square_hinge_loss(targets, outputs):
  	# z = yf(x), each element is yf(x) for that row
	batch_loss = 0
	z = np.multiply(targets, outputs)
	#Loss function
	def op(x):
		return np.square(1-x) if x <= 1 else 0
	#Vectorize it
	op = np.vectorize(op, otypes=[np.float32])
	z = op(z)
	batch_loss = z.sum()
	return batch_loss

def logistic_loss(targets, outputs):
	# z = yf(x), each element is yf(x) for that row
	batch_loss = 0
	z = np.multiply(targets, outputs)
	z = np.log(1 + np.exp(-1*z))
	batch_loss = z.sum()
	return batch_loss

def perceptron_loss(targets, outputs):
 	# z = yf(x), each element is yf(x) for that row
	batch_loss = 0
	z = np.multiply(targets, outputs)
	#Loss fucntion
	def op(x):
		return -x if x <= 0 else 0
	#Vectorize it
	op = np.vectorize(op, otypes=[np.float32])
	z = op(z)
	batch_loss = z.sum()
	return batch_loss

####################Regulariser Functions########################
#Return the regulariser value given the weights vector
#Don't regularize over the bias term in any of the functions

def L2_regulariser(weights):
	#All but the last bias term
	L2reg = weights[:-1]
	L2reg = np.square(L2reg)
	return L2reg.sum()

def L4_regulariser(weights):
	L4reg = weights[:-1]
	L4reg = np.square(L4reg)
	#Raise to power 4
	L4reg = np.square(L4reg)
	return L4reg.sum()

####################Gradient Computation#########################
#Return gradients as vectors of apt. size
def square_hinge_grad(weights, inputs, targets, outputs):
	d = np.size(weights)
	#Initialise output gradient
	grad = np.zeros(d)
	z = np.multiply(targets, outputs)
	
	def op(x):
		return (1-x) if x <= 1 else 0
	#Vectorize it
	op = np.vectorize(op, otypes=[np.float32])
	new = op(z)
	
	const_arr = -2*np.multiply(targets, new)

	for i in range(d):
		#Peform required element wise operations and add result into grad element
		grad[i] = np.multiply(inputs[:,i], const_arr).sum() 
	
	return grad

def logistic_grad(weights,inputs, targets, outputs):
	d = np.size(weights)
	#Initialise output gradient
	grad = np.zeros(d)
	z = np.multiply(targets, outputs)

	const_arr = -1*np.multiply(targets, 1 - 1/(1+np.exp(-1*z)))

	for i in range(d):
		#Peform required element wise operations and add result into grad element
		grad[i] = np.multiply(inputs[:,i], const_arr).sum() 
	
	return grad

def perceptron_grad(weights,inputs, targets, outputs):
	d = np.size(weights)
	#Initialise output gradient
	grad = np.zeros(d)
	z = np.multiply(targets, outputs)
	
	def op(x):
		return 1 if x <= 0 else 0
	#Vectorize it
	op = np.vectorize(op, otypes=[np.float32])
	new = op(z)
	
	const_arr = -1*np.multiply(targets, new)

	for i in range(d):
		#Peform required element wise operations and add result into grad element
		grad[i] = np.multiply(inputs[:,i], const_arr).sum() 
	
	return grad

#Return gradients (In this case 1D derivative) of regularizer functions
def L2_grad(weights):
	L2grad = 2*weights
	#No gradient for bias term
	L2grad[-1] = 0	
	return L2grad

def L4_grad(weights):
	#Cube each element and multiply by 4
    L2grad = 4*np.power(weights, 3)
    L2grad[-1] = 0
    return L2grad

#A look up table for the program 'main.py' to interact with 
#this library. Ensures that correct combination of loss functions 
#and gradients are used 

loss_functions = {"square_hinge_loss" : square_hinge_loss, 
                  "logistic_loss" : logistic_loss,
                  "perceptron_loss" : perceptron_loss}

loss_grad_functions = {"square_hinge_loss" : square_hinge_grad, 
                       "logistic_loss" : logistic_grad,
                       "perceptron_loss" : perceptron_grad}

regularizer_functions = {"L2": L2_regulariser,
                         "L4": L4_regulariser}

regularizer_grad_functions = {"L2" : L2_grad,
                              "L4" : L4_grad}
