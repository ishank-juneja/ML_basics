#Computes mean absolute error
#For a set of real valued data 
print __name__

def abs_error( data ):

	data = data - np.mean(data)
	data = np.absolute(data)
	mae = np.sum(data)/np.size(data)

	return mae

#Computes mean sqaure error
#For a set of real valued data 
def MSE( data ):
	
	data = data - np.mean(data)
	data = np.square(data)
	mse = np.sum(data)/np.size(data) 

	return mae