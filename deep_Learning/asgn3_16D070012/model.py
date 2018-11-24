#model.py is a helper class that makes it easy to create a Neural Net in TensorFlow

import tensorflow as tf
import numpy as np
import math
from matplotlib import pyplot as plt

class myNeuralNet:
	def __init__(self, dim_input_data, dim_output_data):  		
		#Member variable for dimension of Input Data Vectors (Number of features...)
		self.dim_input_data = dim_input_data
		#Variable for dimension of output labels
		self.dim_output_data = dim_output_data
		#TF Placeholder for input data   
		self.x =  tf.placeholder(tf.float32, [None, dim_input_data])
		#TF Placeholder for labels 
		self.y_ = tf.placeholder(tf.float32, [None, dim_output_data])
		#Initilaise Container to store all the layers of the network 
		#Initilaise it with the placeholder as the first layer
		self.layer_list = [self.x]
		
	def addHiddenLayer(self, layer_dim, activation_fn = None, regularizer_fn = None):
		#Input to current layer is output of previous layer
		layer_input = self.layer_list[-1]
		#Add this layer to flow graph by making connection to input
		layer = tf.layers.dense(layer_input, layer_dim, activation = activation_fn, kernel_regularizer = regularizer_fn)
		#Append to internal list
		self.layer_list.append(layer)

	def addFinalLayer(self, activation_fn = None, regularizer_fn = None): 
		#Dimensionality of final layer is stored in self.dim_output_data
		#Input to current layer is output of previous layer
		layer_input = self.layer_list[-1]
		# The output of the final layer are logits -- Un-normalized log probability values
		layer = tf.layers.dense(layer_input, self.dim_output_data, activation = activation_fn, kernel_regularizer = regularizer_fn)
		#Append final layer to internal list
		self.layer_list.append(layer)		

	def eval(self):
		#Function to predict labels once model has been trained
		#self.logits contains unnormalized linear scale logits
		self.logits = self.layer_list[-1]
		#Aplying a squashing softmax gives probabilities that sum to 1
		self.probabilities = tf.nn.softmax(self.logits)
		#The predicted matrix in this case will be
		self.predictions = tf.one_hot(tf.argmax(self.probabilities, dimension = 1), depth = self.dim_output_data)
		#The index to be returned in case of the MNIST task 
		self.predicted_labels = tf.argmax(self.probabilities, 1)

	def setup_training(self, learn_rate):
		# Define and store loss (cost function) as self.loss
		# Change this to operate on the current batch
		# Anything that updates over time in tensorflow is a variable
		self.loss =  tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.y_, logits = self.logits)
		# Define the train step as self.train_step = ..., using an optimizer from tf.train and call minimize(self.loss)
		self.train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)

	def setup_metrics(self):
		# Function to compute accuracy on Validation data set
		# Use the predicted labels and compare them with the input labels(placeholder defined in __init__)
		# to calculate accuracy, and store it as self.accuracy
		correct_prediction = tf.equal(self.predicted_labels, tf.argmax(self.y_, 1))
    	# Calculate accuracy
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		
		#For speech 
		#correct_prediction = tf.equal(self.predicted_sp, self.y_)
		#self.accuracy_sp = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		#Accuracy accepts logits (probabilities sum = 1) as predicted labels and 
		#internally compares with one hot encoded labels self.y_
		#Also the function returns 2 parameters and there are some reasons 
		#For acc_update to be run first using a sperate TF session 
		#self.accuracy, self.acc_update = tf.metrics.accuracy(self.y_, self.predictions)
		#When using this we need to do Initialise TF local variables 
		#used to setup metrics etc sess.run(tf.local_variables_initializer())

	def train(self, sess, train_data, train_labels, max_epochs, batch_size, train_size, valid_data, valid_labels, valid_size, test_data, print_step = 100): 
		# sess is the tensorflow session passed from main, used to run the computation graph
		# All the functions uptil now were just constructing the computation graph
		
		#Drops the last few rows of the training data set to make it an
        #Integeral multiple of Batch Size 
		train_data = train_data[:(train_size//batch_size)*batch_size]
		train_labels = train_labels[:(train_size//batch_size)*batch_size]
		#Split the data into batches
		train_data_batches = np.split(train_data, train_size//batch_size)
        #Similar split for the output labels 
		train_labels_batches = np.split(train_labels, train_size//batch_size)
		#Read in the validation data set once and for all
		self.valid_data = valid_data
		self.valid_labels = valid_labels
		#Initialise List to store validation loss
		L_val_loss = []
		#List to hold training loss
		L_train_loss = [] 

		# one 'epoch' represents that the network has seen the entire dataset once
		# We look at the data not all at once but in Batches
		steps_per_epoch = int(train_size/batch_size)
		max_steps = max_epochs * steps_per_epoch
		for step in range(max_steps):
			# read the current batch of data from the training data batch collection
			#Modulo since loop back whenever 1 epoch (pass over dataset) is complete
			self.cur_batch = train_data_batches[step%steps_per_epoch]
			self.cur_lables = train_labels_batches[step%steps_per_epoch]
			
			# Now run train_step and self.loss on this batch of training data
			# All the inputs to the Graph must be provided using dictioinary
			_, train_loss = sess.run([self.train_step, self.loss], feed_dict={self.x: self.cur_batch, self.y_: self.cur_lables})

			#Check on validation Data set every 100 or so iterations
			if (step % print_step) == 0:
				# Report Accuracy and loss on validation data set by running	
				val_acc, val_loss= sess.run([self.accuracy, self.loss], feed_dict={self.x: self.valid_data, self.y_: self.valid_labels})
				# Above computation results in numpy arrays, PRINT and store above values
				net_train_loss = np.sum(train_loss)/batch_size
				net_val_loss = np.sum(val_loss)/valid_size
				print("Metrics after SGD Training Step: {}".format(step)) 
				print("Training Loss is: {:.3f}".format(net_train_loss))
				print("Validation Loss is: {:.3f}".format(net_val_loss))
				print("Validation Accuracy is: {:.2f}% \n".format(100*val_acc))
				L_val_loss.append(net_val_loss)
				#Update train_loss less often for simplicity
				L_train_loss.append(net_train_loss) 

		# -- for loop ends --

		# Code to plot Training loss
		train_plot = plt.figure() 	
		plt.plot(L_train_loss)
		plt.xlabel('Print Steps')
		plt.ylabel('Training Loss')
		
		#Plot Validation loss
		val_plot = plt.figure() 	
		plt.plot(L_val_loss)
		plt.xlabel('Print Steps')
		plt.ylabel('Validation Loss')
		plt.show()
		
		# Run predictions on the test set
		self.test_data = test_data
		test_predictions = sess.run(self.predicted_labels, feed_dict={self.x: self.test_data})
		return test_predictions
'''
	#Output is 1 Dimentional STOP/GO
	def train_speech(self, sess, train_signal, train_lbls, max_epochs, batch_size, train_size, valid_signal, valid_lbls, valid_size, test_signal, print_step = 100): 
		# sess is the tensorflow session passed from main, used to run the computation graph
		# All the functions uptil now were just constructing the computation graph
		
		#Drops the last few rows of the training data set to make it an
        #Integeral multiple of Batch Size 
		train_data = train_signal[:(train_size//batch_size)*batch_size]
		train_labels = train_lbls[:(train_size//batch_size)*batch_size]
		#Split the data into batches
		train_data_batches = np.split(train_data, train_size//batch_size)
        #Similar split for the output labels 
		train_labels_batches = np.split(train_labels, train_size//batch_size)
		#Read in the validation data set once and for all
		self.valid_data = valid_signal
		self.valid_labels = valid_lbls
		#Initialise List to store validation loss
		L_val_loss = []
		#List to hold training loss
		L_train_loss = [] 

		# one 'epoch' represents that the network has seen the entire dataset once
		# We look at the data not all at once but in Batches
		steps_per_epoch = int(train_size/batch_size)
		max_steps = max_epochs * steps_per_epoch
		for step in range(max_steps):
			#Read the current batch of data from the training data batch collection
			#Modulo since loop back whenever 1 epoch (pass over dataset) is complete
			self.cur_batch = train_data_batches[step%steps_per_epoch]
			self.cur_lables = train_labels_batches[step%steps_per_epoch]
			# Now run train_step and self.loss on this batch of training data
			# All the inputs to the Graph must be provided using dictioinary
			_, train_loss, logits = sess.run([self.train_step_sp, self.loss_sp, self.logits], feed_dict={self.x: self.cur_batch, self.y_: self.cur_lables})
			print train_loss
			print logits

			#Check on validation Data set every 100 or so iterations
			if (step % print_step) == 0:
				# Report Accuracy and loss on validation data set by running	
				val_acc, val_loss, predictions = sess.run([self.accuracy_sp, self.loss_sp, self.predicted_sp], feed_dict={self.x: self.valid_data, self.y_: self.valid_labels})
				# Above computation results in numpy arrays, PRINT and store above values
				net_train_loss = np.sum(train_loss)/batch_size
				net_val_loss = np.sum(val_loss)/valid_size
				print("Metrics after SGD Training Step: {}".format(step)) 
				print("Training Loss is: {:.3f}".format(net_train_loss))
				print("Validation Loss is: {:.3f}".format(net_val_loss))
				print("Validation Accuracy is: {:.2f}% \n".format(100*val_acc))
				L_val_loss.append(net_val_loss)
				#Update train_loss less often for simplicity
				L_train_loss.append(net_train_loss)
				for i in range(valid_size):
					print predictions[i], valid_lbls[i]

		# -- for loop ends --

		# Code to plot Training loss
		train_plot = plt.figure() 	
		plt.plot(L_train_loss)
		plt.xlabel('Print Steps')
		plt.ylabel('Training Loss')
		
		#Plot Validation loss
		val_plot = plt.figure() 	
		plt.plot(L_val_loss)
		plt.xlabel('Print Steps')
		plt.ylabel('Validation Loss')
		plt.show()
		
		# Run predictions on the test set
		self.test_data = test_signal
		test_predictions = sess.run(self.predicted_sp, feed_dict={self.x: self.test_data})
		return test_predictions

'''