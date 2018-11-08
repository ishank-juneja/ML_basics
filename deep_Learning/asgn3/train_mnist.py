from __future__ import print_function

import tensorflow as tf
import math
import numpy as np

#Import TF Helper Class
from model import myNeuralNet

# x denotes features, y denotes labels

#Load training data
xtrain = np.load('data/mnist/xtrain.npy')
ytrain = np.load('data/mnist/ytrain.npy')

#Load validation data
xval = np.load('data/mnist/xval.npy')
yval = np.load('data/mnist/yval.npy')

#Load test data
xtest = np.load('data/mnist/xtest.npy')

#Image dimensions, 28 x 28 rolled out images 
dim_input = 784
dim_output = 10

#Training parameters used while running TF session
max_epochs = 10
learn_rate = 1e-4
batch_size = 50

#The ytrain and yval lengths will match automatically 
train_size = len(xtrain)
valid_size = len(xval)
test_size = len(xtest)

#Why are these used..?
total_images = []
total_labels = []

# Create Computation Graph using Class defined in model.py
nn_instance = myNeuralNet(dim_input, dim_output)
#Add hidden layers  
nn_instance.addHiddenLayer(1000, activation_fn = tf.nn.relu)
# add more hidden layers here by calling addHiddenLayer as much as you want
# a net of depth 3 (3 hidden units) should be sufficient for most tasks

#Add the output layer of the NN
nn_instance.addFinalLayer()
#Setup the computational flow graph 
#of training using the myNN class 
nn_instance.setup_training(learn_rate)	
#Setup the computational flow graph 
#For error computation
nn_instance.setup_metrics()

# Training steps	
with tf.Session() as sess:
	#Initialize all TF variables used as part of the flow graph
	sess.run(tf.global_variables_initializer())
	#Using the currently running session, perform training
	test_pred = nn_instance.train(sess, xtrain, ytrain, max_epochs, train_size, batch_size) 
	# fill in other arguments as you modify the train(self, sess, ...) in model.py
	# you will have to pass xtrain, ytrain, etc ... also as arguments so that you can sample batches in train() of model.py


# write code here to store test_pred in relevant file
out_file_path = "predictions/predictions_mnist.txt"
out_file = open(out_file_path, w)
fp.write(test_pred) 	