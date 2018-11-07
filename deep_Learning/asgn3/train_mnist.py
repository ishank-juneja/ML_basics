from __future__ import print_function

import tensorflow as tf
import math
import numpy as np

from model import myNeuralNet

# x denotes features, y denotes labels
xtrain = np.load('data/mnist/xtrain.npy')
ytrain = np.load('data/mnist/ytrain.npy')

xval = np.load('data/mnist/xval.npy')
yval = np.load('data/mnist/yval.npy')

xtest = np.load('data/mnist/xtest.npy')

dim_input = 784
dim_output = 10

max_epochs = 10
learn_rate = 1e-4
batch_size = 50

#The ytrain and yval lengths will match automatically 
train_size = len(xtrain)
valid_size = len(xval)
test_size = len(xtest)

total_images = []
total_labels = []

# Create Computation Graph using Class defined in model.py
nn_instance = myNeuralNet(dim_input, dim_output)
#Add hidden layers  
nn_instance.addHiddenLayer(1000, activation_fn=tf.nn.relu)
# add more hidden layers here by calling addHiddenLayer as much as you want
# a net of depth 3 should be sufficient for most tasks
nn_instance.addFinalLayer()
nn_instance.setup_training(learn_rate)
nn_instance.setup_metrics()

# Training steps
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	test_pred = nn_instance.train(sess, max_epochs, train_size, batch_size) # fill in other arguments as you modify the train(self, sess, ...) in model.py
	# you will have to pass xtrain, ytrain, etc ... also as arguments so that you can sample batches in train() of model.py


# write code here to store test_pred in relevant file
out_file_path = "predictions/predictions_mnist.txt"
out_file = open(out_file_path, w)
fp.write(test_pred) 	