from __future__ import print_function

import numpy as np
import tensorflow as tf
import numpy as np
import math

from model import myNeuralNet

'''
For this task, the inputting code is not fully given to you.
The reason for this is that there are two important data cleaning tasks that you need to handle
1. Handling categorical values
	Hint: Look up one-hot encoding on the internet
2. Handling missing attributes
	Hint: You might want to replace such missing values with a reasonable statistical measure of that attribute

For more information, look at /data/census/description
'''

dim_input = -1 # change this according to your encoding of the input features
dim_output = 1 # binary class classification can be done using sigmoid

max_epochs = 50
learn_rate = 1e-4
batch_size = 50

def file_size(fname):
	with open(fname) as f:
		for i, l in enumerate(f):
			pass
	return i + 1

train_fname = 'data/census/train_data'
valid_fname = 'data/census/validation_data'
test_fname = 'data/census/test_data'
train_size = file_size(train_fname)
valid_size = file_size(valid_fname)
test_size = file_size(test_fname)

print(train_size)
print(valid_size)
print(test_size)
# Import data
'''
Note:
In this framework, we'll read the files as it is and perform data "cleaning" once we've read the batches.
You might want to go for the other route(like in train_mnist.py) - where the entire data is read in at once
and then you can perform data manipulations on it.

Also note that the last column is -1 in the test dataset.
This is to make sure that the same code works for all the three inputting framework,
but the input labels in test case are garbage (since they are all -1). 
'''

def get_clean_batches(batch_dict): # this is used because batchers (defined later) give us batch as python dictionary
	batch_list = list(batch_dict.items())
	no_instances = batch_list[0][1].shape[0] # will be batch_size in most cases ...
	# print(no_instances) # but because train_size is not exactly divisible by batch_size, no_instances might be different in a few cases
	inp_batch = np.empty(shape=(dim_input,no_instances))
	inp_label = np.empty(shape=(1,no_instances))
	rang = len(batch_list)
	for index in range(rang):
		elem_to_append = batch_list[index][1]
		if index == rang-1: # last element in a row is the label
			inp_label[0] = elem_to_append
		else: # all other elements are features
			inp_batch[index] = elem_to_append
	inp_batch = np.transpose(inp_batch)
	inp_answer = np.transpose(inp_answer)
	# perform further cleaning here before returning
	return inp_batch, inp_labels

# define tensorflow objects to get input
dataset_train = tf.contrib.data.make_csv_dataset(train_fname, batch_size)
iterator_train = dataset_train.make_initializable_iterator()
train_batcher = iterator_train.get_next()
# print(dataset_train)
# print(iterator_train)

dataset_valid = tf.contrib.data.make_csv_dataset(valid_fname, valid_size) # all validation instances in a single tensor
iterator_valid = dataset_valid.make_initializable_iterator()
valid_batcher = iterator_valid.get_next()

dataset_test = tf.contrib.data.make_csv_dataset(test_fname, test_size) # all test instances in a single tensor
iterator_test = dataset_test.make_initializable_iterator()
test_batcher = iterator_test.get_next()

# Create Computation Graph
nn_instance = myNeuralNet(dim_input, dim_output)
nn_instance.addHiddenLayer(200, activation_fn=tf.nn.relu)
# add more hidden layers here by calling addHiddenLayer as much as you want
# a net of depth 3 should be sufficient for most tasks
nn_instance.addFinalLayer()
nn_instance.setup_training(learn_rate)
nn_instance.setup_metrics()

# Instantiate Session
with tf.Session() as sess:
	sess.run([iterator_train.initializer, iterator_valid.initializer, iterator_test.initializer])
	sess.run(tf.global_variables_initializer())

	test_pred = nn_instance.train(sess) # add more arguments here
	# you will have to pass the train_batcher, valid_batcher, test_batcher to this for it to batch

	# In this framework, you will have to perform cleaning of data, and for such things you might want to
	# make use of get_clean_batches ...
	# For that, you'll have to import that in model.py

# write code here to store test_pred in relevant file