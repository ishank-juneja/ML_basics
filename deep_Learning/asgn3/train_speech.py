from __future__ import print_function

import tensorflow as tf
import math
import random
import numpy as np

import librosa # for reading speech signals
# you'll have to install this like tensorflow

from model import myNeuralNet

# Defining properties
dim_input = 900
dim_output = 1

# strings for file reading
train_fname = 'data/speech/train/flist.txt'
val_fname = 'data/speech/validation/flist.txt'
test_fname = 'data/speech/test/flist.txt'

''' util function for reading'''
# count_go = 0
# count_stop = 0
def interpretLine(line, append_str):
	global count_go, count_stop
	cleanLine = line.strip()
	fileName = append_str + cleanLine
	if 'go_' in cleanLine:
		label = 1.0
		# count_go += 1
	else:
		label = 0.0
		# count_stop += 1        
	return fileName, label

''' util function for speech sampling'''
def sample(fpath):
	signal,_ = librosa.load(fpath)
	signal = librosa.feature.mfcc(y=signal, n_mfcc=int(dim_input/45))
	# print("after mfcc:", signal.shape)
	signal = signal.flatten()
	signal = np.resize(signal, new_shape=(dim_input,1))
	return np.transpose(signal)

# Import data
# storing train data
train_input = [] # list of strings - each entry is a filepath
train_labels = [] # list of labels - each entry is a float
with open(train_fname) as f:
	append_str = 'data/tens_speech/train/'
	for line in f:
		fileName, label = interpretLine(line, append_str)
		train_input.append(fileName)
		train_labels.append(label)

# storing validation data - similar
valid_input = []
valid_labels = []
with open(val_fname) as f:
	append_str = 'data/tens_speech/validation/'
	for line in f:
		fileName, label = interpretLine(line, append_str)
		valid_input.append(fileName)
		valid_labels.append(label)

# storing test data
test_input = []
# remember you don't have test labels
with open(test_fname) as f:
	append_str = 'data/tens_speech/test/'
	for line in f:
		fileName = append_str + line.strip()
		test_input.append(fileName)

train_size = len(train_input)
valid_size = len(valid_input)
test_size = len(test_input)

''' Create arrays for training, validation, test '''

# read and store mfcc for training set
train_signal = np.empty(shape=(train_size, dim_input))
train_lbls = np.empty(shape=(train_size, dim_output))
# this will take a lot of time
for index_train in range(train_size):
	train_signal[index_train] = sample(train_input[index_train])
	train_lbls[index_train] = np.transpose( np.reshape(np.array(train_labels[index_train]), newshape=(dim_output,1) ) )
	if index_train%100 == 0:
		print("Read ", index_train, " instances out of full train set.")
print("Read full training set.")
# print(count_go)
# print(count_stop)
        
# print(train_signal.shape)
# print(train_lbls.shape)

# print(train_signal[0])
# print(train_lbls[0])

# read and store mfcc for validation set
valid_signal = np.empty(shape=(valid_size, dim_input))
valid_lbls = np.empty(shape=(valid_size, dim_output))
for index_valid in range(valid_size):
	valid_signal[index_valid] = sample(valid_input[index_valid])
	valid_lbls[index_valid] = np.transpose( np.reshape(np.array(valid_labels[index_valid]), newshape=(dim_output,1) ) )
	if index_valid%100 == 0:
		print("Read ", index_valid, " instances out of full validation set.")
print("Read full validation set.")

# read and store mfcc for test set (only signals here, no labels)
test_signal = np.empty(shape=(test_size, dim_input))
for index_test in range(test_size):
	test_signal[index_test] = sample(test_input[index_test])
	if index_test%100 == 0:
		print("Read ", index_test, " instances out of full test set.")
print("Read full test set.")

# Inputting part done ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

max_epochs = 40
learn_rate = 1e-4
batch_size = 32

# Create Computation Graph
nn_instance = myNeuralNet(dim_input, dim_output)
nn_instance.addHiddenLayer(500, activation_fn=tf.nn.relu)
# add more hidden layers here by calling addHiddenLayer as much as you want
# a net of depth 3 should be sufficient for most tasks
nn_instance.addFinalLayer()
nn_instance.setup_training(learn_rate)
nn_instance.setup_metrics()

# Training steps
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	test_pred = nn_instance.train(sess) # add more arguments here

# write code here to store test_pred in relevant file