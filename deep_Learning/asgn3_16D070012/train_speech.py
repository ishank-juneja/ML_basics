from __future__ import print_function

import tensorflow as tf
import math
import numpy as np
#For reading speech signals
import librosa 

from model import myNeuralNet

# Defining properties
#Stream of bits  
dim_input = 900
#Either the word STOP or the word GO
dim_output = 2

# Dataset paths
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
		label = [0, 1]
		# count_go += 1
	else:
		label = [1, 0]
		# count_stop += 1        
	return fileName, label

''' util function for speech sampling'''
def sample(fpath):
	signal, _ = librosa.load(fpath)
	#signal now contains Mel-frequency cepstral coeffcients
	#A footprint on whoch we can train our model
	signal = librosa.feature.mfcc(y=signal, n_mfcc=int(dim_input/45))
	#There may be some change in columns of signal.shape 
	#Depending on the duration of signal 
	#rows fixed at dim_input/45  
	#print("after mfcc:", signal.shape)
	#Make signal into column vector for training
	signal = signal.flatten()
	#Makes all training data uniform in size by 
	#using circular interpolation
	signal = np.resize(signal, new_shape=(dim_input,1))
	#Return as a row vector for pythonic convenience
	return np.transpose(signal)

# Import data
# Populate with training data
train_input = [] # list of strings - each entry is a filepath
train_labels = [] # list of labels - each entry is a float

with open(train_fname) as f:
	append_str = 'data/speech/train/'
	for line in f:
		fileName, label = interpretLine(line, append_str)
		train_input.append(fileName)
		train_labels.append(label)

# Populate with validation data
valid_input = []
valid_labels = []

with open(val_fname) as f:
	append_str = 'data/speech/validation/'
	for line in f:
		fileName, label = interpretLine(line, append_str)
		valid_input.append(fileName)
		valid_labels.append(label)

# Populate with test data
test_input = []
# File format is different for test dataset
# The summary file which is being parsed for 
# File names does not have any training labels
with open(test_fname) as f:
	append_str = 'data/speech/test/'
	for line in f:
		fileName = append_str + line.strip()
		test_input.append(fileName)

train_size = len(train_input)
valid_size = len(valid_input)
test_size = len(test_input)

''' Create arrays for training, validation, test '''

#train_size = 200
# Read and store the MFCC (Spectral Decomposition) for training set
#Each column of this Matrix is a speech clip
train_signal = np.empty(shape=(train_size, dim_input))
#Each entry in this list/array is a label (dim_output = 1)
train_lbls = np.empty(shape=(train_size, dim_output))
# Converting raw speech .wav files into parsable data
# Uses librosa spectral decoposition functionality is a slow step
for index_train in range(train_size):
	train_signal[index_train] = sample(train_input[index_train])
	train_lbls[index_train] = np.transpose(np.reshape(np.array(train_labels[index_train]), newshape=(dim_output,1) ) )
	#Use print statements to keep track of progress
	if index_train%100 == 0:
		print("Read ", index_train, " instances out of full train set.")
print("Read full training set.")
# print(count_go)
# print(count_stop)
        
#print(train_signal.shape)
#print(train_lbls.shape)

#print(train_signal[0])
#print(train_lbls[0])

#valid_size = 200
# Read and store the MFCC for validation set, identical to above
valid_signal = np.empty(shape=(valid_size, dim_input))
valid_lbls = np.empty(shape=(valid_size, dim_output))
for index_valid in range(valid_size):
	valid_signal[index_valid] = sample(valid_input[index_valid])
	valid_lbls[index_valid] = np.transpose( np.reshape(np.array(valid_labels[index_valid]), newshape=(dim_output,1) ) )
	if index_valid%100 == 0:
		print("Read ", index_valid, " instances out of full validation set.")
print("Read full validation set.")

#test_size = 200
# Read and store the MFCC for test set (only signals here, no labels)
test_signal = np.empty(shape=(test_size, dim_input))
for index_test in range(test_size):
	test_signal[index_test] = sample(test_input[index_test])
	if index_test%100 == 0:
		print("Read ", index_test, " instances out of full test set.")
print("Read full test set.")

# Inputting part done ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 

max_epochs = 40
learn_rate = 1e-6
batch_size = 32

# Create Computation Graph
nn_instance = myNeuralNet(dim_input, dim_output)
# Add hidden layers to flow graph
nn_instance.addHiddenLayer(1000, activation_fn=tf.nn.relu)
nn_instance.addHiddenLayer(1000, activation_fn=tf.nn.relu)
nn_instance.addHiddenLayer(1000, activation_fn=tf.nn.relu)

#Add the output layer of the NN
nn_instance.addFinalLayer()
#Make graph connections for output metrics
nn_instance.eval()
#Training flow graph
nn_instance.setup_training(learn_rate)
#Error and Accuracy
nn_instance.setup_metrics()

# Training steps
print(train_signal.shape)
print(train_lbls.shape)
with tf.Session() as sess:
	#Init variables
	sess.run(tf.global_variables_initializer())
	#Use currently running session perform training
	test_pred = nn_instance.train(sess, train_signal, train_lbls, max_epochs, batch_size,
								 train_size, valid_signal, valid_lbls, valid_size, test_signal) 

#Write the returned results to file 
out_file_path = "predictions/predictions_speech.txt"

#Commented out to prevent edits
#with open(out_file_path, 'w') as f:
#	for item in test_pred:
#		if(item == 1):
#			f.write("go\n")
#		else:
#			f.write("stop\n")

