#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
#Use anynode to avoid names
from anytree import AnyNode

"""	CART algorithm with regression on output class label
	main.py: A program to implement a regresion tree on a muti-attributed data-set
	accepts the command line parameters 
	1. --trainfile: The training data in .csv format 
	2. --test_file: The validation data in .csv format
	3. --min_leaf_size: Puts a lower limit on the number of training
		examples that can be put into a single leaf node
	4. mean_squared: Presence indicates use of Mean squared loss function
	5. absolute: Use Absolute error as measure of loss	
	Example usage
	python main.py --train_data train.csv --test_data test.csv --min_leaf_size 20 --absolute
	
	The features in train.csv and test.csv are either continuous-valued or discrete-valued
	
	The result of the validation Data from the model are saved to the file output.csv
"""

__author__ = "Ishank Juneja"

#Computes mean absolute error
#For a set of real valued data 
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

#gets the optimal split threshold for a 
#particular attribute an returns the 
#corresponding error measure 
def optimal_split_MSE( train_data ):
	min_MSE  = np.inf
	split = 0
	for ii in (1, train_data.att_name.shape[0]-2) :
		df_top =  train_data[0:ii]
		df_bottom = train_data[ii:]
		cur_MSE = df_top.iloc[:,-1].std() + df_bottom.iloc[:,-1].std() 
		if(cur_MSE < min_MSE):
			min_MSE = cur_MSE
			split = i
return (split, error) 

def optimal_split_MAE( train_data ):

return (split, error)

#Scope for improvement : Avoid passing mode again and again
def make_tree( df , mode ):

	# Reached Leaf Node
	if (df.shape[0] <= min_size) :
		#get last column for class and take mean to assign label
		this_node = AnyNode(parent = None, Leaf = True, Label = np.mean(df.iloc[:,-1]), \
															 att = None, sp = None)
		return this_node

	# Start with min possible value
	#Update as we go along
	min_error = np.inf
	cur_error = 0
	optimal_att = ''
	split_at = 0

	for column in df :
		#Sort values as per this column
		#Easy to divide on the basis of 
		#label once this is done 
		df.sort_values(column)
		df.reset_index(drop = True, inplace = True)
		if(mode == 0) :
			(split, cur_error) = optimal_split_MSE(df)
		else :
			(split, cur_error) = optimal_split_MAE(df)

		if (cur_error < min_error) :
			min_error = cur_error
			optimal_att = column
			split_pt = split

		#Parent is none for now
		#On returning the child is linked to the parent
		this_node = AnyNode(parent = None, Leaf = False , Label = None, \
									att = optimal_att, sp = split_pt)	
		
		mask = df[optimal_att] > split_at
		df_right = df[mask]
		df_left = df[~mask]

		right_child = make_tree(df_right, mode)
		left_child = make_tree(df_left, mode)
		
		right_child.parent =  this_node
		left_child.parent =  this_node

return this_node

#Main Program

#Command line parameters
if (len(sys.argv) != 8) :
	sys.exit('Error : Correct usage is python main.py --train_data \
		trainFile.csv --test_data testName.csv --min_leaf_size <int> --<absolute/mean_squared>')

# train.csv is sys.argv[2]
#Train data is stored as a pandas dataframe
train_data = pd.read_csv(sys.argv[2])

#Minmimum number of training examples that 
#Are allowed in a ;eaf node  
min_size = 2*int(sys.argv[6]) + 1

#Either Abs or MSE
split_method = sys.argv[7]

if (split_method != '--mean_squared' or split_method != '--absolute') : 
	sys.exit('Error : Correct usage is when the last parameter is either --absolute or \
							mean_squared--')
elif (split_method = '--mean_squared') :
	mode = 0
else :
	mode = 1

#Initialise Tree  
#Extend it using Recurssion
#Decision tree model is stored in root
root = make_tree(train_data, mode)

# if (split_method = '--mean_squared') : 
	
# 	# Split as per MSE

# else :
# 	# Split as per MAE 

