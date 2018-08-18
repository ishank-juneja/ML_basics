#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
#Use anynode to avoid names
from anytree import AnyNode, RenderTree

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

#gets the optimal split threshold for a 
#particular attribute an returns the 
#corresponding error measure 
def optimal_split_MSE( train_data, attr ):
	min_MSE  = np.inf
	thresh = train_data[attr].min()
	split = 0
	df_bottom = train_data

	#scope for improvement
	#instead of boolean indexing
	#do all operations on sorted array
	while(True):
		mask = train_data[attr] > thresh
		df_bottom = train_data[mask]
		
		#Check continue condition
		if(df_bottom.empty) :
			break

		df_top = train_data[~mask]
		cur_MSE = df_top.iloc[:,-1].std() + df_bottom.iloc[:,-1].std() 
		if(cur_MSE < min_MSE):
			min_MSE = cur_MSE
			split = thresh
		thresh = df_bottom[attr].min()
	return (split, min_MSE)

def optimal_split_MAE( train_data , attr ):
	min_MAE  = np.inf
	thresh = train_data[attr].min()
	split = 0
	df_bottom = train_data

	#scope for improvement
	#instead of boolean indexing
	#do all operations on sorted array
	while(True):
		mask = train_data[attr] > thresh
		df_bottom = train_data[mask]
		
		#Check continue condition
		if(df_bottom.empty) :
			break

		df_top = train_data[~mask]
		cur_MAE = df_top.iloc[:,-1].mad() + df_bottom.iloc[:,-1].mad() 
		if(cur_MAE < min_MAE):
			min_MAE = cur_MAE
			split = thresh
		thresh = df_bottom[attr].min()
	return (split, min_MAE)

#Scope for improvement : Avoid passing mode again and again
def make_tree( df , mode ):

	# Reached Leaf Node
	if (df.shape[0] <= min_size) :
		#get last column for class and take mean to assign label
		this_node = AnyNode(parent = None, Leaf = True, Label = np.mean(df.iloc[:,-1]), \
							 att = None, sp = None, size = df.shape[0])
		return this_node

	# Start with min possible value
	#Update as we go along
	min_err = np.inf
	cur_err = 0
	optimal_att = ''
	split_pt = 0

	for column in df.iloc[:, :-1] :
		#Sort values as per this column
		#Easy to divide on the basis of 
		#label once this is done 
		#df.sort_values(column, inplace = True)
		#df.reset_index(drop = True, inplace = True)
		print ('Processing column ', column)
		if(mode == 0) :
			(split, cur_err) = optimal_split_MSE(df, column)
		else :
			(split, cur_err) = optimal_split_MAE(df, column)
			print(split, cur_err)

		if (cur_err < min_err) :
			min_err = cur_err
			optimal_att = column
			split_pt = split

	#Parent is none for now
	#On returning the child is linked to the parent
	print ('Chosen values are')
	print ('Attribute name ', optimal_att)
	print ('Split Point', split_pt)

	this_node = AnyNode(parent = None, Leaf = False , Label = None, \
						att = optimal_att, sp = split_pt, size = df.shape[0])	
	
	mask = df[optimal_att] > split_pt
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

if (split_method == '--mean_squared' or split_method == '--absolute') : 
	if (split_method == '--mean_squared') :
		mode = 0
	else :
		mode = 1
else :
	sys.exit('Error : Correct usage is when the last parameter is either --absolute or \
							mean_squared--')


#Initialise Tree  
#Extend it using Recurssion
#Decision tree model is stored in root
root = make_tree(train_data, mode)
print (RenderTree(root))
# if (split_method = '--mean_squared') : 
	
# 	# Split as per MSE

# else :
# 	# Split as per MAE 

