#!/usr/bin/env python

"""main.py: A program to implement a regresion tree on a muti-attributed data-set
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


__author__      = "Ishank Juneja"
