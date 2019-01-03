**CS419M (Autumn 2018): Assignment 1**

The file main.py should create a decision tree model which should be used to
infer the output on test data. The file main .py/main.cpp should write the output (on test
data) to output.csv as mentioned on Kaggle submission page.

**Programming**<br>
Implementing a Regression Tree<br>
In this assignment, you will be implementing a regression tree from scratch. Each decision node
will correspond to one of the features in train.csv, which is selected by choosing the feature that
minimises loss. For this assignment, you will be experimenting with two loss functions viz. mean
squared loss and absolute loss. The file main.py/main.cpp should accept command line
arguments: --train_file, --test_file, --min_leaf_size. Note that the min_leaf_size parameter will
constrain the number of examples in the leaf node. For example, if min_leaf_size=20, leaf
nodes can not have less than 20 examples. Your code should also accept another two
command line arguments mean_squared and absolute as (example for python):
python main.py --train_data train.csv --test_data test.csv --min_leaf_size 20 --absolute
The features in train.csv and test.csv are either continuous-valued or discrete-valued. 

Details about data attributes are given on Kaggle competition page.
The task for this problem is hosted on Kaggle. Please go to Competition 1 and Competition 2,
after you have created an account on Kaggle using your roll number and GPO ID. You can
download this small toy dataset to save time on computation during initial experimentation.

A. Use either Python or C++ code to implement your Regression Tree. Report the best loss
values to report.pdf. <br>
B. Build a complete regression tree using your training samples in train.csv . Prune the
decision tree as discussed in the class. You can use 1-fold cross validation. Plot the
graph between loss and number of nodes in the regression tree. Note that you will have
two graphs, one for absolute loss and one for squared loss. Report this graph to
report.pdf <br>
C. Predict the output on second dataset given here (Competition 2) and modify the model to
optimize for this bigger dataset. Report the best loss value to report.pdf <br>
D. Training time and inference time should also be mentioned in report. Report should also
contain a brief note on implementation and any extra experiments or modifications you
wish to emphasize. <br>

Additional Clarifications:
1. You can tune the min_leaf_size hyper-parameter to improve performance in competition.
Mention the parameter value in report.pdf .
