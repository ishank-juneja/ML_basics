# CS 419 Assignment 3

In this assignment, you are expected to learn 3 types of Machine Learning models. Models to
classify Text dataset, Image dataset and Speech dataset. The folder we share with you consists
all the required files you need to complete the assignment.<br>
● The model.py file contains the skeleton classes and functions that you should implement
to construct the architecture of the models in different tasks. After implementing this
script, you should import this file to your training scripts and construct your model with
appropriate parameters, activation functions etc.<br>
Look at model.py and pay attention to the comments there.
● The data/census/ folder contains the text dataset as csv values. It is a binary
classification dataset with 14 features. The features are a mixture of categorical and
numerical types. You will have to perform appropriate feature engineering(like one-hot
encoding) on the attributes and feed the data to your model. The description of dataset
can be found at /data/census/description<br>
Look at train_census.py to gather more insights from the comments.<br>
● Store your predictions on the test dataset in /predictions/predictions_census.txt<br>
● The train_mnist.py should contain the code to predict the mnist dataset output. Read the
train, test and validation files from data/mnist/ and store your predictions in
predictions/predictions_mnist.txt for the test files.<br>
● The train_speech.py contains the code to read the files given in data/speech/ into your
code. This dataset contains Speech files for the words “Go” and “Stop” from different
speakers. You are required to build a model which learns to classify the given input
audio file to “Go” or “Stop” respectively. Again, store the predictions accordingly.<br>
● ALL THE PREDICTION FILES SHOULD HAVE THE FORMAT:<br>
○ PREDICTION OF INSTANCE1<br>
○ PREDICTION OF INSTANCE2<br>
○ …<br>
○ PREDICTION OF INSTANCE<br>
(Every prediction is to be given in a single line in the output file) Strictly follow the
format suggested or else penalty will be imposed.<br>
● Your final submission should be a .tar.gz bundle of a directory organized exactly as
described here.<br>
● Your report must contain a detailed description of the approach you followed for
implementing model.py. If you added new functions or classes to the skeleton file we
shared, add a detailed description about them in the report.<br>
● Tune the hyperparameters of the model you created and add your observations in the
report mentioned which combination of the hyperparameters gives you the best
prediction.

Some Guidelines :

1. This assignment is designed with the aim to :
a. Let you witness neural nets in action on a bunch of different tasks
b. Get you to implement a simple yet powerful architecture by requiring you to
understand a few core tensorflow functionalities & concepts
2. Because there is no recipe for this, the assignment is a bit open-ended. This means that
there is no fixed set of tasks that you have to do. Although there is a framework of tasks
that you need to complete, but they can be done in different ways.
a. As a consequence of this, you can try (and are indeed highly encouraged) to
explore the effects of different kinds of parameters
i. Number of layers
ii. Number of nodes in each layer
iii. Different activation functions (sigmoid, tanh, relu, etc.)
iv. Different regularisers
v. Learning rate, epochs, Batch Size, etc
b. And you should report the effects of these on the model in terms of
i. Accuracy and Loss
ii. Training Time/Steps
iii. Any other relevant metrics
3. We suggest that you follow the order
a. Mnist : get a basic architecture working in model.py
i. This will involve figuring out tensorflow functionality, and getting used to
computational graph model (and removing bugs in model.py)
b. Speech : work with hyperparameters
i. Once you’ve figured out model.py, then work with the hyperparameters
and training - plot accuracy and/or loss, etc. Try different sizes of nets
(different architectures if you want) and get the best performance you can
c. Census : work on data inputting also
4. Your end goal should be
a. Learning about how basic tensorflow models are built, and what different terms
represent
b. experimenting and reporting the effects of different ways to train a net, and try to
get the best performance you can (Remember there are some marks for your
performance on the test set in all the three tasks)
Do not hesitate to post on moodle / contact the TAs to ask for clarifications.
