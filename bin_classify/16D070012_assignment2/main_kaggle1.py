import argparse
#argparse docs linked below
#https://docs.python.org/3/library/argparse.html

from scipy.optimize import minimize


import utils
import numpy as np

class DataLoader(object):
    # this class has a standard iterator declared
    # __len__ returns the number of batches (size of the object)
    # __get_item__ handles integer based indexing of the object 
    
    #Constructor
    def __init__(self, data_file, batch_size):
        #Read training data
        with open(data_file, 'r') as df:
            data = df.readlines()

        #Drop the first row which is labels in CSV file    
        data = data[1:]
        # // is floor division
        #Drops the last few rows of the training data set to make it an
        #Integeral multiple of Batch Size 
        data = data[:(len(data)//batch_size)*batch_size]
        #Randomly shuffle the rows in input data
        np.random.shuffle(data)
        #Split at , points (since CSV file)
        data = np.array([[float(col) for col in row.split(',')] for row in data])
        #Seperate out data columns and Output (Y) column 
        input_data, targets = data[:, 1:-1], data[:, -1]
        #Append ones onto input data so that dimention matches with feature vector with bias
        input_data = np.hstack([input_data, np.ones((len(input_data), 1), dtype=np.float32)])

        #Initialise class data members
        self.num_features = input_data.shape[1]#Columns
        self.current_batch_index = 0#Not used 
        #Input data and output columns separated into batch_size sized batches
        self.input_batches = np.split(input_data, len(input_data)//batch_size)
        #Similar split for the output (binary labels) 
        self.target_batches = np.split(targets, len(targets)//batch_size)

    def __len__(self):
        return len(self.input_batches)

    def __getitem__(self,i):

        #Get a particular row
        batch_input_data = self.input_batches[i]
        batch_targets = self.target_batches[i]
        return batch_input_data, batch_targets

def std_label(x):
    return -1 if (x <= 0) else 1

#Don't know why this syntax was used
def classify(inputs, weights):
    #this functions returns w^Tx . The output  is batch_size*1
    #May involve redundant steps, check later
	return np.dot(inputs, np.reshape(weights, (np.size(weights), 1)).reshape((-1,)))

# this function calculates the loss for a current batch
def get_objective_function(trainx,trainy,loss_type, regularizer_type, loss_weight):
    #Refer the loss_functions dictionary to get correct function
    loss_function = utils.loss_functions[loss_type]
    #Regularizer is an optional parameter
    if regularizer_type != None:
        #Refer the regularizer functions dic.
        regularizer_function = utils.regularizer_functions[regularizer_type]
           
    #Define a function within that invokes the actual loss function 
    #To compute the batch loss, later return a reference to this function 
   
    #Only parameter is the current weights
    def objective_function(weights):
        loss = 0
        #Batch inputs and outputs
        inputs, targets = trainx,trainy
        #Get predicted outputs by Vector dot product
        outputs = classify(inputs, weights)
        #Increase loss by C*loss 
        loss += loss_weight*loss_function(targets, outputs)
        if regularizer_type != None:
            #If Regularizer enabled, conmpute it and add it to objective            
            loss += regularizer_function(weights)
        return loss
    #Define a function in this scope and then return its reference
    return objective_function

#Structure is the same as abov function
def get_gradient_function(trainx,trainy,loss_type, regularizer_type, loss_weight):
    
    loss_grad_function = utils.loss_grad_functions[loss_type]
    if regularizer_type != None:
        regularizer_grad_function = utils.regularizer_grad_functions[regularizer_type]
        
    def gradient_function(weights):
        #Initialise gradient vector
        gradient = np.zeros(len(weights), dtype=np.float32)
        X=trainx
        Y=trainy
        outputs = classify(X,weights)
        gradient = loss_weight*loss_grad_function(weights,X,Y,outputs)/len(trainx)
        if regularizer_type != None:
            gradient += regularizer_grad_function(weights)
        return gradient
    return gradient_function

def train(data_loader, loss_type, regularizer_type, loss_weight):#Loss weight is the C value
    #Start with a random value, size of vector W is d+1 to account for bias term
    initial_model_parameters = np.random.random((data_loader.num_features))

    num_epochs=1000
    #Number of times entire data will be processed
    for i in range(num_epochs):
        loss=0
        if(i==0):
            #Only use the random weight vector for the 1st iteration
            start_parameters=initial_model_parameters
        #Inner loop run once per batch
        for j in range(len(data_loader)):#Total number of batches
            trainx, trainy=data_loader[j]#Calls __getitem()__ , get the jth batch
            
            #Code to make 0 labels -1, added by me
            op = np.vectorize(std_label, otypes=[np.float32])
            trainy = op(trainy) 

            #Return pointers to the required functions with values already passed
            objective_function = get_objective_function(trainx,trainy,loss_type, 
                                                regularizer_type,loss_weight)
            gradient_function = get_gradient_function(trainx,trainy, loss_type, 
                                              regularizer_type, loss_weight)
            
            # Most imp. line in the entire program
            # Refer docs, details too cumbersome  
            # Minimize returns an optimization result object with Array stored in att. x
            trained_model_parameters = minimize(objective_function, 
                                        start_parameters, 
                                        method="CG", 
                                        jac=gradient_function,
                                        options={'disp': False,
                                                 'maxiter': 1})
            #Use the correct objective function to update loss values
            loss+=objective_function(trained_model_parameters.x)
            #Update start parameters to inproved value for next iteration
            start_parameters=trained_model_parameters.x
        # prints the batch loss
        print("loss is  ",loss)
        
    print("Optimizer information:")
    print(trained_model_parameters)
    return trained_model_parameters.x
            

def test(inputs, weights):
    outputs = classify(inputs, weights)
    probs = 1/(1+np.exp(-outputs))
    # this is done to get all terms in 0 or 1 You can change for -1 and 1
    return np.round(probs)

def write_csv_file(outputs, output_file):
    # dumps the output file
    with open(output_file, "w") as out_file:
        out_file.write("ID,Output\n")
        for i in range(len(outputs)):
            out_file.write("{}, {}".format(i+1, str(int(outputs[i]))) + "\n")
def get_data(data_file):
    with open(data_file, 'r') as df:
        data = df.readlines()

    data = data[1:]
    data = np.array([[float(col) for col in row.split(',')] for row in data])
    data = data[:, 1:]
    input_data = np.hstack([data, np.ones((len(data), 1), dtype=np.float32)])

    return input_data


def main(args):
    #Load the CSV data and divide it into sequentially chunks of size batch_size
    train_data_loader = DataLoader(args.train_data_file, args.batch_size)
    #Load the test data
    test_data = get_data(args.test_data_file)

    #Perform training, receive a trained and optimized value of W vector
    trained_model_parameters = train(train_data_loader, args.loss_type, args.regularizer_type, args.loss_weight)
    
    #Receive outputs for test data 
    test_data_output = test(test_data, trained_model_parameters)

    #Write these to a file
    write_csv_file(test_data_output, "kaggle1_TAgenerated.csv")

#This is were the code starts execution
#https://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__=="__main__":
    #Initialise a parser instance
    parser = argparse.ArgumentParser()

    #Add arguments to the parser 1 at a time
    #This step tells the parser what it shoudl expect before it acually reads the arguments
    #The --<string name> indicate optional arguments that follow these special symbols
    
    #Pick a loss function type: logistic_loss
    parser.add_argument("--loss", action="store", dest="loss_type", type=str, help="Loss function to be used", default="logistic_loss")
    #Pick a regularizer type: Options 
    parser.add_argument("--regularizer", action="store", dest="regularizer_type", type=str, help="Regularizer to be used", default=None)
    parser.add_argument("--batch-size", action="store", dest="batch_size", type=int, help="Batch size for training", default=20)
    parser.add_argument("--train-data", action="store", dest="train_data_file", type=str, help="Train data file", default="train.csv")
    #Testing/Validation data
    parser.add_argument("--test-data", action="store", dest="test_data_file", type=str, help="Test data file", default="test.csv")
    #The C value...
    parser.add_argument("--loss-weight", action="store", dest="loss_weight", type=float, help="Relative weight", default=1.0)    
    #Reads Command line arguments, converts read arguments to the appropriate type
    args = parser.parse_args()

    main(args)

