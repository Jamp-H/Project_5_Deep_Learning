###############################################################################
#
# AUTHOR(S): Joshua Holguin, Jacob
# DESCRIPTION: program that will implement stochastic gradient descent algorithm
# for a one layer neural network with edits made to the regularization parameter
# VERSION: 0.0.1v
#
###############################################################################
import numpy as np
from sklearn.preprocessing import scale
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
### TODO:
### - Point to show min of each val loss curve
### - Get best_param_val - reg param that minimized val loss
### - retrain entire data set on ^^
### - Make pred on test set
### - PDF Doc


def getNormX(data):
    xValuesUnscaled = data[:,:-1]
    X_sc = scale(xValuesUnscaled)
    return X_sc


def getY(data):
    return data[:,-1]

# Function: run_single_layered_NN
# INPUT ARGS:
# X_mat - scaled X matrix for a data set
# y_vec - corresponding outputs for our X_mat
# hidden_layers - Vector of hidden layers
# num_epochs - integer value for the number of epochs wanted
# data_set - string value decribing the data set being ran (Test, Train, Subtrain, ect.)
# Return: array of history objects containing data about the models created
def run_single_layered_NN(X_mat, y_vec, hidden_layers, num_epochs, data_set):
    # set model variable to keep track on which number model is being ran
    model_number = 1

    # creat list of model data
    model_data_list = []

    # create a neural network with 1 hidden layer
    for hidden_layer in hidden_layers:
        # set model for single layered NN
        model = keras.Sequential([
        keras.layers.Flatten(input_shape=(np.size(X_mat, 1), )), # input layer
        keras.layers.Dense(hidden_layer, activation='sigmoid', use_bias=False), # hidden layer
        keras.layers.Dense(1, activation='sigmoid', use_bias=False) # output layer
        ])

        # compile the models
        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # fit the models
        print(f"\nModel {model_number} {data_set}")
        print("==============================================")
        model_data = model.fit(
                                    x=X_mat,
                                    y=y_vec,
                                    epochs=num_epochs,
                                    verbose=0,
                                    validation_split=.05)

        # update model number
        model_number += 1

        # apend model data to list
        model_data_list.append(model_data)



    return model_data_list


# Function: graph_model_data
# INPUT ARGS:
# model_data_list: lsit of data generated by NN model
# num_epochs: total number of epoches used for the models
# line_style: style the line should be on the output graph
# model_data: a list of valuesdectionary or values from our single NN model
def graph_model_data(model_data_list, num_epochs, set):

    colors = ['lightblue', 'darkblue', 'orange' , 'black']
    line_style = ['solid', 'dashed']



    set_index = 0


    for model_data in model_data_list:
        color_index = 0
        model_index = 1
        for data in model_data:
            if(set[set_index] == 'Train'):
                plt.plot(range(0,num_epochs), data.history['val_loss'], markevery=num_epochs,
                                color = colors[color_index], linestyle = line_style[set_index],
                                label = f"Model {model_index} {set[set_index]} Data")

            if(set[set_index] == 'Subtrain'):
                plt.plot(range(0,num_epochs), data.history['val_loss'], markevery=num_epochs,
                                color = colors[color_index], linestyle = line_style[set_index],
                                label = f"Model {model_index} {set[set_index]} Data")


            if(set[set_index] == 'Validation'):
                plt.plot(range(0,num_epochs), data.history['val_loss'],
                                color = colors[color_index], linestyle = line_style[set_index],
                                label = f"Model {model_index} {set[set_index]} Data")

            color_index += 1

            model_index += 1
        set_index += 1
    # add grid to graphs
    plt.grid(True)

    # Add x nd y labels
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.legend(fontsize = 'small', loc = 'upper right',
                bbox_to_anchor = (1.05, 1.05))

    #display graph
    plt.savefig("Loss Graph")


def init(data, epochs):
    # Get X (matrix) and y (vec) data
    X_sc = getNormX(data)
    y_vec = getY(data)
    # make array or hidden units
    hidden_units_vec = 2 ** np.arange(4)

    # divide data into 80% train and 20% test
    X_train, X_test = np.split( X_sc, [int(.8 * len(X_sc))])
    y_train, y_test = np.split( y_vec, [int(.8 * len(y_vec))])

    # split train data into 50% subtrain and 50% validation
    X_subtrain, X_validation = np.split( X_train, [int(.5 * len(X_train))])
    y_subtrain, y_validation = np.split( y_train, [int(.5 * len(y_train))])

    np.random.seed(0)

    model_data_subtrain_list = run_single_layered_NN(X_subtrain, y_subtrain,
                                    hidden_units_vec, epochs, "Subtrain")

    model_data_valid_list = run_single_layered_NN(X_validation, y_validation,
                                    hidden_units_vec, epochs, "Validataion")

    model_data_list = [model_data_subtrain_list, model_data_valid_list]

    # plot data
    graph_model_data(model_data_list, epochs, ["Subtrain" ,"Validation"])



def main():
    # get spam data
    spam = np.genfromtxt("spam.data", delimiter=" ")

    init(spam, 10)

main()
