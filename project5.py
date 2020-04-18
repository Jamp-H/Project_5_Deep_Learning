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
#import tensorflow_docs as tfdocs
#import tensorflow_docs.modeling
#import tensorflow_docs.plots
### TODO:
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
def run_single_layered_NN(X_mat, y_vec, val_data ,hidden_layers, num_epochs, data_set):
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
                                    validation_data = val_data,
                                    verbose=0,)

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
def graph_model_data(model_data_list, num_epochs):
    colors = ['lightblue', 'darkblue', 'orange' , 'black', 'yellow']
    line_style = ['solid', 'dashed']
    
    model_index = 0
    color_index = 0
    set_index = 0
    
    for data in model_data_list:
        plt.plot(range(0,num_epochs), data.history['loss'], markevery=num_epochs,
                        color = colors[color_index], linestyle = 'solid',
                        label = f"Model {model_index} Subtrain Data")
        valMin = np.amin(data.history['val_loss'])
        argMin = np.argmin(data.history['val_loss'])
        plt.plot(argMin, valMin, marker='o', color = colors[color_index])
        
        plt.plot(range(0,num_epochs), data.history['val_loss'],
                        color = colors[color_index], linestyle = 'dashed',
                        label = f"Model {model_index} Validation Data")
        valMin = np.amin(data.history['val_loss'])
        argMin = np.argmin(data.history['val_loss'])
        plt.plot(argMin, valMin, marker='o', color = colors[color_index])

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
    plt.clf()

def init(data, epochs):
    np.random.seed(5)
    # Get X (matrix) and y (vec) data
    X_sc = np.random.permutation(getNormX(data))
    y_vec = getY(data)
    # make array or hidden units
    hidden_units_vec = 2 ** np.arange(4)

    # divide data into 80% train and 20% test
    X_train, X_test = np.split( X_sc, [int(.8 * len(X_sc))])
    y_train, y_test = np.split( y_vec, [int(.8 * len(y_vec))])

    # split train data into 50% subtrain and 50% validation
    X_subtrain, X_validation = np.split( X_train, [int(.5 * len(X_train))])
    y_subtrain, y_validation = np.split( y_train, [int(.5 * len(y_train))])
    val_data = (X_validation,y_validation)

    model_data_subtrain_list = run_single_layered_NN(X_subtrain, y_subtrain, val_data,
                                    hidden_units_vec, epochs, "Subtrain")


    model_data_list = [model_data_subtrain_list]


    # plot data
    print(model_data_subtrain_list[0].history.keys())


    graph_model_data(model_data_subtrain_list, epochs)



    best_num_units = []
    # get best number of epochs besed off validation data
    for model in model_data_subtrain_list:
        best_num_units.append(np.amin(model.history['val_loss']))
    print("loss", best_num_units)
    print("2 pow", hidden_units_vec)
    # Retrain whole train data based of best num of epochs
    plt.plot(hidden_units_vec, best_num_units)
    valMin = np.amin(best_num_units)
    argMin = np.argmin(best_num_units)
    plt.plot(argMin,valMin,marker="o")
                
    plt.xlabel('Number Of Units in Hidden Layer')
    plt.ylabel("Loss")

    #display graph
    plt.savefig("Units Graph")


def main():
    # get spam data
    spam = np.genfromtxt("spam.data", delimiter=" ")

    init(spam, 40)

main()
