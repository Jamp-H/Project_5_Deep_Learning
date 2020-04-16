###############################################################################
#
# AUTHOR(S): Joshua Holguin, Jacob
# DESCRIPTION: program that will implement stochastic gradient descent algorithm
# for a one layer neural network with edits made to the regularization parameter
# VERSION: 0.0.1v
#
###############################################################################
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale
### TODO:
### - Add comments and function headers
### - Edit Regularization parameters
### - Loop over Regularization parameter(s)
### - Plot Log loss vs regularization parameter (diff color each set)
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


def trainTestSplit(fold, foldArray, x, y):
	split = {
		"test": foldArray == fold,
		"train": foldArray != fold
	}
	xArray = {}
	yArray = {}
	for setName, isSet in split.items():
		xArray[setName] = x[isSet, :]
		yArray[setName] = y[isSet]
	return split, xArray, yArray


def makeNetwork(x, hiddenUnits):
	inputs = tf.keras.Input(shape=(x.shape[1],))
	hidden = tf.keras.layers.Dense(
			hiddenUnits, activation="sigmoid", use_bias=False)(inputs)
	outputs = tf.keras.layers.Dense(
		 1, activation="sigmoid", use_bias=False)(hidden)
	network = tf.keras.Model(inputs=inputs, outputs=outputs, name="spam_model")
	network.compile(
			optimizer=tf.keras.optimizers.Adam(),
			loss=tf.keras.losses.binary_crossentropy,
	)
	return network


def runNetwork(epochs, xArray, yArray, network):
	history = network.fit(
			xArray["train"],
			yArray["train"],
			epochs=epochs,
			verbose=2,
			validation_split=0.5)
	test_scores = dict(zip(
			network.metrics_names,
			network.evaluate(xArray["test"], yArray["test"])))
	history.history

def run_single_layered_NN(X_mat, y_vec, hidden_layers, num_epochs, data_set):
    # set model variable to keep track on which number model is being ran
    model_number = 1

    # list of colors for hidden layers

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

def init(data, numFolds, epochs):
	# Get X (matrix) and Y (vec) data
	X = getNormX(data)
	Y = getY(data)
	hidden_units_vec = range(6)
	hidden_units_vec = np.power(hidden_units_vec, 2)
	print(hidden_units_vec)
	foldIDArray = np.arange(numFolds)
	np.random.seed(0)
	folds = np.random.permutation(np.tile(foldIDArray, len(Y))[:len(Y)])
	split, xArray, yArray = trainTestSplit(0, folds, X, Y)
	network = makeNetwork(X, 100)
	runNetwork(epochs, xArray, yArray, network)


def main():
	# get spam data
	spam = np.genfromtxt("spam.data", delimiter=" ")

	init(spam, 5, 256)

main()
