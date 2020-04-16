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


def init(data, numFolds, epochs):
	# Get X (matrix) and Y (vec) data
	X = getNormX(data)
	Y = getY(data)

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
