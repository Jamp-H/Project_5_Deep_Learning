import tensorflow as tf
import numpy as np

spam = np.genfromtxt("spam.data", delimiter=" ")

def getNormX(data):
	xValuesUnscaled = data[:,:-1]
	mean = xValuesUnscaled.mean(axis=0)
	std = xValuesUnscaled.std(axis=0)
	return (xValuesUnscaled-mean)/std

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
	X = getNormX(data)
	Y = getY(data)
	foldIDArray = np.arange(numFolds)
	np.random.seed(0)
	folds = np.random.permutation(np.tile(foldIDArray, len(Y))[:len(Y)])
	split, xArray, yArray = trainTestSplit(0, folds, X, Y)
	network = makeNetwork(X, 100)
	runNetwork(epochs, xArray, yArray, network)

init(spam, 5, 256)
