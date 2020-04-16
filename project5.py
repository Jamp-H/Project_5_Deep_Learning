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

def init(data, numFolds):
	X = getNormX(data)
	Y = getY(data)
	foldIDArray = np.arange(numFolds)
	np.random.seed(0)
	folds = np.random.permutation(np.tile(foldIDArray, len(Y))[:len(Y)])
	trainTestSplit(0, folds, X, Y)


init(spam, 5)
