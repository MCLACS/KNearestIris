import os    
os.environ['MPLCONFIGDIR'] = "matplot-temp"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def loadAndProcess():
    print('Loading...')
    # load the data...
    iris = load_iris()
    X = iris.data
    y = iris.target
    featureNames = iris.feature_names
    labelNames = iris.target_names
    
    return labelNames, featureNames, X, y
    
def buildTrainAndTest(X, y):
    print('Building train and test sets...')
    # create the train and test sets for X and y
    # traning has 67% of the rows and test has 33% of the rows...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 12)

    # print shape of data sets...
    print('Entire set shape= %s' % str(X.shape))
    print('Training set shape= %s' % str(X_train.shape))
    print('Test set shape= %s' % str(X_test.shape))
    
    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    # train the algorithm...
    print('Training...')
    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(X_train, y_train)
    return (knn, knn.score( X_train , y_train))
    
def test(knn, X_test , y_test):
    # test the decision tree algorithm...
    print('Testing...')
    return knn.score(X_test , y_test)
    
def predict(knn, irisMeasurements):
    print('Precicting...')
    y_classification = knn.predict(irisMeasurements)
    return (y_classification[0])
       
def main():
    print("Running Main...")
    labelNames, featureNames, X, y = loadAndProcess()
    
    #print features and labels...
    print("Features %s" % featureNames)
    print("Labels %s" % labelNames)
    
    # print shape...
    print('X shape: %s' % str(X.shape))
    print('y shape: %s' % str(y.shape))
    
    # print data...
    print('first five rows of X= \n%s' % X[0:6, :])
    print('first 150 rows of y= \n%s' % y[0:150])

    X_train, X_test, y_train, y_test = buildTrainAndTest(X, y)
    print("X_train = %s\n" % X_train)
    print("X_test = %s\n" % X_test)
    print("y_train = %s\n" % y_train)
    print("y_test = %s\n" % y_test)

    knn, score = train(X_train, y_train)
    print("Score on train data %s\n" % score)

    score = test(knn, X_test , y_test)
    print("Score on test data %s\n" % score)

    prediction = predict(knn, [[5,  3.5, 1.3, 0.3]])
    flowerType = labelNames[prediction];
    print("Prediction: f([5,  3.5, 1.3, 0.3])->%s\n" % flowerType)
