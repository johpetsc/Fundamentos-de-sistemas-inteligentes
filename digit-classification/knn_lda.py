import pylab as pl
import random as rand
import collections
import numpy as np
from mnist import MNIST
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

mndata = MNIST('data')
mndata.gz = True

def training(mndata): # function that returns the train data
    images, labels = mndata.load_training()

    #index = rand.randrange(0, len(images))
    
    #print(mndata.display(images[index]))
    #print(labels[index])

    Train = collections.namedtuple('Train', ['images', 'labels'])

    return Train(images, labels)

def testing(mndata): # function that returns the test data
    images, labels = mndata.load_testing()

    #index = rand.randrange(0, len(images))
    
    #print(mndata.display(images[index]))
    #print(labels[index])

    Test = collections.namedtuple('Test', ['images', 'labels'])

    return Test(images, labels)

def knn_model():
    train = training(mndata)
    X_train = np.array(train.images)
    Y_train = np.array(train.labels)

    test = testing(mndata)
    X_test = np.array(test.images)
    Y_test = np.array(test.labels)

    knn = KNeighborsClassifier(n_neighbors=6)

    knn.fit(X_train, Y_train)

    print("...")

    pred = knn.predict(X_test)
    
    print(accuracy_score(Y_test, pred))
    print("Confusion Matrix KNN")
    print (confusion_matrix(Y_test, pred))

def lda_model():
    train = training(mndata)
    X_train = np.array(train.images)
    Y_train = np.array(train.labels)

    test = testing(mndata)
    X_test = np.array(test.images)
    Y_test = np.array(test.labels)

    clf = LinearDiscriminantAnalysis()

    clf.fit(X_train, Y_train)

    print("...")

    pred = clf.predict(X_test)

    print(accuracy_score(Y_test, pred))
    print("Confusion Matrix LDA")
    print (confusion_matrix(Y_test, pred))

lda_model()
knn_model()

