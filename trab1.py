#import matplotlib.pyplot as plt
import pylab as pl
import random as rand
import collections
import numpy as np
from pathlib import Path
from mnist import MNIST
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

mndata = MNIST('imagens')
mndata.gz = True

def Training(mndata):
    images, labels = mndata.load_testing()

    index = rand.randrange(0, len(images))
    
    print(mndata.display(images[index]))
    print(labels[index])

    Train = collections.namedtuple('Train', ['images', 'labels'])

    return Train(images, labels)

def Testing(mndata):
    images, labels = mndata.load_training()

    index = rand.randrange(0, len(images))
    
    print(mndata.display(images[index]))
    print(labels[index])

    Test = collections.namedtuple('Test', ['images', 'labels'])

    return Test(images, labels)

def knnk():
    train = Training(mndata)
    X_train = np.array(train.images)
    Y_train = np.array(train.labels)

    test = Testing(mndata)
    X_test = np.array(test.images)
    Y_test = np.array(test.labels)

    knn = KNeighborsClassifier(n_neighbors=0)

    if Path('./model.pkl').is_file():
        knn = joblib.load('model.pkl')
    else:
        knn.fit(X_train, Y_train)
        joblib.dump(knn, 'model.pkl')

    print(len(Y_test))
    print(np.shape(Y_test))

    pred = knn.predict(X_train)

    print(accuracy_score(Y_test, pred))


knnk()