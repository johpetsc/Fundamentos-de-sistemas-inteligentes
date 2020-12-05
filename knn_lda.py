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

mndata = MNIST('imagens')
mndata.gz = True

def Training(mndata): #funcao para ler os dados de treino
    images, labels = mndata.load_testing()

    #index = rand.randrange(0, len(images))
    
    #print(mndata.display(images[index]))
    #print(labels[index])

    Train = collections.namedtuple('Train', ['images', 'labels'])

    return Train(images, labels)

def Testing(mndata): #funcao para ler os dados de teste
    images, labels = mndata.load_training()

    #index = rand.randrange(0, len(images))
    
    #print(mndata.display(images[index]))
    #print(labels[index])

    Test = collections.namedtuple('Test', ['images', 'labels'])

    return Test(images, labels)

def knnk():
    train = Training(mndata)
    X_train = np.array(train.images)
    Y_train = np.array(train.labels)

    test = Testing(mndata)
    X_test = np.array(test.images)
    Y_test = np.array(test.labels)

    knn = KNeighborsClassifier(n_neighbors=6)

    knn.fit(X_train, Y_train)

    print("...")

    pred = knn.predict(X_test)
    
    print(accuracy_score(Y_test, pred))
    print("Matriz de Confusao KNN")
    print (confusion_matrix(Y_test, pred))

def lda():
    train = Training(mndata)
    X_train = np.array(train.images)
    Y_train = np.array(train.labels)

    test = Testing(mndata)
    X_test = np.array(test.images)
    Y_test = np.array(test.labels)

    clf = LinearDiscriminantAnalysis()

    clf.fit(X_train, Y_train)

    print("...")

    pred = clf.predict(X_test)

    print(accuracy_score(Y_test, pred))
    print("Matriz de Confusao LDA")
    print (confusion_matrix(Y_test, pred))

lda()
knnk()

