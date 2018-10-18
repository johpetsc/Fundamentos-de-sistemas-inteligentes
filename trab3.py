# -*- coding: utf-8 -*- 
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
import sys

#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#Leitura do arquivo .csv usando a biblioteca pandas
peitos = pd.read_csv("imagens/wdbc.data.csv", header=None)
labels = peitos[peitos.columns[1]] 
images = peitos[peitos.columns[2:len(peitos.columns)]].values

#Limpa os warnings da tela
print("\033[H\033[J")
print("Pressione ENTER para continuar.\n")
sys.stdin.read(1)

print("\n\n\n\n\nLista de atributos do arquivo:\n")
print(peitos.head(569))
sys.stdin.read(1)
c = 2
#"Divida os dados em dois conjuntos (70/30), treinamento e teste, de forma aleatória."
xtrain, xtest, ytrain, ytest = train_test_split(images, labels, test_size = 0.3)
#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
for x in range(-5, 15):
    #"Treine um SVM com kernel linear, encontre uma margem ótima C, plote em um gráfico os erros de teste vs o valor de C em escala logarítmica[2^-5, 2^15]."
    svm = SVC(C = c ** x, kernel = 'linear')
    fit = svm.fit(xtrain, ytrain)
    predict = svm.predict(xtest)
    accuracy = accuracy_score(ytest, predict)

    print("\n\nPrecisão com kernel linear e c = 2^", x, ":\n")
    print(accuracy)

#"Treine um SVM com kernel FBR."
svm = SVC(kernel = 'rbf')
fit = svm.fit(xtrain, ytrain)
predict = svm.predict(xtest)
accuracy = accuracy_score(ytest, predict)

print("\n\nPrecisão com kernel FBR:\n")
print(accuracy)

