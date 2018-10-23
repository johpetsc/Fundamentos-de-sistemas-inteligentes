# -*- coding: utf-8 -*- 
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, learning_curve
from matplotlib import pyplot as plt
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
erros = [0] * 21
for x in range(-5, 15):
    #"Treine um SVM com kernel linear, encontre uma margem ótima C, plote em um gráfico os erros de teste vs o valor de C em escala logarítmica[2^-5, 2^15]."
    svm = SVC(C = c ** x, kernel = 'linear')
    fit = svm.fit(xtrain, ytrain)
    predict = svm.predict(xtest)
    accuracy = accuracy_score(ytest, predict)
    erros[x] = len(ytest) - (accuracy * len(ytest))
    print("\n\nPrecisão com kernel linear e c = 2^", x, ":\n")
    print(accuracy)
    print("Erros: ",erros[x])

#"Plota o gráfico dos erros de teste vs o valor de c."

c = ('2^-5', '2^-4','2^-3','2^-2','2^-1','2^0','2^1','2^2','2^3','2^4','2^5','2^6','2^7','2^8','2^9','2^10','2^11','2^12','2^13','2^14','2^15')
y_pos = np.arange(len(c))
plt.barh(y_pos, erros)
plt.yticks(y_pos,c)

plt.xlabel('Numero de Erros')
plt.ylabel('Valores de C')
plt.show()
#"Treine um SVM com kernel Gaussiano com sigma=10."
def gaussianKernel(X1, X2, sigma = 10):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            x1 = x1.flatten()
            x2 = x2.flatten()
            gram_matrix[i, j] = np.exp(- np.sum( np.power((x1 - x2),2) ) / float( 2*(sigma**2) ) )
    return gram_matrix

clf = SVC(kernel="precomputed")
model = clf.fit( gaussianKernel(xtrain,xtrain), ytrain )
predict1 = model.predict(gaussianKernel(xtest,xtrain))
accuracy1 = accuracy_score(ytest, predict1)
print("\n\nPrecisão com kernel Gaussiano e Sigma = 10:\n")
print(accuracy1)
print(classification_report(ytest, predict1)) 

#Plota a matriz de confusão
cm = confusion_matrix(ytest, predict1)
plt.matshow(cm)
plt.ylabel('X')
plt.xlabel('Y')
plt.title('MATRIZ DE CONFUSAO GAUSSIANO')
plt.colorbar()
plt.show()

#"Treine um SVM com kernel FBR."
svm = SVC(kernel = 'rbf')
fit = svm.fit(xtrain, ytrain)
predict = svm.predict(xtest)
accuracy = accuracy_score(ytest, predict)

print("\n\nPrecisão com kernel FBR:\n")
print(accuracy)
print(classification_report(ytest, predict1)) 

#Plota a matriz de confusão
cm = confusion_matrix(ytest, predict)
plt.matshow(cm)
plt.ylabel('X')
plt.xlabel('Y')
plt.title('MATRIZ DE CONFUSAO FBR')
plt.colorbar()
plt.show()

