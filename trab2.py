# -*- coding: utf-8 -*- 
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sys

#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#Leitura do arquivo .csv usando a biblioteca pandas
names = ['Class','Specimen Number','Eccentricity','Aspect Ratio','Elongation','Solidity','Stochastic Convexity','Isoperimetric Factor','Maximal Indentation Depth','Lobedness','Average Intensity','Average Contrast','Smoothness','Third moment','Uniformity','Entropy']
folhas = pd.read_csv("imagens/leaf.csv", names = names, header=None)
labels = folhas.pop('Class').values 
images = folhas.values

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#Determinando os valores da floresta utilizando a biblioteca sklearn
rf = RandomForestClassifier(n_estimators=50)
fit = rf.fit(images,labels)
predict = cross_val_predict(rf, images, labels, cv = 10)
score = cross_val_score(rf, images, labels, cv = 10)
prob = rf.predict_proba(images)

#Limpa os warnings da tela
print("\033[H\033[J")
print("Pressione ENTER para continuar.\n")
sys.stdin.read(1)

print("\n\n\n\n\nValores das folhas:\n")
print(folhas.head(340))
sys.stdin.read(1)
print("\n\nProbabillidade:\n")
print(prob)
sys.stdin.read(1)
print("\n\nValor validação cruzada:\n")
print(score)
sys.stdin.read(1)
print("\n\nCaminho de decisão da árvore\n")
print(rf.decision_path(images))
sys.stdin.read(1)
print("\n\nPrecisão média:\n")
print(accuracy_score(labels, predict))
sys.stdin.read(1)
print("\n\nMatriz de confusão:\n")
print(confusion_matrix(labels, predict))
sys.stdin.read(1)

#Plota um gráfico dos atributos/features mais importantes
feat_importances = pd.Series(rf.feature_importances_, index=folhas.columns)
feat_importances.nlargest(14).plot(kind='barh', title = 'Importancia dos Atributos/Features')

cm = confusion_matrix(labels, predict)
plt.matshow(cm)
plt.ylabel('X')
plt.xlabel('Y')
plt.title('MATRIZ DE CONFUSAO')
plt.colorbar()
plt.show()
