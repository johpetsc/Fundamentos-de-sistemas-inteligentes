# -*- coding: utf-8 -*- 
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

names = ['Class','Specimen Number','Eccentricity','Aspect Ratio','Elongation','Solidity','Stochastic Convexity','Isoperimetric Factor','Maximal Indentation Depth','Lobedness','Average Intensity','Average Contrast','Smoothness','Third moment','Uniformity','Entropy']
folhas = pd.read_csv("data/leaf.csv", names = names, header=None)
labels = folhas.pop('Class').values 
images = folhas.values

rf = RandomForestClassifier(n_estimators=25)
fit = rf.fit(images,labels)
predict = cross_val_predict(rf, images, labels, cv = 10)
score = cross_val_score(rf, images, labels, cv = 10)
prob = rf.predict_proba(images)

print("\n\n\n\n\nLeaf Values:\n")
print(folhas.head(340))
print("\n\nProbability:\n")
print(prob)
print("\n\nCross validation score:\n")
print(score)
print("\n\nDecision path:\n")
print(rf.decision_path(images))
print("\n\nAccuracy:\n")
print(accuracy_score(labels, predict))

feat_importances = pd.Series(rf.feature_importances_, index=folhas.columns)
feat_importances.nlargest(14).plot(kind='barh', title = 'Importancia dos Atributos/Features')
cm = confusion_matrix(labels, predict)
plt.matshow(cm)
plt.ylabel('X')
plt.xlabel('Y')
plt.title('MATRIZ DE CONFUSAO')
plt.colorbar()
plt.show()
