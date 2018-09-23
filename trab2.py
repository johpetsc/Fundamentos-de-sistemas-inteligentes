import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
folhas = pd.read_csv("imagens/leaf.csv", header=None)
print(folhas.head(340))
labels = folhas.pop(0).values
images = folhas.values

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
rf = RandomForestClassifier(n_estimators=50)
fit = rf.fit(images,labels)
predict = cross_val_predict(rf, images, labels)
prob = rf.predict_proba(images)
print(prob)
print(rf.decision_path(images))
print (accuracy_score(labels, predict))

cm = confusion_matrix(labels, predict)
plt.matshow(cm)
plt.ylabel('X')
plt.xlabel('Y')
plt.title('Confusion matrix')
plt.colorbar()
plt.show()


#print(confusion_matrix(labels, y_pred))