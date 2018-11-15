# -*- coding utf-8 -*- 
import pandas as pd
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold,cross_val_score, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,accuracy_score, make_scorer
import matplotlib.pyplot as plt

head = ['0','word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order','word_freq_mail','word_freq_receive', 'word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business', 'word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money',
'word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet',
'word_freq_857','word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts',
'word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re',
'word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!',
'char_freq_$','char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total','spam']
df = pd.read_table('imagens/spambase.data.txt', delim_whitespace=True, header=None,names=head)

for item in head:
	if item != '0': 
		df[item] = ''

df[['word_freq_make','word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order','word_freq_mail','word_freq_receive', 'word_freq_will','word_freq_people','word_freq_report','word_freq_addresses','word_freq_free','word_freq_business',
'word_freq_email','word_freq_you','word_freq_credit','word_freq_your','word_freq_font','word_freq_000','word_freq_money',
'word_freq_hp','word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs','word_freq_telnet',
'word_freq_857','word_freq_data','word_freq_415','word_freq_85','word_freq_technology','word_freq_1999','word_freq_parts',
'word_freq_pm','word_freq_direct','word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re',
'word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(','char_freq_[','char_freq_!',
'char_freq_$','char_freq_#','capital_run_length_average','capital_run_length_longest','capital_run_length_total','spam']] = df['0'].str.split(',', expand=True)
del df['0']
print(df.head(6))

y = df['spam']
X = df.drop(['spam'],axis=1)

def classification_report_with_accuracy_score(y_true, y_pred):

    print classification_report(y_true, y_pred) 
    return accuracy_score(y_true, y_pred) 

clf = MLPClassifier(hidden_layer_sizes=(57,57,57))
predictions = cross_val_predict(clf,X,y,cv=10)
score = cross_val_score(clf,X,y,cv=10,scoring = make_scorer(classification_report_with_accuracy_score))
print(score)



