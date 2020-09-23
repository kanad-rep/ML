# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:24:56 2019

@author: Kanad Das
"""

import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn import grid_search
from sklearn import cross_validation
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

file3 = open(r"E:\Work\Machine Learning\Assignments\Assignment 1\Spambase data\spambase.data","r")
f = list(csv.reader(file3,delimiter = ","))
file3.close()
data1 = pd.DataFrame(data = f)

bnb = BernoulliNB()
gnb = GaussianNB()
mnb = MultinomialNB()

#Splitting of the dataset
trn_data,tst_data = train_test_split(data1, test_size = 0.2)
trn_class_label = trn_data.iloc[:,-1]
tst_class_label = tst_data.iloc[:,-1]

##KNN classifier
#K=[]
#for i in range(3,30):
#    K.append(int(i))
#param_grid =[{'n_neighbors':K,'metric':['euclidean','manhattan','minkowski','hamming']}]
#svr = KNeighborsClassifier()
#
#grid = grid_search.GridSearchCV(svr,param_grid,cv=2,scoring='accuracy')          
#grid.fit(trn_data,trn_class_label)    
#clf= grid.best_estimator_
#print ('\n The best grid is as follows: \n')
#print (grid.best_estimator_)

#Naive Bayes Classifier
clf = bnb.fit(trn_data,trn_class_label)
#clf = gnb.fit(trn_data,trn_class_label)
#clf = mnb.fit(trn_data,trn_class_label)

#predicted_class_label = cross_validation.cross_val_predict(mnb, tst_data, tst_class_label, cv=10) 
predicted_class_label = clf.predict(tst_data)     
predicted_class_label = list(predicted_class_label)

## Evaluation using Precision, Recall and F-measure  
print ('\n Confusion Matrix \n')
print (confusion_matrix(tst_class_label, predicted_class_label))
pr=precision_score(tst_class_label, predicted_class_label, average='macro')
print ('\n Precision:'+str(pr))
re=recall_score(tst_class_label, predicted_class_label, average='macro')
print ('\n Recall:'+str(re))
fm=f1_score(tst_class_label, predicted_class_label, average='macro') 
print ('\n F-measure:'+str(fm))

print ('\n Confusion Matrix \n')
print (confusion_matrix(tst_class_label, predicted_class_label))
pr=precision_score(tst_class_label, predicted_class_label, average='micro')
print ('\n Precision:'+str(pr))
re=recall_score(tst_class_label, predicted_class_label, average='micro')
print ('\n Recall:'+str(re))
fm=f1_score(tst_class_label, predicted_class_label, average='micro') 
print ('\n F-measure:'+str(fm))