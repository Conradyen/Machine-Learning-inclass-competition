#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:51:19 2017

@author: yen
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


train = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/training_data.csv",\
                    delimiter=",",header = None).values
train_label = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/training_labels.csv",\
                          delimiter=",",header = None).values
test = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/test_data.csv",\
                   delimiter=",",header = None).values
                   
groups = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/training_subjects.csv",
                     delimiter=",",header = None).values
label = train_label[:,0].ravel()

#log data
train_1 = train+1.5
log_train = np.log(train_1)
#sqrt data
sqrt_train = np.sqrt(train+1.5)

X_train = sqrt_train[1912:6373,:]
X_val = sqrt_train[1:1913,:]
y_train = train_label[1912:6373,:]
y_val = train_label[1:1913,:]


                   
standard_scaler = StandardScaler()
train_standardscal = standard_scaler.fit(train).transform(train)
test_standardscal = standard_scaler.fit(train).transform(test)

minmaxscaler = MinMaxScaler(feature_range=(-1, 1))
train_minmax = minmaxscaler.fit(train).transform(train)
test_minmax = minmaxscaler.fit(train).transform(test)




param_grid = {
        'min_samples_split' :[5,6,7,8],
        'n_estimators':[400,500,600]
}
rfc = RandomForestClassifier(max_depth = 9,min_samples_split = 4,n_estimators = 400)
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(train_standardscal,label)
CV_rfc.grid_scores_
'''
[mean: 0.92500, std: 0.01839, params: {'min_samples_split': 2, 'n_estimators': 200},
 mean: 0.92296, std: 0.02032, params: {'min_samples_split': 2, 'n_estimators': 300},
 mean: 0.92594, std: 0.01831, params: {'min_samples_split': 2, 'n_estimators': 400},
 mean: 0.92609, std: 0.01909, params: {'min_samples_split': 3, 'n_estimators': 200},
 mean: 0.92390, std: 0.01856, params: {'min_samples_split': 3, 'n_estimators': 300},
 mean: 0.92531, std: 0.01828, params: {'min_samples_split': 3, 'n_estimators': 400},
 mean: 0.92515, std: 0.02063, params: {'min_samples_split': 4, 'n_estimators': 200},
 mean: 0.92468, std: 0.01794, params: {'min_samples_split': 4, 'n_estimators': 300},
 mean: 0.92358, std: 0.01781, params: {'min_samples_split': 4, 'n_estimators': 400},
 mean: 0.92531, std: 0.02068, params: {'min_samples_split': 5, 'n_estimators': 200},
 mean: 0.92578, std: 0.01846, params: {'min_samples_split': 5, 'n_estimators': 300},
 mean: 0.92641, std: 0.01877, params: {'min_samples_split': 5, 'n_estimators': 400}]
'''


group_kfold = GroupKFold(n_splits=5)
score = 0
for k,(train_set,test_set) in enumerate(group_kfold.split(train_minmax,label,groups)):
    model = RandomForestClassifier(n_estimators = 200,max_depth = 9,min_samples_split = 4)
    model.fit(train_minmax[train_set],label[train_set])
#    pred = model.predict(test_standardscal)
    score = model.score(train_minmax[test_set],label[test_set])
#    with open('/Users/yen/Documents/STAT613/kaggle_composition/Random_forast/'+str(score)+'.csv', 'x') as f :
#        f.write('ID,Prediction\n')
#        for i in range(len(pred)) :
#            f.write("".join([str(i+1),',',str(pred[i]),'\n']))
    print(score)
    print(score)
#print(score/5)

#Logistic regression
train_1 = train+1.5
log_train = np.log(train_1)

group_kfold = GroupKFold(n_splits=5)
for c in [1,0.3,0.1,0.03,0.01,0.003,0.001]:
    print(c)
    score_sum = 0
    for k,(train_set,test_set) in enumerate(group_kfold.split(train_minmax,label,groups)):
        model = LogisticRegression(C = c)
        model.fit(log_train[train_set],label[train_set])
    #    pred = model.predict(test_standardscal)
        score = model.score(log_train[test_set],label[test_set])
    #    with open('/Users/yen/Documents/STAT613/kaggle_composition/Logistic_reg/'+str(score)+'.csv', 'x') as f :
    #        f.write('ID,Prediction\n')
    #        for i in range(len(pred)) :
    #            f.write("".join([str(i+1),',',str(pred[i]),'\n']))
        print(score)
        score_sum += score 
    print(score_sum/5)

#c = 1

model = LogisticRegression(C = 0.1)
model.fit(train_standardscal[train_set],label[train_set])
model.predict()

model = RandomForestClassifier(n_estimators = 200,max_depth = 9,min_samples_split = 4)
model.fit(train_standardscal,label)
pred = model.predict(test_standardscal)

##LDA
from sklearn.lda import LDA
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, train_label, test_size=0.3)
lda = LDA()
lda.fit(X_train,y_train)
score = lda.score(X_val,y_val)
print(score)

group_kfold = GroupKFold(n_splits=5)
for k,(train_set,test_set) in enumerate(group_kfold.split(train_minmax,label,groups)):
    lda = LDA()
    lda.fit(train_minmax[train_set],label[train_set])
    score = lda.score(train_minmax[test_set],label[test_set])
    print(score)
'''
0.899569583931
0.938892882818
0.947894361171
0.972527472527
0.944292237443
'''
##knn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, train_label, test_size=0.3)
knn = KNeighborsClassifier(n_neighbors = 19)
knn.fit(X_train,y_train)
score = knn.score(X_val,y_val)
print(score)

group_kfold = GroupKFold(n_splits=5)
for n in range(17,30,2):
    score_sum = 0
    for k,(train_set,test_set) in enumerate(group_kfold.split(train_minmax,label,groups)):
        knn = KNeighborsClassifier(n_neighbors = n)
        knn.fit(train_minmax[train_set],label[train_set])
        score = knn.score(train_minmax[test_set],label[test_set])
        score_sum += score
        print("{:^10}{:^10}".format(n,score))
    print("avg score:",score_sum/5)

##NB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, train_label, test_size=0.3)
nb = GaussianNB()
nb.fit(X_train,y_train)
score = nb.score(X_val,y_val)
print(score)

group_kfold = GroupKFold(n_splits=5)
score_sum = 0
for k,(train_set,test_set) in enumerate(group_kfold.split(train_minmax,label,groups)):
        nb = GaussianNB()
        nb.fit(train_minmax[train_set],label[train_set])
        score = nb.score(train_minmax[test_set],label[test_set])
        score_sum += score
        print("{:^10}".format(score))
print("avg score:",score_sum/5)

with open('/Users/yen/Documents/STAT613/kaggle_composition/Random_forast/0.92762.csv', 'x') as f :
        f.write('ID,Prediction\n')
        for i in range(len(pred)) :
            f.write("".join([str(i+1),',',str(pred[i]),'\n']))