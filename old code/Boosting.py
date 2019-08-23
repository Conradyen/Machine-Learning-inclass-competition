#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:11:51 2017

@author: yen
"""
import statsmodels.discrete.discrete_model

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale,MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans

train = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/training_data.csv",\
                    delimiter=",",header = None).values
train_label = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/training_labels.csv",\
                          delimiter=",",header = None).values
test = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/test_data.csv",\
                   delimiter=",",header = None).values
label = train_label[:,0]
                   
standard_scaler = StandardScaler()
train_standardscal = standard_scaler.fit(train).transform(train)
test_standardscal = standard_scaler.fit(train).transform(test)

def compute_error(label,predict):
    
    label_list = label.transpose().tolist()
    count = 0
    for i in range(len(predict)):
        if label_list[i] == predict[i]:
            count += 1
    error = 1-((count / len(predict)))
    return '{:.4f} '.format(error)
scaler = MinMaxScaler()
train_scal = scaler.fit(train).transform(train)
train_log = np.log(train_scal)
##kmeans ==================================================
kmeans = KMeans(n_clusters = 2).fit(train_standardscal)                   
pred_kmeans = kmeans.labels_
import collections
collections.Counter(pred_kmeans)
compute_error(label,pred_kmeans)
kmeans_pred = kmeans.predict(test_standardscal)               
#==========================================================                  
train_scale = scale(train)
test_scale = scale(test)

label = train_label[:,0]
   # 'reg_lambda':[1,2,3,4,5],              
cv_params = {'n_estimators':[1300,1100]}
ind_params = { 'seed':0, 
              'subsample':0.7,
              'min_child_weight':3,
              'max_depth':3,
              'learning_rate': 0.03,
              'num_class':6,
             'objective':"multi:softmax",
             'colsample_bytree':0.8}


optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
optimized_GBM.fit(train_standardscal,label)

optimized_GBM.grid_scores_
#[mean: 0.94147, std: 0.02211, params: {'n_estimators': 800},
# mean: 0.94084, std: 0.02232, params: {'n_estimators': 500},
# mean: 0.93457, std: 0.02266, params: {'n_estimators': 300},
# mean: 0.92013, std: 0.02107, params: {'n_estimators': 100}]
xgd01 = xgb.XGBClassifier(seed = 0, 
              subsample = 0.7,
              min_child_weight =3,
              max_depth = 3,
              learning_rate = 0.03,
              reg_alpha = 0.05,
              n_estimators = 1300,
              num_class = 6,
             objective = "multi:softmax",
             colsample_bytree = 0.8)
xgd01.fit(train_standardscal,label)
pred = xgd01.predict(test)

with open('/Users/yen/Documents/STAT613/kaggle_composition/kmeans_0.9718.csv', 'x') as f :
        f.write('ID,Prediction\n')
        for i in range(len(kmeans_pred)) :
            f.write("".join([str(i+1),',',str(kmeans_pred[i]),'\n']))