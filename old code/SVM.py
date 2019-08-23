#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:25:17 2017

@author: yen
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GroupKFold

train = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/training_data.csv",\
                    delimiter=",",header = None).values
train_label = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/training_labels.csv",\
                          delimiter=",",header = None).values.ravel()
test = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/test_data.csv",\
                   delimiter=",",header = None).values
groups = pd.read_csv("/Users/yen/Documents/STAT613/kaggle_composition/training_subjects.csv",
                     delimiter=",",header = None).values

X_train, X_val, y_train, y_val = train_test_split(train, train_label, test_size=0.3, \
                                                  shuffle = False)

scaler = MinMaxScaler(feature_range=(-1, 1))
train_scal = scaler.fit(train).transform(train)
test_scal = scaler.fit(train).transform(test)
##right scaler !!!!!!
standard_scaler = StandardScaler()
train_standardscal = standard_scaler.fit(train).transform(train)
mean = train_standardscal.mean(axis=0)

test_standardscal = standard_scaler.fit(train).transform(test)

X_train, X_val, y_train, y_val = train_test_split(train_standardscal, train_label, test_size=0.3, \
                                                  random_state=420)

#model 1                   
model = SVC(C = 0.01,kernel = 'linear',decision_function_shape='ovo')
model.fit(X_train,y_train)
model.score( X_val,y_val)

pred = model.predict(X_val)
#model 2
model_linear = LinearSVC(penalty = 'l2',C = 0.03,max_iter=1000)
model_linear.fit(X_train,y_train)

pred = model.predict(X_val)

def compute_accuracy(label,predict):
    
    label_list = label.transpose().tolist()[0]
    predict_list = predict.tolist()
    count = 0
    for i in range(len(predict_list)):
        if label_list[i] == predict_list[i]:
            count += 1
    accuracy = (count / len(predict)) * 100
    return '{:.4f} %'.format(accuracy)

compute_accuracy(y_val,pred)

gamma = np.linspace(0.002, 0.003, num=50)
for g in gamma:
    model = SVC(kernel = 'rbf',gamma = g,decision_function_shape='ovo')
    model.fit(X_train,y_train)
    pred = model.predict(X_val)
    acc = compute_accuracy(y_val,pred)
    print('{:^20}{:^20}'.format(str(round(g,4)),acc))
#kernel = ['linear','poly','rbf','sigmoid']
#
for c in [1,1.22,1.23,1.24,1.25,1.26,1.27]:
    model_linear = SVC(C = c,kernel = 'linear')
    model_linear.fit(X_train,y_train)
    pred = model_linear.predict(X_val)
    acc = compute_accuracy(y_val,pred)
    print('{:^5}{:^5}'.format(str(c),acc))
    
for c in [1,1.3,1.1,1.4,1.2,1.5]:
    model = SVC(C = c,kernel = 'rbf',gamma = 0.0023,decision_function_shape='ovr')
    model.fit(X_train,y_train)
    pred = model.predict(X_val)
    acc = compute_accuracy(y_val,pred)
    print('{:^5}{:^5}'.format(str(c),acc))

model = SVC(C = 0.01,kernel = 'linear',decision_function_shape='ovo')
k_fold = KFold(n_splits = 18)
sum_acc = 0
for k,(train_set,test_set) in enumerate(k_fold.split(train_standardscal,train_label)):
 #   print(train,test)
    model = SVC(C = 1.2,kernel = 'rbf',gamma = 0.0023,decision_function_shape='ovo')
    model.fit(train_standardscal[train_set],train_label[train_set])
    acc = model.score(train_standardscal[test_set],train_label[test_set])
#    pred = model.predict(train_standardscal[test_set])
#    acc = compute_accuracy(train_label[test_set],pred)
    sum_acc+=acc
    print(acc)
print(sum_acc/18)

for k,(train_set,test_set) in enumerate(k_fold.split(train,train_label)):
    model_linear = LinearSVC(C = 0.03)
    model_linear.fit(train[train_set],train_label[train_set])
    pred = model_linear.predict(train[test_set])
    acc = compute_accuracy(train_label[test_set],pred)
    print(acc)

print('{:^10}{:^10}{:^10}{:^10}'.format('n','k','lr','acc'))
#for n in range(5,10):
group_kfold = GroupKFold(n_splits=18)
acc_sum = 0
for k,(train_set,test_set) in enumerate(group_kfold.split(train_standardscal,train_label,groups)):
    model = SVC(C = 0.01,kernel = 'linear',decision_function_shape='ovo')
    model.fit(train_standardscal[train_set],train_label[train_set])
    acc = model.score(train_standardscal[test_set],train_label[test_set])
    acc_sum+=acc
    print('{:^10}{:^10}'.format(k,acc))

#n = 5,lr = 0.01
'''
    n         k         lr       acc    
    2         0         1     0.913084405396925
    2         1         1     0.9083490269930948
    2         0        0.1    0.9159083777847505
    2         1        0.1    0.9058380414312618
    2         0        0.01   0.9221838719799185
    2         1        0.01   0.9074074074074074
    2         0       0.001   0.9137119548164417
    2         1       0.001   0.9011299435028248
    3         0         1     0.8787163756488909
    3         1         1     0.9479362101313321
    3         2         1     0.9245994344957588
    3         0        0.1    0.8806040585181689
    3         1        0.1    0.9530956848030019
    3         2        0.1    0.9293119698397738
    3         0        0.01   0.8806040585181689
    3         1        0.01   0.9530956848030019
    3         2        0.01   0.9307257304429783
    3         0       0.001   0.8834355828220859
    3         1       0.001   0.9469981238273921
    3         2       0.001   0.9170593779453345
    4         0         1     0.8602076124567474
    4         1         1     0.9425207756232687
    4         2         1     0.9018369690011481
    4         3         1     0.9598163030998852
    4         0        0.1    0.8560553633217993
    4         1        0.1    0.945983379501385
    4         2        0.1    0.9024110218140069
    4         3        0.1    0.9621125143513203
    4         0        0.01   0.8629757785467128
    4         1        0.01   0.9404432132963989
    4         2        0.01   0.9024110218140069
    4         3        0.01   0.9644087256027555
    4         0       0.001   0.8608996539792387
    4         1       0.001   0.9376731301939059
    4         2       0.001   0.8938002296211252
    4         3       0.001   0.9569460390355913
    5         0         1     0.8658536585365854
    5         1         1     0.9180445722501798
    5         2         1     0.9200571020699501
    5         3         1     0.967032967032967
    5         4         1     0.94337899543379
    5         0        0.1    0.870157819225251
    5         1        0.1    0.918763479511143
    5         2        0.1    0.9200571020699501
    5         3        0.1    0.967032967032967
    5         4        0.1    0.9506849315068493
    5         0        0.01   0.8687230989956959
    5         1        0.01   0.9166067577282531
    5         2        0.01   0.9250535331905781
    5         3        0.01   0.967948717948718
    5         4        0.01   0.9625570776255707
    5         0       0.001   0.8565279770444764
    5         1       0.001   0.9115744069015097
    5         2       0.001   0.9293361884368309
    5         3       0.001   0.967032967032967
    5         4       0.001   0.9634703196347032
'''    

print('{:^10}{:^10}{:^10}{:^10}'.format('n','k','lr','acc'))
for n in range(2,6):
    group_kfold = GroupKFold(n_splits=n)
    for c in [1,1.5]:
        for k,(train_set,test_set) in enumerate(group_kfold.split(train_standardscal,train_label,groups)):
            model = SVC(C = c,kernel = 'rbf',gamma = 0.0023,decision_function_shape='ovr')
            model.fit(train_standardscal[train_set],train_label[train_set])
            acc = model.score(train_standardscal[test_set],train_label[test_set])
            print('{:^10}{:^10}{:^10}{:^10}'.format(n,k,str(c),acc))
            
'''

'''

group_kfold = GroupKFold(n_splits=5)
for k,(train_set,test_set) in enumerate(group_kfold.split(train_standardscal,train_label,groups)):
    model = SVC(C = 0.01,kernel = 'linear',decision_function_shape='ovo')
    model.fit(train_standardscal[train_set],train_label[train_set])
    test_pred = model.predict(test_standardscal)
    train_pred = model.predict(train_standardscal[test_set])
    acc = compute_accuracy(train_label[test_set],train_pred)
#    with open('/Users/yen/Documents/STAT613/kaggle_composition/group_kfold_svm/'+acc+'_k_predict', 'x') as f :
#        f.write('ID,Prediction\n')
#        for i in range(len(test_pred)) :
#            f.write("".join([str(i+1),',',str(test_pred[i]),'\n']))
    print(acc)

SVC(C = c,kernel = 'rbf',gamma = 0.0023,decision_function_shape='ovr')

model = SVC(C = 0.01,kernel = 'linear',decision_function_shape='ovo')
model.fit(train_standardscal,train_label)

model_linear = LinearSVC(penalty = 'l2',C = 0.03)
model_linear.fit(train,train_label)

y_pred = model.predict(test_standardscal)

with open('/Users/yen/Documents/STAT613/kaggle_composition/svm_standard_scaled_X.csv', 'x') as f :
        f.write('ID,Prediction\n')
        for i in range(len(y_pred)) :
            f.write("".join([str(i+1),',',str(y_pred[i]),'\n']))