#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:17:30 2019

@author: jovial
"""
import time
import numpy as np
from data_utils import load_data, split_data
from linear_regression import LinearRegressionMean,\
LinearRegressionMedian,LinearRegressionMajority,\
LinearRegressionLeastSquares


X_labeled,y_labeled,X_unlabeled = load_data('data/YearPredictionMSD_100.npz')

X_train,y_train,X_test,y_test = split_data(X_labeled,y_labeled,2/3)

N_rond = np.power(2,np.array([[5,6,7,8,9,10,11]]).T)

train_error = np.zeros((len(N_rond),4))
valid_error = np.zeros((len(N_rond),4))
learning_time = np.zeros((len(N_rond),4))

j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0],:]
    y_Train = y_train[:i[0]]
    
    clf = LinearRegressionMean()
    
    t0 = time.perf_counter()
    clf.fit(X_Train,y_Train)
    t1 = time.perf_counter()-t0
    
    learning_time[j,0]=t1
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,0] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,0] = err_train
    
    
j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0],:]
    y_Train = y_train[:i[0]]
    
    clf = LinearRegressionMedian()
  
    t0 = time.perf_counter()
    clf.fit(X_Train,y_Train)
    t1 = time.perf_counter()-t0
    
    learning_time[j,1]=t1
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,1] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,1] = err_train   
    
    
j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0],:]
    y_Train = y_train[:i[0]]
    
    clf = LinearRegressionLeastSquares()
    
    t0 = time.perf_counter()
    clf.fit(X_Train,y_Train)
    t1 = time.perf_counter()-t0
    
    learning_time[j,3]=t1
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,3] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,3] = err_train   
    
    
j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0],:]
    y_Train = y_train[:i[0]]
    
    clf = LinearRegressionMajority()
    
    t0 = time.perf_counter()
    clf.fit(X_Train,y_Train)
    t1 = time.perf_counter()-t0
    
    learning_time[j,2]=t1
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,2] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,2] = err_train       
    
np.savez("perform_estim_const.npz",N_rond=N_rond,valid_error=valid_error,train_error=train_error,learning_time=learning_time)    
    
    
    
    
    
    
    
    