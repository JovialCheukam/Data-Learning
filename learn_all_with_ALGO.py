#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 08:51:12 2019

@author: jovial
"""

from data_utils import split_data
import numpy as np
from linear_regression import LinearRegressionRidge,LinearRegressionMp,LinearRegressionOmp,\
LinearRegressionOls




def learn_all_with_Ridge(X,y):
    X_train,y_train,X_test,y_test = split_data(X,y,2/3)

    lamb = np.array([np.arange(0.1,0.32,0.01)]).T
    valid_error = np.zeros((len(lamb),1))
    
    j = -1
    for i in lamb:
        j = j+1
   
        clf = LinearRegressionRidge(i[0])
    
    
        clf.fit(X_train,y_train)
    
    
        y_pred_test = clf.predict(X_test)
    
        err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
        valid_error[j,0] = err_valid
    
    
    clf1 = LinearRegressionRidge(lamb[np.argmin(valid_error),0])
    clf1.fit(X_train,y_train)
    
    return clf1
    
    
    
def learn_all_with_Mp(X,y):
    X_train,y_train,X_test,y_test = split_data(X,y,2/3)

    k_max = np.array([np.arange(1,23,1)]).T
    valid_error = np.zeros((len(k_max),1))
    
    j = -1
    for i in k_max:
        j = j+1
   
        clf = LinearRegressionMp(i[0])
    
    
        clf.fit(X_train,y_train)
    
    
        y_pred_test = clf.predict(X_test)
    
        err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
        valid_error[j,0] = err_valid
    
    
    clf1 = LinearRegressionMp(k_max[np.argmin(valid_error),0])
    clf1.fit(X_train,y_train)
    
    return clf1    
    
    
    
def learn_all_with_Omp(X,y):
    X_train,y_train,X_test,y_test = split_data(X,y,2/3)

    k_max = np.array([np.arange(1,23,1)]).T
    valid_error = np.zeros((len(k_max),1))
    
    j = -1
    for i in k_max:
        j = j+1
   
        clf = LinearRegressionOmp(i[0])
    
    
        clf.fit(X_train,y_train)
    
    
        y_pred_test = clf.predict(X_test)
    
        err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
        valid_error[j,0] = err_valid
    
    
    clf1 = LinearRegressionOmp(k_max[np.argmin(valid_error),0])
    clf1.fit(X_train,y_train)
    
    return clf1      
    
    
    
    
def learn_all_with_Ols(X,y):
    X_train,y_train,X_test,y_test = split_data(X,y,2/3)

    k_max = np.array([np.arange(1,23,1)]).T
    valid_error = np.zeros((len(k_max),1))
    
    j = -1
    for i in k_max:
        j = j+1
   
        clf = LinearRegressionOls(i[0])
    
    
        clf.fit(X_train,y_train)
    
    
        y_pred_test = clf.predict(X_test)
    
        err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
        valid_error[j,0] = err_valid
    
    
    clf1 = LinearRegressionOls(k_max[np.argmin(valid_error),0])
    clf1.fit(X_train,y_train)
    
    return clf1      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   