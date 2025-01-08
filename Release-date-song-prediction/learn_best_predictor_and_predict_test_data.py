#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:17:59 2019

@author: jovial
"""

from learn_all_with_ALGO import learn_all_with_Ridge
from data_utils import load_data, split_data
import numpy as np


def learn_best_predictor_and_predict_test_data():
    X_labeled,y_labeled,X_unlabeled = load_data('data/YearPredictionMSD_100.npz')

    X_train,y_train,X_test,y_Test = split_data(X_labeled[:750,:],y_labeled[:750],2/3)

    clf = learn_all_with_Ridge(X_train,y_train)
    
    y_pred = clf.predict(X_test)
    
    error_quadratiq = ((y_pred - y_Test)**2).sum()/len(y_Test)
    
    print("The score of that the predictor is %f" % (1-np.sqrt(error_quadratiq)/sum(y_Test)))

    y_test = clf.predict(X_unlabeled)
    
    np.savez("test_prediction_results.npz",y_test=y_test)
###############################################################
###############################################################
############   ESSAI DE LA FONCTION ###########################    
learn_best_predictor_and_predict_test_data()
print(np.load("test_prediction_results.npz")["y_test"][:50])
    
    
    
    
    
    