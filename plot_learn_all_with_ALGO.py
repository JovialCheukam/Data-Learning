#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:21:52 2019

@author: jovial
"""

import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_data,split_data
from linear_regression import LinearRegressionLeastSquares
from learn_all_with_ALGO import learn_all_with_Ols,\
learn_all_with_Omp,learn_all_with_Mp,learn_all_with_Ridge

X_labeled,y_labeled,X_unlabeled = load_data('data/YearPredictionMSD_100.npz')

X_train,y_train,X_test,y_test = split_data(X_labeled[:750,:],y_labeled[:750],2/3)




N_rond = np.power(2,np.array([[5,6,7,8,9]]).T)-12

valid_error = np.zeros((len(N_rond),5))


j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0],:]
    y_Train = y_train[:i[0]]
    
    clf = learn_all_with_Ridge(X_Train,y_Train) 
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,0] = err_valid
    
   
    
    
j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0]-12,:]
    y_Train = y_train[:i[0]-12]
    
    clf = learn_all_with_Mp(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,1] = err_valid
    
     
    
    
j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0]-12,:]
    y_Train = y_train[:i[0]-12]
    
    clf = LinearRegressionLeastSquares()
    
    
    clf.fit(X_Train,y_Train)
    
    
    
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,3] = err_valid
    
    
    
    
j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0]-12,:]
    y_Train = y_train[:i[0]-12]
    
    clf = learn_all_with_Omp(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,2] = err_valid
    
          
    


j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0]-12,:]
    y_Train = y_train[:i[0]-12]
    
    clf = learn_all_with_Ols(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,4] = err_valid
    
        
    




axes = plt.gca()
 
axes.set_ylim(14,25)

axes.set_xlim(0,800)

plt.plot(N_rond,valid_error[:,2],color="blue",label="err_valid_Omp")
 

plt.plot(N_rond,valid_error[:,0],color="yellow",label="err_valid_Ridge")
 

plt.plot(N_rond,valid_error[:,1],color="red",label="err_valid_Mp")
 

plt.plot(N_rond,valid_error[:,3],color="green",label="err_valid_LeastSQ")
 

plt.plot(N_rond,valid_error[:,4],color="black",label="err_valid_Ols")
 

plt.title("Erreurs d'apprentissage et de validation par taille de donnees")  

plt.legend(loc= "best")

plt.xlabel("taille des donnees d'apprentissage")

plt.ylabel("Erreurs")

plt.savefig("erreur8.png")


