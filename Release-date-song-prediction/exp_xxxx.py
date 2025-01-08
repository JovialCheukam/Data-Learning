#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 03:43:23 2019

@author: jovial
"""

import matplotlib.pyplot as plt
import numpy as np
from algorithms import ols,omp,mp,normalize_dictionary
from data_utils import load_data,split_data
from linear_regression import LinearRegressionLeastSquares,\
LinearRegressionRidge,LinearRegressionMp,LinearRegressionOmp,\
LinearRegressionOls

X_labeled,y_labeled,X_unlabeled = load_data('data/YearPredictionMSD_100.npz')

X_train,y_train,X_test,y_test = split_data(X_labeled,y_labeled,2/3)

X = normalize_dictionary(X_train[:500,:])[0]
y = y_train[:500]


k_max = 20

k_iter = range(k_max)

wMp,errMp = mp(X,y,k_max)

wOmp,errOmp = omp(X,y,k_max)

wOls,errOls = ols(X,y,k_max)

plt.figure(1)

plt.scatter(k_iter,errMp[:k_max],color="red",label="Modele_MP")
 
plt.scatter(k_iter,errOmp[:k_max],color="blue",label="Modele_OMP")
 
plt.scatter(k_iter,errOls[:k_max],color="black",label="Modele_OLS")
 
plt.legend(loc= "best")

plt.title("Evolution des residus en fonction des iterations")
plt.xlabel("rangs des iterations")

plt.ylabel("Residus")

plt.savefig("residus.png")

##############################################################
##############################################################






N_rond = np.power(2,np.array([[5,6,7,8,9]]).T)

train_error = np.zeros((len(N_rond),5))
valid_error = np.zeros((len(N_rond),5))


j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0]-12,:]
    y_Train = y_train[:i[0]-12]
    
    clf = LinearRegressionRidge(0.5)
    
    
    clf.fit(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,0] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,0] = err_train
    
    
j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0]-12,:]
    y_Train = y_train[:i[0]-12]
    
    clf = LinearRegressionMp(20)
  
    
    clf.fit(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,1] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,1] = err_train   
    
    
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
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,3] = err_train   
    
    
j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0]-12,:]
    y_Train = y_train[:i[0]-12]
    
    clf = LinearRegressionOmp(20)
    
    
    clf.fit(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,2] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,2] = err_train       
    


j = -1
for i in N_rond:
    j = j+1
    X_Train = X_train[:i[0]-12,:]
    y_Train = y_train[:i[0]-12]
    
    clf = LinearRegressionOls(20)
    
    
    clf.fit(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,4] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,4] = err_train       
    



plt.figure(2)
axes = plt.gca()
 
axes.set_ylim(8,23)

axes.set_xlim(0,900)

plt.plot(N_rond,valid_error[:,2],color="blue",label="err_valid_Omp")
 
plt.plot(N_rond,train_error[:,2],"b--",label="err_train_Omp")
 

plt.plot(N_rond,valid_error[:,0],color="yellow",label="err_valid_Ridge")
 
plt.plot(N_rond,train_error[:,0],"y--",label="err_train_Ridge")
 

plt.plot(N_rond,valid_error[:,1],color="red",label="err_valid_Mp")
 
plt.plot(N_rond,train_error[:,1],"r--",label="err_train_Mp")
 

plt.plot(N_rond,valid_error[:,3],color="green",label="err_valid_LeastSQ")
 
plt.plot(N_rond,train_error[:,3],"g--",label="err_train_LeastSQ")
 

plt.plot(N_rond,valid_error[:,4],color="black",label="err_valid_Ols")
 
plt.plot(N_rond,train_error[:,4],"k--",label="err_train_Ols")
 

plt.title("Erreurs d'apprentissage et de validation par taille de donnees")  

plt.legend(loc= "best")

plt.xlabel("taille des donnees d'apprentissage")

plt.ylabel("Erreurs")

plt.savefig("erreur3.png")















