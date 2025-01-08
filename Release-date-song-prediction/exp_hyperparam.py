#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 05:39:54 2019

@author: jovial
"""

import matplotlib.pyplot as plt
import numpy as np
from algorithms import normalize_dictionary
from data_utils import load_data,split_data
from linear_regression import LinearRegressionRidge,LinearRegressionMp,LinearRegressionOmp,\
LinearRegressionOls

X_labeled,y_labeled,X_unlabeled = load_data('data/YearPredictionMSD_100.npz')

X_train,y_train,X_Test,y_Test = split_data(X_labeled,y_labeled,2/3)

X = normalize_dictionary(X_train[:500,:])[0]
y = y_train[:500]


X_Train,y_Train,X_test,y_test = split_data(X,y,2/3)



lamb = np.array([np.arange(0.1,0.65,0.01)]).T

k_max = np.array([np.arange(1,56,1)]).T


train_error = np.zeros((len(lamb),4))
valid_error = np.zeros((len(lamb),4))


j = -1
for i in lamb:
    j = j+1
   
    clf = LinearRegressionRidge(i[0])
    
    
    clf.fit(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,0] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,0] = err_train
    
j = -1
for i in k_max:
    j = j+1
    
    
    clf = LinearRegressionMp(i[0])
  
    
    clf.fit(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,1] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,1] = err_train   
    
    

    
j = -1
for i in k_max:
    j = j+1
    
    clf = LinearRegressionOmp(i[0])
    
    
    clf.fit(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,2] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,2] = err_train       
    


j = -1
for i in k_max:
    j = j+1
    
    clf = LinearRegressionOls(i[0])
    
    
    clf.fit(X_Train,y_Train)
    
    
    y_pred_test = clf.predict(X_test)
    
    err_valid = np.sqrt(((y_pred_test-y_test)**2).sum()/len(y_test))
    valid_error[j,3] = err_valid
    
    y_pred_train = clf.predict(X_Train)
    
    err_train = np.sqrt(((y_pred_train-y_Train)**2).sum()/len(y_Train))
    train_error[j,3] = err_train       
    



plt.figure(1)
#axes = plt.gca()
 
#axes.set_ylim(8,23)

#axes.set_xlim(0,900)

plt.plot(k_max,valid_error[:,2],color="blue",label="err_valid_Omp")
 
plt.plot(k_max,train_error[:,2],"b--",label="err_train_Omp")
plt.title("Erreurs d'apprentissage et de validation par valeur d'hyperparametre")  

plt.legend(loc= "best")

plt.xlabel("valeur de l'hyperparametre k_max")

plt.ylabel("Erreurs")

plt.savefig("erreur4.png") 
#############################################################
plt.figure(2)

plt.plot(lamb,valid_error[:,0],color="yellow",label="err_valid_Ridge")
 
plt.plot(lamb,train_error[:,0],"y--",label="err_train_Ridge")
plt.title("Erreurs d'apprentissage et de validation par valeur d'hyperparametre")  

plt.legend(loc= "best")

plt.xlabel("valeur de l'hyperparametre lambda")

plt.ylabel("Erreurs")

plt.savefig("erreur5.png") 
############################################################

plt.figure(3)
plt.plot(k_max,valid_error[:,1],color="red",label="err_valid_Mp")
 
plt.plot(k_max,train_error[:,1],"r--",label="err_train_Mp")

plt.title("Erreurs d'apprentissage et de validation par valeur d'hyperparametre")  

plt.legend(loc= "best")

plt.xlabel("valeur de l'hyperparametre k_max")

plt.ylabel("Erreurs")

plt.savefig("erreur6.png")
###############################################################

plt.figure(4)
plt.plot(k_max,valid_error[:,3],color="black",label="err_valid_Ols")
 
plt.plot(k_max,train_error[:,3],"k--",label="err_train_Ols")
 

plt.title("Erreurs d'apprentissage et de validation par valeur d'hyperparametre")  

plt.legend(loc= "best")

plt.xlabel("valeur de l'hyperparametre")

plt.ylabel("Erreurs")

plt.savefig("erreur7.png")

