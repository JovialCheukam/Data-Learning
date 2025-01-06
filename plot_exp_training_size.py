#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:14:59 2019

@author: jovial
"""
import numpy as np
import matplotlib.pyplot as plt

results = np.load("perform_estim_const.npz")


plt.figure(1)
axes = plt.gca()
 
axes.set_ylim(8,30)

axes.set_xlim(0,3250)
 
plt.plot(results["N_rond"],results["valid_error"][:,0],color="red",label="err_valid_mean")
 
plt.plot(results["N_rond"],results["train_error"][:,0],"r--",label="err_train_mean")
 
 
plt.plot(results["N_rond"],results["valid_error"][:,1],color="blue",label="err_valid_med")
 
plt.plot(results["N_rond"],results["train_error"][:,1],"b--",label="err_train_med")
 
 
plt.plot(results["N_rond"],results["valid_error"][:,2],color="yellow",label="err_valid_maj")
 
plt.plot(results["N_rond"],results["train_error"][:,2],"y--",label="err_train_maj")
  
 
plt.plot(results["N_rond"],results["valid_error"][:,3],color="green",label="err_valid_med")
 
plt.plot(results["N_rond"],results["train_error"][:,3],"g--",label="err_train_med")

plt.title("Erreurs d'apprentissage et de validation par taille de donnees")  

plt.legend(loc= "best")

plt.xlabel("taille des donnees d'apprentissage")

plt.ylabel("Erreurs")

plt.savefig("erreur1.png")



plt.figure(2)
axes = plt.gca()
 
#axes.set_ylim(8,30)

#axes.set_xlim(0,3250)
 
plt.semilogy(results["N_rond"],results["learning_time"][:,0],color="red",label="Model_mean")
 
plt.semilogy(results["N_rond"],results["learning_time"][:,1],color="blue",label="Model_median")
 
 
plt.semilogy(results["N_rond"],results["learning_time"][:,2],color="yellow",label="Model_maj")
 
plt.semilogy(results["N_rond"],results["learning_time"][:,3],color="green",label="Model_leastSq")
 
plt.title("Temps d'apprentissage par taille de donnees")  

plt.legend(loc= "best")

plt.xlabel("taille des donnees d'apprentissage")

plt.ylabel("Temps")

plt.savefig("erreur2.png")