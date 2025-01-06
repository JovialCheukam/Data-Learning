#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Valentin Emiya, AMU & CNRS LIS
"""
import numpy as np


def normalize_dictionary(X):
    """
    Normalize matrix to have unit l2-norm columns

    Parameters
    ----------
    X : np.ndarray [n, d]
        Matrix to be normalized

    Returns
    -------
    X_normalized : np.ndarray [n, d]
        Normalized matrix
    norm_coefs : np.ndarray [d]
        Normalization coefficients (i.e., l2-norm of each column of ``X``)
    """
    # Check arguments
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2

    alpha = np.linalg.norm(X,axis =0)
    
    return np.true_divide(X,alpha),alpha


def ridge_regression(X, y, lambda_ridge):
    """
    Ridge regression estimation

    Minimize $\left\| X w - y\right\|_2^2 + \lambda \left\|w\right\|_2^2$
    with respect to vector $w$, for $\lambda > 0$ given a matrix $X$ and a
    vector $y$.

    Note that no constant term is added.

    Parameters
    ----------
    X : np.ndarray [n, d]
        Data matrix composed of ``n`` training examples in dimension ``d``
    y : np.ndarray [n]
        Labels of the ``n`` training examples
    lambda_ridge : float
        Non-negative penalty coefficient

    Returns
    -------
    w : np.ndarray [d]
        Estimated weight vector
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    I = np.eye(X.shape[1])
    return (np.linalg.inv(X.T @ X + lambda_ridge*I) @ X.T) @ y


def mp(X, y, n_iter):
    """
    Matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    r = y
    w = np.zeros((X.shape[1],))
    c = np.zeros((X.shape[1],))
    error_norm = np.zeros((n_iter+1,))
    
    for k in range(n_iter):
        
        error_norm[k] = np.linalg.norm(r) 
        for m in range(X.shape[1]):
            c[m] = np.vdot(X[:,m],r)
        m_chap = np.argmax(np.abs(c))
        w[m_chap] = w[m_chap] + c[m_chap]
        r = r - c[m_chap]*X[:,m_chap]
    return w, error_norm   

def omp(X, y, n_iter):
    """
    Orthogonal matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    r = y
    w = np.zeros((X.shape[1],))
    c = np.zeros((X.shape[1],))
    omega = []
    error_norm = np.zeros((n_iter+1,))
    
    for k in range(n_iter):
        
        error_norm[k] = np.linalg.norm(r) 
        for m in range(X.shape[1]):
            c[m] = np.vdot(X[:,m],r)
        m_chap = np.argmax(np.abs(c))
        omega.append(m_chap)
        w[omega] = np.linalg.pinv(X[:,omega]) @ y
        r = y - X @ w
    return w, error_norm  


def ols(X, y, n_iter):
    """
    Orthogonal matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    r = y
    w = np.zeros((X.shape[1],))
    
    omega = []
    
    error_norm = np.zeros((n_iter+1,))
    
    for k in range(n_iter):
        
        error_norm[k] = np.linalg.norm(r)
        norm_err1 = []
        norm_err2 = []
        for m in range(X.shape[1]):
            if m not in omega:
               omega.append(m)
               u = np.linalg.pinv(X[:,omega]) @ y
               norm_err1.append((np.linalg.norm(y - X[:,omega] @ u),m))
               omega.remove(m)
        for i,j in norm_err1:
            norm_err2.append(i)
        omega.append(norm_err1[np.argmin(norm_err2)][1])      
        w[omega] = np.linalg.pinv(X[:,omega]) @ y
        r = y - X @ w
    return w, error_norm  
