import numpy as np


def load_data(filename):
    """
    Load data examples from a npz file

    Parameters
    ----------
    filename : str
        Name of the npz file containing the data to be loaded

    Returns
    -------
    X_labeled : np.ndarray [n, d]
        Array of n feature vectors with size d
    y_labeled : np.ndarray [n]
        Vector of n labels related to the n feature vectors
    X_unlabeled :

    """
    data = np.load(filename)
    
    
    return np.array(data["X_labeled"]),np.array(data["y_labeled"]),np.array(data["X_unlabeled"])


def randomize_data(X, y):
    """
    Randomly permute the examples in the labeled set (X, y), i.e. the rows
    of X and the elements of y, simultaneously.

    Parameters
    ----------
    X : np.ndarray [n, d]
        Array of n feature vectors with size d
    y : np.ndarray [n]
        Vector of n labels related to the n feature vectors

    Returns
    -------
    Xr : np.ndarray [n, d]
        Permuted version of X
    yr : np.ndarray [n]
        Permuted version of y

    Raises
    ------
    ValueError
        If the number of rows in X differs from the number of elements in y.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError('Number of rows in X ({}) differs from the number of '
                         'elements in y'
                         .format(X.shape[0], y.shape[0]))
    X_plus = np.c_[X,y.reshape((y.shape[0],1))]
    X_permut = np.random.permutation(X_plus)
    return X_permut[:,:X.shape[1]],X_permut[:,X.shape[1]]

def split_data(X, y, ratio):
    """
    Split a set of n labeled examples into two subsets as a random partition.

    split_data(X, y, ratio) returns a tuple (X1, y1, X2, y2). The n input
    labeled examples (X,y) are randomly permuted and split as a partition
    {(X1, y1), (X2, y2)}. The respective size n1 and n2 is such that
    n1/n approximately equals the input argument `ratio` and n1+n2 = n.

    Parameters
    ----------
    X : np.ndarray [n, d]
        Array of n feature vectors with size d
    y : np.ndarray [n]
        Vector of n labels related to the n feature vectors
    ratio : float
        Ratio of data to be extracted into (X1, y1)

    Returns
    -------
    X1 : np.ndarray [n1, d]
        Array of n1 feature vectors
    y1 : np.ndarray [n1]
        Vector of n1 label
    X2 : np.ndarray [n2, d]
        Array of n2 feature vectors
    y2 : np.ndarray [n2]
        Vector of n2 labels selected

    """
    Xr,yr = randomize_data(X,y)
    nb_lines = int(ratio*(X.shape[0]))
    X1 = Xr[:nb_lines,:]
    y1 = yr[:nb_lines]
    X2 = Xr[nb_lines:,:]
    y2 = yr[nb_lines:]
    return X1,y1,X2,y2
    
    
    
    
    
