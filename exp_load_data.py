import numpy as np
from data_utils import load_data


""" Load training data and print dimensions as well as a few coefficients
in the first and last places and at random locations.
"""
X,y,Xu = load_data('data/YearPredictionMSD_100.npz')
print(y.shape)
