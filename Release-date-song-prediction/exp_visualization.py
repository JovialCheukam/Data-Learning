#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Valentin Emiya, AMU & CNRS LIS
"""
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data


""" Build the histogram of the years of the songs from the training set and
export the figure to the image file hist_train.png
"""
plt.hist(load_data('data/YearPredictionMSD_100.npz')[1])
plt.savefig('Histogramme.png')