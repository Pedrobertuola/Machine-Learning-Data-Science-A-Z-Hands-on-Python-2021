#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR Hyper Parameter Tuning
"""

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Boston_P = load_boston()

x = Boston_P.data
y = Boston_P.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, train_size=0.75,random_state=76)

from sklearn.preprocessing import MinMaxScaler

Sc = MinMaxScaler(feature_range= (0,1))
x_train = Sc.fit_transform(x_train)
x_test = Sc.fit_transform(x_test)

y_train = y_train.reshape(-1,1)
y_train = Sc.fit_transform(y_train)

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ['rbf','linear'], 
              'gamma':  [1,0.1,0.01]}

grid = GridSearchCV(SVR(), parameters, refit = True,verbose=2, scoring='neg_mean_squared_error')

grid.fit(x, y)

best_params = grid.best_params_