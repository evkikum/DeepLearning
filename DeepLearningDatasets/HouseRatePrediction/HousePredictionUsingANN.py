#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:07:29 2020

@author: evkikum

ANN using Regression approach
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split


tf.__version__

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/DeepLearning/DeepLearning/DeepLearningDatasets/HouseRatePrediction")

dataset = pd.read_excel('Folds5x2_pp.xlsx')
dataset.info()
dataset.describe()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:,-1]

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

ann = tf.keras.models.Sequential()

## Add the input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation="relu"))
## Add the 2nd hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation="relu"))

## Add output later
ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer='adam', loss = 'mean_squared_error')
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)


pred = ann.predict(X)
dataset["pred"] = pred
dataset["diff"] = abs((dataset["pred"] - dataset["PE"])*100/dataset["pred"])



