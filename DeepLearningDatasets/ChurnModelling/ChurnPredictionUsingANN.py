#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:02:01 2020

@author: evkikum

ANN using Classification approach

"""



import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

tf.__version__

os.chdir(r"/home/evkikum/Desktop/Datascience/Python/DeepLearning/DeepLearning/DeepLearningDatasets/ChurnModelling")

dataset = pd.read_csv("Churn_Modelling.csv")
dataset.info()

df = dataset.iloc[:, 3:-1]
df.info()
df["Gender"].value_counts()
df["Geography"].value_counts()
df["Gender"] = df["Gender"].apply(lambda x: 1 if x == "Female" else 1)
df["Geography"] = df["Geography"].astype('category')

df = pd.get_dummies(df)
df.info()
df_stats = df.describe()

X = df
y = dataset.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

X_train = scale(X_train)
X_test = scale(X_test)

output_df = pd.DataFrame(index = ["Train Score", "Test Score"])

## Using Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 81.2 % 
model.score(X_test, y_test)  ## 81.4 %%
output_df["LR"] = [model.score(X_train, y_train), model.score(X_test, y_test)]

## Using KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 87 % 
model.score(X_test, y_test)   ## 83 %
output_df["KNN"] = [model.score(X_train, y_train), model.score(X_test, y_test)]


## Using GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 87 % 
model.score(X_test, y_test)   ## 86 %
output_df["GB"] = [model.score(X_train, y_train), model.score(X_test, y_test)]

## Using RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_train, y_train)  ## 100 % 
model.score(X_test, y_test)   ## 86 %
output_df["RF"] = [model.score(X_train, y_train), model.score(X_test, y_test)]

## INitialize the ANN
ann = tf.keras.models.Sequential()

## Adding the input layer and first hidden layer  
## relu -- Rectifier activation function
ann.add(tf.keras.layers.Dense(units = 6, activation = "relu"))

## Adding the 2nd layer
## relu -- Rectifier activation function
ann.add(tf.keras.layers.Dense(units = 6, activation = "relu"))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

##Compiling the ANN
ann.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

ann.fit(X_train, y_train, batch_size = 32, epochs = 100 )   ## 86.34 %
y_pred = ann.predict(X_test)

y_pred = (y_pred > 0.5)
##print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


output_df["ANN"] = [np.NAN, accuracy_score(y_test, y_pred)]