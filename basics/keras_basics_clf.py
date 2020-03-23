# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:32:44 2020

@author: smouz

"""
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

print('Current working dir:', os.getcwd())

#%%
#                               Classification
# =============================================================================

from keras.utils import to_categorical


titanic_df = pd.read_csv("data/titanic.csv")


# Convert the target to categorical: target
y = to_categorical(titanic_df.survived)
X = titanic_df.drop('survived', axis=1)

# check for missing values
assert ~titanic_df.isnull().sum().any(), 'Contains missing values'


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_cols = X_train.shape[1]

#%%

# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(50, activation='relu', input_shape = (n_cols,)))
model.add(Dense(50, activation='relu',))
model.add(Dense(2, activation='softmax'))
# compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fit to traing data
model.fit(X_train, y_train, validation_split=0.2, epochs=100)

# Calculate predictions: predictions
predictions = model.predict(X_test)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:, 1]

# print predicted_prob_true
# print(predicted_prob_true)

#%%

