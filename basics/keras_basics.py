# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:22:29 2019

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
auto_df = pd.read_csv('data/auto-mpg.csv')

auto_df.replace('?', np.nan, inplace=True)
auto_df.horsepower = auto_df.horsepower.astype(float)

# drop missing values
auto_df = auto_df.dropna(axis=0, how='any')

target = 'mpg'

y = auto_df[target]

X = auto_df.select_dtypes(exclude='object').drop(target, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




#%%

#                                   Regression
# =============================================================================

# Save the number of columns in predictors: n_cols
n_cols = X_train.values.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# model.compile(optimizer=SGD(), loss='mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

# Fit the model
model.fit(X_train.values, y_train, validation_split=0.2, epochs=100)


#%%
#
# =============================================================================
model.summary()

y_pred = model.predict(X_test)

print('RMSE:', np.sqrt(MSE(y_test, y_pred)))



#%%
#                               Classification
# =============================================================================

from keras.utils import to_categorical


titanic_df = pd.read_csv("data/titanic.csv")


# Convert the target to categorical: target
y = to_categorical(titanic_df.survived)
X = titanic_df.drop('survived', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

predictors = X_train

#%%
# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(predictors, y)

# Calculate predictions: predictions
predictions = model.predict(X_test)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:, 1]

# print predicted_prob_true
print(predicted_prob_true)

#%%















