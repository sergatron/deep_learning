# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:59:43 2020

@author: smouz
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Import necessary modules
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

#%%
# =============================================================================

# MNIST digit images 

# =============================================================================

mnist_df = pd.read_csv('data/mnist.csv')
mnist_df.head()

# NOTE: this data has no HEADER
# extract predictors and target 
# X: everything after first column
# y: first column
X = mnist_df.iloc[:, 1:]
y = mnist_df.iloc[:, :1]

# convert target to categorical; there are 10 digits in this case
y = to_categorical(y.values)

#%%

# Train Model


# Create the model: model
model = Sequential()
optimizer = Adam(0.001)

# Add the first hidden layer
model.add(Dense(64, activation='relu', input_shape=(784,)))
# Add the second hidden layer
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))


model.add(Dense(64, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Add the output layer
# there are ten possible digits
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_monitor = EarlyStopping(patience=5, verbose=False)

# Fit the model
model_trained = model.fit(X, 
                          y, 
                          validation_split=0.2,
                          epochs=100,
                          callbacks=[early_stopping_monitor],
                          verbose=False
                          )

print(model.summary())
print("Accuracy:", model_trained.history['val_accuracy'][-1])


