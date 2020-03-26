# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:50:33 2020

@author: smouz

"""


import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split


#%%
# =============================================================================
# Batch Normalization Effects
# =============================================================================

X = np.load('data/digits_pixels.npy')
y = np.load('data/digits_target.npy')

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

INPUT_SHAPE = X_train.shape[1]
OPTIMIZER = 'adam'

#%%
def compare_histories_acc(h1,h2):
  plt.plot(h1.history['accuracy'])
  plt.plot(h1.history['val_accuracy'])
  plt.plot(h2.history['accuracy'])
  plt.plot(h2.history['val_accuracy'])
  plt.title("Batch Normalization Effects")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Test',
              'Train w/ Batch Normalization',
              'Test w/ Batch Normalization'],
             loc='best')
  plt.show()


def standard_model():
    model = Sequential()
    model.add(Dense(50, input_shape = (INPUT_SHAPE,), activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    # output layer
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer = OPTIMIZER,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    return model

#%%
EPOCHS = 8

standard_model = standard_model()

# Build your deep network
batchnorm_model = Sequential()
batchnorm_model.add(Dense(50, input_shape=(INPUT_SHAPE,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
batchnorm_model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

#### Fit Both model
# Train your standard model, storing its history
history1 = standard_model.fit(X_train,
                              y_train,
                              validation_data=(X_test,y_test),
                              epochs=EPOCHS,
                              verbose=0)

# Train the batch normalized model you recently built, store its history
history2 = batchnorm_model.fit(X_train,
                               y_train,
                               validation_data=(X_test, y_test),
                               epochs=EPOCHS, verbose=0)

### Plot Comparison
# Call compare_acc_histories passing in both model histories
compare_histories_acc(history1, history2)
