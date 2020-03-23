
"""
DataCamp

Fine tuning Deep Learning Models

"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Import necessary modules
import tensorflow as tf
import keras
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

print('Current working dir:', os.getcwd())

#%%
auto_df = pd.read_csv('data/auto-mpg.csv')

auto_df.replace('?', np.nan, inplace=True)
auto_df.horsepower = auto_df.horsepower.astype(float)

# drop missing values
auto_df = auto_df.dropna(axis=0, how='any')

y = auto_df['mpg']
X = auto_df.select_dtypes(exclude='object').drop('mpg', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#%%

#                                   Regression
# =============================================================================

# Save the number of columns in X: n_cols
n_cols = X_train.values.shape[1]

# Set up the model: model
model = Sequential()
# Add the first layer
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
# Add the output layer
model.add(Dense(1))


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# model.compile(optimizer=SGD(), loss='mean_squared_error')

# define early stopping params
early_stop = EarlyStopping(patience=5, verbose=2)

# Fit the model
model.fit(X_train.values,
          y_train,
          validation_split=0.3,
          epochs=100,
          callbacks=[early_stop],
          )

# Summary
# =============================================================================
model.summary()

y_pred = model.predict(X_test)

print('RMSE:', np.sqrt(MSE(y_test, y_pred)))




#%%

# =============================================================================

# SGD params

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
input_shape = (n_cols,)

def get_new_model(input_shape = input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape = input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return(model)

# Import the SGD optimizer
from keras.optimizers import SGD

# Create list of learning rates: lr_to_test
lr_to_test = [0.001, 0.01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )

    # Build new model to test, unaffected by previous models
    model = get_new_model()

    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)

    # Compile the model
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy')

    # Fit the model
    model.fit(X_train, y_train)
#%%

# =============================================================================

# Evaluating model accuracy on validation dataset

# =============================================================================


# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
              )

# Fit the model
hist = model.fit(X_train,
                 y_train,
                 validation_split=0.3,
                 epochs=50,
                 )

#%%

# =============================================================================

# Early stopping: Optimizing the optimization
#
# =============================================================================

# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = X_train.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=5)

# Fit the model
model.fit(X_train,
          y_train,
          epochs=150,
          validation_split=0.3,
          callbacks=[early_stopping_monitor]
          )




#%%

# =============================================================================

# Adding Layers to Network

# =============================================================================

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=4)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(100, activation='relu'))
# Add the output layer
model_2.add(Dense(2, activation='softmax'))
# Compile model_2
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model.fit(X, 
                             y, 
                             epochs=75, 
                             validation_split=0.2, 
                             callbacks=[early_stopping_monitor], 
                             verbose=False)

# Fit model_2
model_2_training = model_2.fit(X, 
                               y, 
                               epochs=75, 
                               validation_split=0.2, 
                               callbacks=[early_stopping_monitor], 
                               verbose=False)


print('Model 1 accuracy:', model_1_training.history['val_accuracy'][-1])
print('Model 2 accuracy:', model_2_training.history['val_accuracy'][-1])
# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.legend(('Model 1', 'Model 2'))
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()



#%%

"""
--- Model Capacity (Network Capacity) ---

Closely related to overfitting and underfitting. This is a key consideration
for finding which models to try next.


To increase capacity:
    - add more layers
    - add more nodes to each or one layer
    
Higher Capacity tends to overfit
Lower Capacity tends to underfit



--- Workfow for Optimizing Model Capacity ---

1. Start with a small network
2. Gradually increase capacity and check validation score
3. Add capacity until val-score stops improving
    - increase either layers or nodes
4. If val-score decrease, then reduce capacity slightly to fine-tune

"""


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
from keras.optimizers import Adam

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

print("Accuracy:", model_trained.history['val_accuracy'][-1])





