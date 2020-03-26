# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:07:06 2020

@author: smouz
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from kears.optimizer import Adam, SGD
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import RandomizedSearchCV, KFold


from sklearn.model_selection import train_test_split

#%%
def plot_accuracy(acc,val_acc):
  # Plot training & validation accuracy values
  plt.figure()
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()

def plot_results(train_accs,test_accs):
  plt.plot(training_sizes, train_accs, 'o-', label="Training Accuracy")
  plt.plot(training_sizes, test_accs, 'o-', label="Test Accuracy")
  plt.title('Accuracy vs Number of training samples')
  plt.xlabel('# of training samples')
  plt.ylabel('Accuracy')
  plt.legend(loc="best")
  plt.show()


#%%

X = np.load('data/digits_pixels.npy')
y = np.load('data/digits_target.npy')

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train.shape
#%%
# Instantiate a Sequential model
model = Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu
model.add(Dense(16, input_shape = (X_train.shape[1],), activation = 'relu'))

# Output layer with 10 neurons (one per digit) and softmax
model.add(Dense(10, activation='softmax'))

# Compile your model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Test if your model works and can process input data
print(model.predict(X_train).shape)

#%%

# Train your model for 60 epochs, using X_test and y_test as validation data
history = model.fit(X_train,
                    y_train,
                    epochs=60,
                    validation_data=(X_test, y_test),
                    verbose=0)

# Extract from the history object loss and val_loss to plot the learning curve
plot_loss(history.history['loss'], history.history['val_loss'])

#%%

# =============================================================================
# Plot Overfitting
# =============================================================================
training_sizes = [ 125,  502,  879, 1255]
early_stop = EarlyStopping(monitor='val_acc', patience=5)

train_accs = []
test_accs = []
for size in training_sizes:
  	# Get a fraction of training data (we only care about the training data)
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # Reset the model to the initial weights and train it on the new data fraction
    model.set_weights(model.get_weights())
    model.fit(X_train_frac,
              y_train_frac,
              epochs = 50,
              callbacks = [early_stop],
              verbose=False
              )

    # Evaluate and store the train fraction and the complete test set results
    train_accs.append(model.evaluate(X_train, y_train)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])

# Plot train vs test accuracies
plot_results(train_accs, test_accs)


#%%

# =============================================================================
# Activation Functions
# =============================================================================
"""
You will try out different activation functions on the multi-label model
you built for your irrigation machine in chapter 2. The function get_model()
returns a copy of this model and applies the activation function, passed on
as a parameter, to its hidden layer.

"""
# Activation functions to try
activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']

# Loop over the activation functions
activation_results = {}

for act in activations:
  # Get a new model with the current activation
  model = get_model(act)
  # Fit the model
  history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, verbose=0)
  activation_results[act] = history


"""
The code used in the previous exercise has been executed to obtain the
activation_results with the difference that 100 epochs instead of 20 are
used. That way you'll have more epochs to further compare how the training
evolves per activation function.

For every history callback of each activation function in activation_results:

The history.history['val_loss'] has been extracted.
The history.history['val_acc'] has been extracted.
Both are saved in two dictionaries:
    val_loss_per_function and val_acc_per_function.

"""
# Create a dataframe from val_loss_per_function
val_loss= pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()


#%%

# =============================================================================
# Batch Sizes
# =============================================================================

def get_model():
  model = Sequential()
  model.add(Dense(4,input_shape=(2,),activation='relu'))
  model.add(Dense(1,activation="sigmoid"))
  model.compile('sgd', 'binary_crossentropy', metrics=['accuracy'])
  return model

model = get_model()

# Fit your model for 5 epochs with a batch of size the training set
model.fit(X_train, y_train, epochs=5, batch_size=X_train.shape[0])
print("\n The accuracy when using the whole training set as a batch was: ",
      model.evaluate(X_test, y_test)[1])


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

def compare_histories_acc(h1,h2):
  plt.plot(h1.history['accuracy'])
  plt.plot(h1.history['val_accuracy'])
  plt.plot(h2.history['accuracy'])
  plt.plot(h2.history['val_accuracy'])
  plt.title("Batch Normalization Effects")
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Test', 'Train with Batch Normalization', 'Test with Batch Normalization'], loc='best')
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

standard_model = standard_model()

# Import batch normalization from keras layers
from keras.layers import BatchNormalization

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


# Train your standard model, storing its history
history1 = standard_model.fit(X_train,
                              y_train,
                              validation_data=(X_test,y_test),
                              epochs=30,
                              verbose=0)

# Train the batch normalized model you recently built, store its history
history2 = batchnorm_model.fit(X_train,
                               y_train,
                               validation_data=(X_test, y_test),
                               epochs=30, verbose=0)

# Call compare_acc_histories passing in both model histories
compare_histories_acc(history1, history2)


#%%

# =============================================================================
# Hyper-param Search w/Scikit-Learn
# =============================================================================
"""
Let's tune the hyperparameters of a binary classification model that does
well classifying the breast cancer dataset.

"""
# Creates a model given an activation and learning rate
def create_model(learning_rate=0.01, activation='relu'):

  	# Create an Adam optimizer with the given learning rate
  	opt = Adam(lr=learning_rate)

  	# Create your binary classification model
  	model = Sequential()
  	model.add(Dense(128, input_shape=(30,), activation=activation))
  	model.add(Dense(256, activation=activation))
  	model.add(Dense(1, activation='sigmoid'))

  	# Compile your model with your optimizer, loss, and metrics
  	model.compile(optimizer=opt, loss='binary_classification', metrics=['accuracy'])
  	return model

# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256],
          'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}

# Create a randomize search cv object passing in the parameters to try
random_search = RandomizedSearchCV(model, param_distributions = params, cv = KFold(3))

# Running random_search.fit(X,y) would start the search,but it takes too long!
random_search.fit(X_train, y_train)

#%%
# =============================================================================
# Cross-Validation w/ Scikit-Learn
# =============================================================================

# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model, epochs = 50,
             batch_size = 128, verbose = 0)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv = 3)

# Print the mean accuracy
print('The mean accuracy was:', kfolds.mean())

# Print the accuracy standard deviation
print('With a standard deviation of:', kfolds.std())

