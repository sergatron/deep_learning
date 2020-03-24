# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import seaborn
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

#%%

# import data
banknotes = pd.read_csv('data/banknotes.csv')

# Use pairplot and set the hue to be our class
sns.pairplot(banknotes, hue='class') 

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations of each class
print('Observations per class: \n', banknotes['class'].value_counts())

#%%

# define predcitors and target
X = banknotes.drop('class', axis=1).values
y = banknotes['class'].values
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a sequential model
model = Sequential()

# Add a dense layer 
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))

# Compile your model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Display a summary of your model
model.summary()

# Train your model for 20 epochs
model.fit(X_train, y_train, epochs=20)

# Evaluate your model accuracy on the test set
# returns: [LOSS, ACCURACY]
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:',accuracy)
#%%
# Import to_categorical from keras utils module
from keras.utils import to_categorical
# =============================================================================
# Multi-Class Classification
# =============================================================================
darts = pd.read_csv('data/darts.csv')


# Instantiate a sequential model
model = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='softmax'))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes 


#%%
#### Split Data ####

# Use to_categorical on your labels
X = darts.drop(['competitor'], axis=1)
y = to_categorical(darts.competitor)

# Now print the to_categorical() result
print('One-hot encoded competitors: \n', y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Train your model on the training data for 200 epochs
model.fit(X_train, y_train, epochs=200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


#%%
#### Predictions ####
x_test_small = X_test.iloc[:5, :]
y_test_small = y_test[:5, :]

# Predict on coords_small_test
preds = model.predict(x_test_small)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred, y_test_small[i]))

# Extract the indexes of the highest probable predictions
preds = [np.argmax(pred) for pred in preds]

# Print preds vs true values
print()
print("{:10} | {}".format('Rounded Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{:25} | {}".format(pred, y_test_small[i]))


#%%


# =============================================================================
# Multi-Label Classification
# =============================================================================


irrigation = pd.read_csv('data/irrigation_machine.csv')
irrigation.drop("Unnamed: 0", axis=1, inplace=True)

X = irrigation.iloc[:, :20]
y = irrigation.iloc[:, 20:]

(sensors_train, sensors_test, parcels_train, parcels_test) = train_test_split(
    X,
    y, 
    test_size=0.15)

#%%

# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(128, input_shape=(20,), activation='relu'))

# Add an output layer of 3 neurons with sigmoid activation
model.add(Dense(3, activation='sigmoid'))

# Compile your model with adam and binary crossentropy loss
model.compile('adam',
           'binary_crossentropy',
           metrics=['accuracy'])

model.summary()

#### Train
# Train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, 
          parcels_train, 
          epochs = 50,
          validation_split = 0.2,
          verbose=False)

#### Predict
# Predict on sensors_test and round up the predictions
preds = model.predict(sensors_test)
# preds_rounded = np.round(preds)
# print('Rounded Predictions: \n', preds_rounded)

#### Evaluate
# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test, parcels_test)[1]

# Print accuracy
print('\nAccuracy:', accuracy)
print()

#%%
### Plot Loss and Accuracy
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


# Train your model and save its history
history = model.fit(sensors_train, 
                    parcels_train, 
                    epochs = 50,
                    validation_data=(sensors_test, parcels_test),
                    verbose=False)

# Plot train vs test loss during training
plot_loss(history.history['loss'], history.history['val_loss'])

# Plot train vs test accuracy during training
plot_accuracy(history.history['accuracy'], history.history['val_accuracy'])



#%%

# =============================================================================
# Early Stopping
# =============================================================================

# Import the early stopping callback
from keras.callbacks import EarlyStopping

# Define a callback to monitor val_acc
monitor_val_acc = EarlyStopping(monitor='val_acc', 
                       patience=5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, 
           epochs=1000, validation_data=(X_test, y_test),
           callbacks=[monitor_val_acc])

#%%

# =============================================================================
# Combining callbacks
# =============================================================================

# Import the EarlyStopping and ModelCheckpoint callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor = 'val_acc', patience=3)

# Save the best model as best_banknote_model.hdf5
modelCheckpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only = True)

# Fit your model for a stupid amount of epochs
history = model.fit(X_train, y_train,
                    epochs = 10000000,
                    callbacks = [monitor_val_acc, modelCheckpoint],
                    validation_data = (X_test, y_test))

#%%






















