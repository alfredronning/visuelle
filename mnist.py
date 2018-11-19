# INITAL ANALYSIST MNIST DATASET

# 70,000 exampels of hand written digits. 60,000 in the training set, and 10,000 in the test set.

# The shape of the images is 28x28 greyscale

# 10 classes from 0 - 10

# examples are almost evenly distributed between the classes

# testing set is simular to the training set

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt


training_fraction = 0.9
training_examples = 60000


#Setting up the data in traing and testing set. x for input and y for label
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#Normalizing the input data between -0.5 and 0.5 to improve the learning.
pixel_mean = x_train.mean(axis=0)
pixel_std = x_train.std(axis=0) + 1e-10 # Prevent division-by-zero errors
x_train = (x_train - pixel_mean) / pixel_std
x_test = (x_test - pixel_mean) / pixel_std



# One hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)



# Change data shape to fit tensorflow
x_train = x_train[:,:,:, np.newaxis].astype(np.float32)
x_test = x_test[:,:,:, np.newaxis].astype(np.float32)



#Splitting the data into training and validation
indexes = np.arange(training_examples)
np.random.shuffle(indexes)
# Select random indexes for train/val set
idx_train = indexes[:int(training_fraction*training_examples)]
idx_val = indexes[int(training_fraction*training_examples):]

x_val = x_train[idx_val]
y_val = y_train[idx_val]

x_train = x_train[idx_train]
y_train = y_train[idx_train]


""" # *****************************Implementing the network in table 1*************************************

# Hyperparameters for the set
learning_rate = 0.001
lossfunction = keras.losses.categorical_crossentropy
#opimize = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
opimize = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
number_of_epochs = 40
batch_size = 128


#Construct model
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(128, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

# ********************************************************************************************************************"""



# **********************Finding the simplest fully connected network that achieves more than 95%******************
 
# Hyperparameters for the set
learning_rate = 0.001
lossfunction = keras.losses.categorical_crossentropy
#opimize = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
opimize = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
number_of_epochs = 120
batch_size = 32


#Construct model
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dropout(0.18))
model.add(Dense(12, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

# ********************************************************************************************************************



""" # ************************************Implementing the best mnist network*************************************

# Hyperparameters for the set
learning_rate = 0.001
lossfunction = keras.losses.categorical_crossentropy
#opimize = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
opimize = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
number_of_epochs = 25
batch_size = 256


#Construct model
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dropout(0.3))
model.add(Dense(784, activation = "relu"))
model.add(Dense(784*2, activation = "relu"))
model.add(Dense(784, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(256, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(28, activation = "relu"))
model.add(Dense(10, activation = "softmax")) 
# ********************************************************************************************************************"""

model.compile(loss = lossfunction, optimizer = opimize, metrics = ['accuracy'])

model.summary()

model.fit(x_train, y_train, 
    batch_size = batch_size,
    epochs = number_of_epochs,
    verbose = 1,
    validation_data = (x_val, y_val))



#test results
final_loss, final_accuracy = model.evaluate(x_test, y_test)
print("Test loss: "+str(final_loss))
print("Test accuracy: "+str(final_accuracy))

#plot confusion matrix
predictions = model.predict_classes(x_test)

cm = confusion_matrix(y_test_values, predictions)
cm_plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_confusion_matrix(cm, cm_plot_labels, title = 'Confusion matrix')

#plot accuracy

history = model.history.history
plt.figure(figsize=(12, 8))
plt.plot(history["val_acc"], label="Validation accuracy")
plt.plot(history["acc"], label="Training accuracy")
plt.plot([number_of_epochs-1], [final_accuracy], 'o', label="Final test loss")
plt.show()

#plot loss
plt.figure(figsize=(12, 8))
plt.title("Model Loss")
plt.plot(history["loss"], label="Validation accuracy")
plt.plot(history["val_loss"], label="Training accuracy")
plt.legend(['test', 'train'], loc='upper left')
plt.plot([number_of_epochs-1], [final_loss], 'o', label="Final test loss")
plt.show()
