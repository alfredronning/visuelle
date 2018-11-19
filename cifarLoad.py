import keras
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt


training_fraction = 1
training_examples = 50000




#Setting up the data in traing and testing set. x for input and y for label
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#Normalizing the input data between -0.5 and 0.5 to improve the learning.
x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
x_train /= 255
x_test /= 255


# One hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


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


model = load_model('./savedModels/best.h5')

learning_rate = 0.001
lossfunction = keras.losses.categorical_crossentropy
#optimize = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
optimize = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
#optimize = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=1e-5)
number_of_epochs = 1
batch_size = 32


model.fit(x_train, y_train, 
    batch_size = batch_size,
    epochs = number_of_epochs,
    verbose = 1,
    validation_data = (x_val, y_val))


#test results
final_loss, final_accuracy = model.evaluate(x_test, y_test)


# Save the model architecture
model.save('./savedModels/simple152.h5')

print("Test loss: "+str(final_loss))
print("Test accuracy: "+str(final_accuracy*100)+" %")

#plot accuracy

history = model.history.history
plt.figure(figsize=(12, 8))
plt.plot(history["acc"], label="Training accuracy")
plt.plot([number_of_epochs-1], [final_accuracy], 'o', label="Final test loss")
plt.show()