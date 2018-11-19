7# INITAL ANALYSIST MNIST DATASET

# 60,000 exampels of colour images. 50,000 in the training set, and 10,000 in the test set.

# The shape of the images is 32x32 3 channels rgb

# 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck (no overlap between automobile and truck)

# examples are evenly distributed. 6000 examples of each class

# testing set is simular to the training set

import keras
from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools



training_fraction = 0.9
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
y_test_values = y_test.copy()
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



""" #***************************************Best Mnist on Cifar*********************************************
learning_rate = 0.001
lossfunction = keras.losses.categorical_crossentropy
#opimize = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
opimize = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
number_of_epochs = 28
batch_size = 256


#Construct model
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dropout(0.3))
model.add(Dense(784, activation = "relu"))
model.add(Dense(784*2, activation = "relu"))
model.add(Dense(784, activation = "relu"))
model.add(Dropout(0.15))
model.add(Dense(256, activation = "relu"))
model.add(Dense(128, activation = "relu"))
model.add(Dense(28, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

model.compile(loss = lossfunction, optimizer = opimize, metrics = ['accuracy']) 
#**********************************************************************************************************"""



""" #**************************************Network from table2*********************************************
# Hyperparameters for the set
learning_rate = 0.001
lossfunction = keras.losses.categorical_crossentropy
#optimize = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
optimize = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#optimize = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=1e-5)
number_of_epochs = 12
batch_size = 32

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss = lossfunction, optimizer = optimize, metrics = ['accuracy'])

model.summary()

# params: 167k
#********************************************************************************************************** """


""" #**************************************Simple cifar over 75%*********************************************
# Hyperparameters for the set
learning_rate = 0.001
lossfunction = keras.losses.categorical_crossentropy
#optimize = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
optimize = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#optimize = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=1e-5)
number_of_epochs = 25
batch_size = 64

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# params: 170k 
#********************************************************************************************************** """

""" #**************************************Simple cifar over 70%*********************************************
# Hyperparameters for the set
learning_rate = 0.001
lossfunction = keras.losses.categorical_crossentropy
#optimize = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
optimize = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#optimize = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=1e-5)
number_of_epochs = 80
batch_size = 32

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Activation('softmax'))

# params: 17,498k 
#********************************************************************************************************** """



#***************************************Best Cifar model************************************************
# Hyperparameters for the set
learning_rate = 0.001
lossfunction = keras.losses.categorical_crossentropy
#optimize = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
optimize = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#optimize = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=1e-5)
number_of_epochs = 15
batch_size = 64

model = Sequential()
model.add(Conv2D(32, (3, 3), padding = "same", input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding = "same"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding = "same"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dropout(0.3))
model.add(Dense(128))
model.add(Activation('elu'))
model.add(Dropout(0.4))
model.add(Dense(64))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

#645,898 params

#**********************************************************************************************************

#Compiling and fitting the model
model.compile(loss = lossfunction, optimizer = optimize, metrics = ['accuracy'])

#function from
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
#for plotting confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


model.summary()
#fitting the model
model.fit(x_train, y_train, 
    batch_size = batch_size,
    epochs = number_of_epochs,
    verbose = 1,
    validation_data = (x_val, y_val))


#test results
final_loss, final_accuracy = model.evaluate(x_test, y_test)



# Save the model architecture
model.save('./savedModels/best.h5')
np.savetxt('./savedModels/bestTrainIdx.txt', idx_train, fmt='%d')


print("Test loss: "+str(final_loss))
print("Test accuracy: "+str(final_accuracy*100)+" %")

#plot confusion matrix
predictions = model.predict_classes(x_test)

cm = confusion_matrix(y_test_values, predictions)
cm_plot_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plot_confusion_matrix(cm, cm_plot_labels, title = 'Confusion matrix')

#plot accuracy
history = model.history.history
plt.figure(figsize=(12, 8))
plt.title("Model accuracy")
plt.plot(history["val_acc"], label="Validation accuracy")
plt.plot(history["acc"], label="Training accuracy")
plt.legend(['test', 'train'], loc='upper left')
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

