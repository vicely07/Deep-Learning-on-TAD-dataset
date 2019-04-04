
import numpy as np

# First, load the train data from a csv file to a numpy Array. Note that last column is assumed to be the
# class label. You will need the numpy library. We will #separate the predictors and class labels to two
#  separate matrices 'x_train' and 'y_train', respectively.

Train = np.loadtxt('train_sequences_6mer_stride1_tokens.csv.count.csv', delimiter=',')
#Train = np.loadtxt('train_12mer_keras_trial.csv', delimiter=',')  # this is our verification our model runs
# Train.shape gives you the number of rows and columns.
print(Train.shape)
# get all rows, all columns except the last column
x_train = Train[:, 1:(Train.shape[1])]
y_train = Train[:, 0] # get the first column (the class label)
print(x_train.shape)
print(y_train.shape)


# Load testing data from a csv file to a numpy Array. Note that last column is class label. We will
# separate
# the predictors and class labels to two separate #matrices 'x_test' and 'y_test', respectively.

Test = np.loadtxt('test_sequences_6mer_stride1_tokens.csv.count.csv', delimiter=',')
#Test = np.loadtxt('test_12mer_keras_trial.csv', delimiter=',')  # this is our verification our model runs
# Test.shape gives you the number of rows and columns.
print(Test.shape)
# get all rows, all columns except the last column
x_test = Test[:, 1:(Test.shape[1])]
y_test = Test[:, 0]
print(x_test.shape)
print(y_test.shape)


# Normalize the predictor matrices between 0 and 1. 255 is the highest value a pixel can take in a gray scale image.

#x_train = x_train / 255
#x_test = x_test / 255


# To normalize between -1 and 1
# x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
# x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)


# Class labels are stored in 'y_train' and 'y_test' matrices as integers. There are 10 different classes
#  in this data set, we need to convert integer class labels to categorical class labels since there will
#  be more than one node in the output layer. We will use the 'np_utils' method under keras library.

from keras.utils import np_utils
NB_CLASSES = 2  # number of classes
print('shape of y_train and y_test before categorical')
print(y_train.shape)
print(y_test.shape)

########################################################################
# y_train = np_utils.to_categorical(y_train, NB_CLASSES)   # changed to_categorical to to_binary
# y_test = np_utils.to_categorical(y_test, NB_CLASSES)     # changed to_categorical to to_binary
########################################################################


from keras.models import Sequential  # what kind of model ? a sequenctial model
from keras.layers.core import Dense, Activation, Dropout  # different layers, activation function, and dropout
from keras.optimizers import SGD  # optimization algorithm

NB_EPOCH = 100  # number of epoch
BATCH_SIZE = 100  # mini batch size
VERBOSE = 1  # display results during training
NB_CLASSES = 2  # number of classes
OPTIMIZER = SGD()  # choose optimizer
# OPTIMIZER = keras.optimizers.Adam() # choose optimizer
N_HIDDEN = 100  # number of nodes in the hidden layer
VALIDATION_SPLIT = 0.2  # 80% training and 20%validation
METRICS = ['accuracy']
LOSS = 'binary_crossentropy'  # We use 'binary crossentropy because there are only 2 values in our output layer
DropOut = 0.3


# Model with one hidden dense layer
# Now, lets build the model architecture. It always starts with an empty model and different layers are
# added one after another.
# Use 'Dense' to add a fully connected layer. 'input_shape' represents the dimensions of one row in the
# training set.
# It ends with an output layer. In this case, the output layer has more than 1 node and hence the activation
# should be 'softmax'. With only 1 node we need to use the Sigmoid activation layer.

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))
print(model.summary())


# Once, the model architecture is set up, the model can be trained. First, indicate which optimizer, loss
# function, and metrics to report.
# Secondly, when the model is trained, provide the training predictors ('x_train') and training class
# ('y_train').
# Here, 'validation_split' is the percentage of training data that will be randomly selected during each
# epoch as the validation set.
# Instead of using 'validation_split', if you have a 'x_valid' and 'y_valid' data, you can use it by simply
# using 'validation_data = (x_valid,y_valid)' instead of 'validation_split = VALIDATION_SPLIT'. In this case,
# the same validation data is used in each epoch.
# One can use k-fold cross validation, because of the amount of time deep leaning models take, it may be
# avoided.
# The history of the optimization will be saved in 'Tuning' and the model from last epoch in 'model'.


#model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
#Tuning = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)


# Plot the 'loss' and 'accuracy' during optimization process.
# Define a function plotHistory that requires the Tuning history. We will use this function later on.

import matplotlib
import matplotlib.pyplot as plt


def plotHistory(Tuning):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(Tuning.history['loss'])
    axs[0].plot(Tuning.history['val_loss'])
    axs[0].set_title('loss vs epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'vali'], loc='upper left')

    axs[1].plot(Tuning.history['acc'])
    axs[1].plot(Tuning.history['val_acc'])
    axs[1].set_title('accuracy vs epoch')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'vali'], loc='upper left')
    plt.show(block=False)
    plt.show()



# Stopping criteria and saving model
# Above, the optimization process will run for all the number of epochs. The optimization process needs
# to be stopped before the model overfits. One stopping criteria is to monitor the value of loss function
# on validation data 'val_loss' and if it doesn't improve for (let's say) 8 consecutive epochs, the
# optimization process can be stopped. Instead of monitoring 'val_loss', one can also monitor 'val_acc',
# 'train_loss', and 'train_acc' etc.


from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=8)


# Also, include a way of saving the best model (in 'hdf5' format) during the optimization process based on
# 'val_loss'. You may need this model later on.

from keras.callbacks import ModelCheckpoint
filepath = "best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
Tuning = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
                   validation_split=VALIDATION_SPLIT, callbacks=[checkpoint, early_stopping_monitor])

plotHistory(Tuning)


# Predicting on testing data
# Load the saved model and perform prediction using 'predict_classes' (for predicting class labels') or
# 'predict' (for probabilities).

from keras.models import load_model
filepath = 'best_model.hdf5'
finalModel = load_model(filepath)
pred_class = model.predict_classes(x_test)  # predict class labels
print('predicted class ', pred_class)
print('Dimensions of pred_class ', pred_class.shape)
pred_p = model.predict(x_test)  # predict probabilities
print('Dimensions of pred_p ', pred_p.shape)
print('predicted probabilities of first 5 \n', pred_p[0:5])

# To obtain accuracy metric, need sklearn library. Note that we need to convert the class labels back to
# interger class labels.

from sklearn.metrics import accuracy_score
# convert categorical to integer class labels
y_classes = [np.argmax(y, axis=None, out=None) for y in y_test]
print('testing accuracy', accuracy_score(y_classes, pred_class))

from sklearn.metrics import accuracy_score
pred_class = model.predict_classes(x_train)
# convert categorical to integer class labels
y_classes = [np.argmax(y, axis=None, out=None) for y in y_train]
print('training accuracy', accuracy_score(y_classes, pred_class))

