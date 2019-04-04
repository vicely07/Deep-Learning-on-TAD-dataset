import numpy as np
import sys
import pandas as pd

from keras.models import Sequential  # what kind of model ? a sequential model
from keras.layers.core import Dense, Activation, Dropout  # different layers, activation function, and dropout
from keras.optimizers import SGD, Adam  # optimization algorithm


NB_EPOCH = 100  # number of epoch
BATCH_SIZE = 100  # mini batch size
VERBOSE = 0  # display results during training
OPTIMIZER = Adam()  # choose optimizer
METRICS = ['accuracy']
LOSS = 'binary_crossentropy'  # We use 'binary crossentropy because there are only 2 values in our output layer
DropOut = 0.3


from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score

# python function to obtain performance metrics


def GetMetrics(model, x, y):
  pred = model.predict_classes(x)
  pred_p = model.predict(x)
  fpr, tpr, thresholdTest = roc_curve(y, pred_p)
  aucv = auc(fpr, tpr)
  precision, recall, fscore, support = precision_recall_fscore_support(y, pred, average='macro')
  print('auc,acc,mcc,precision,recall,fscore,support:', aucv, accuracy_score(y, pred), matthews_corrcoef(y, pred), precision, recall, fscore, support)
  return [aucv, accuracy_score(y, pred), matthews_corrcoef(y, pred), precision, recall, fscore]

# Python function to build a NN with dense layers.
# N_HIDDEN_LAYERS: number of hidden dense layers
# N_HIDDEN_NODES: number of nodes in each hidden layer


def layerDNN(x_train, y_train, x_val, y_val, N_HIDDEN_NODES, N_HIDDEN_LAYERS):
  model = Sequential()
  model.add(Dense(N_HIDDEN_NODES, input_shape=(x_train.shape[1],)))
  model.add(Activation('relu'))
  model.add(Dropout(0.3))
  for i in range(1, N_HIDDEN_LAYERS):
    model.add(Dense(N_HIDDEN_NODES))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  print(model.summary())
  from keras.callbacks import EarlyStopping
  early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=8)
  from keras.callbacks import ModelCheckpoint
  filepath = str(N_HIDDEN_LAYERS) + "_" + str(N_HIDDEN_NODES) + "_best_model.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
  model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
  Tuning = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE,
                     validation_data=(x_val, y_val), callbacks=[checkpoint, early_stopping_monitor])
  print("training performance")
  GetMetrics(load_model(filepath), x_train, y_train)
  print("validation performance")
  GetMetrics(load_model(filepath), x_val, y_val)
  return Tuning, model

# python function to perform 10-fold cross validation on the layerDNN model
# N_HIDDEN_LAYERS: number of hidden dense layers
# N_HIDDEN_NODES: number of nodes in each hidden layer


def layerDNN_kfold(x_train, y_train, x_test, y_test, N_HIDDEN_NODES, N_HIDDEN_LAYERS):
  Metrics = np.ones(shape=(10, 6))
  kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
  k = 0
  for train, val in kfold.split(x_train, y_train):
    tuning, model = layerDNN(x_train[train], y_train[train], x_train[val], y_train[val], N_HIDDEN_NODES, N_HIDDEN_LAYERS)
    print("testing accuracy")
    Metrics[k, :] = GetMetrics(model, x_test, y_test)
    k = k + 1
  return Metrics


# need four numpy matrices
# x_train = training features
# y_train = training class labels
# x_test = testing features
# y_test = testing class labels



# load training file (CSV file), file column should be class label (0 or 1)
# replace sys.argv[1] by file name
Train = np.loadtxt(sys.argv[1], delimiter=',')
print(Train.shape)
x_train = Train[:, 1:(Train.shape[1])]
y_train = Train[:, 0]
print(x_train.shape)
print("class labels")
print(set(y_train))

# load testing file ((CSV file), file column should be class label (0 or 1)
# Replace sys.argv[2] by file name
Test = np.loadtxt(sys.argv[2], delimiter=',')  # this is our verification our model runs
print(Test.shape)
x_test = Test[:, 1:(Test.shape[1])]
y_test = Test[:, 0]
print(x_test.shape)
print(y_test.shape)


# perform 10-fold cross validation using 2 hidden layers and 50 nodes in each hidden layer
Metrics = layerDNN_kfold(x_train, y_train, x_test, y_test, 200, 1)
print("average testing metrics")
print(np.average(Metrics, axis=0))
print("std testing metrics")
print(np.std(Metrics, axis=0))
print("End of this run.")
print("########################################################################################################")
