#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jose Manuel Casas: casasjm@uniovi.es
CENN paper: https://arxiv.org/abs/2205.05623: 
"""

# This module defines the Cosmic microwave background
# extraction neural network (CENN) architecture and trains it
# For using it: python CENN_Execute.py <Number of GPU>

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU, Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, MaxPool2D, BatchNormalization, Activation, Input
import os
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanSquaredError

from CENN_Input import reading_the_data

print (tf.__version__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


##############################################################################

# Original hyperparameters set

# learning_rate = 0.05
# batch_size = 32
# num_epochs = 500
# regularizer = keras.regularizers.l2(0.001)
# activation_function = tf.keras.layers.LeakyReLU(alpha=0.2)
# optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
# loss = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)

##############################################################################

def parse_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('GPU', type=str, help = 'Number of GPU you want to use')
    args = parser.parse_args()
    
    return args

#def error(y_pred, y_true):
    
class WeightedMSE(keras.losses.Loss):
    def call(self, y_true, y_pred):
        weights = tf.where(tf.greater(y_true, 0.1), 2.0, 1.0)  # Example weighting
        return tf.reduce_mean(weights * tf.square(y_true - y_pred))

    #return K.mean(K.square(y_pred - y_true), axis=-1)

def build_model_Conv(learning_rate):
    
  regularizer = keras.regularizers.l2(0.0001)
  activation_function = tf.keras.layers.LeakyReLU(alpha= 0.2)
  channels_order = 'channels_last'
  
  # Patches of 256x256 pixels and 3 frequency input channels
  
  inputs = Input(shape=(256, 256, 3))

  x = Conv2D(8, 9, 2, padding='same', kernel_regularizer=regularizer)(inputs)
  x = activation_function(x)
  x = BatchNormalization()(x)

  x = Conv2D(16, 9, 2, padding='same', kernel_regularizer=regularizer)(x)
  x = activation_function(x)
  x = BatchNormalization()(x)

  x = Conv2D(64, 7, 2, padding='same', kernel_regularizer=regularizer)(x)
  x = activation_function(x)
  x = BatchNormalization()(x)

  x = Conv2D(128, 7, 2, padding='same', kernel_regularizer=regularizer)(x)
  x = activation_function(x)
  x = BatchNormalization()(x)

  x = Conv2D(256, 5, 2, padding='same', kernel_regularizer=regularizer)(x)
  x = activation_function(x)
  x = BatchNormalization()(x)


  x = Conv2D(512, 3, 2, padding='same', kernel_regularizer=regularizer)(x)
  x = activation_function(x)
  x = BatchNormalization()(x)
  x = Dropout(0.3)(x)


  # Decoder

  x = Conv2DTranspose(256, 3, 2, padding='same', kernel_regularizer=regularizer)(x)
  x = activation_function(x)
  x = BatchNormalization()(x)

  x = Conv2DTranspose(128, 5, 2, padding='same', kernel_regularizer=regularizer)(x)
  x = activation_function(x)
  x = BatchNormalization()(x)

  x = Conv2DTranspose(64, 7, 2, padding='same', kernel_regularizer=regularizer)(x)
  x = activation_function(x)
  x = BatchNormalization()(x)

  x = Conv2DTranspose(16, 9, 2, padding='same', kernel_regularizer=regularizer)(x)
  x = activation_function(x)
  x = BatchNormalization()(x)

  x = Conv2DTranspose(8, 9, 2, padding='same', kernel_regularizer=regularizer)(x)
  x = activation_function(x)
  x = BatchNormalization()(x)

  outputs = Conv2DTranspose(1, 9, 2, padding='same')(x)

  model = keras.Model(inputs, outputs)
  
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss=WeightedMSE(), metrics=[MeanSquaredError()])

  
  return model

def train(learning_rate, batch_size, num_epochs, test_frequency, Patch_Size, Filtro, train_file_path, test_file_path):
 
    inputs_test, labels_test, inputs_train, labels_train = reading_the_data(train_file_path, test_file_path, Patch_Size)
    
    model = build_model_Conv(learning_rate)

    model.summary()

    Checkpoint = keras.callbacks.ModelCheckpoint('Models_'+Filtro+'/'+train_file_path[14:-3]+'_checkpoint-{val_loss:.5f}-{epoch:02d}.h5', monitor='val_loss', save_best_only=True)
    Best = keras.callbacks.ModelCheckpoint('Models_'+ Filtro+'/'+train_file_path[14:-3]+'_Red_'+ Filtro +'.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(inputs_train, labels_train, batch_size = batch_size, shuffle=True,
              epochs = num_epochs, verbose = 1, validation_freq = test_frequency,
              validation_data = (inputs_test, labels_test), callbacks=[Checkpoint, Best])

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('./Models_'+Filtro+'/'+train_file_path[26:-3]+'_Loss_'+Filtro+'.pdf')

    results = model.predict(inputs_test)

    #loss_error = error(results, labels_test)


