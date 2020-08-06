# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:26:14 2020

@author: incog
"""

import os

import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Activation, Bidirectional, Flatten, AdditiveAttention
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

MODEL_DIR = './model'




def save_weights(epoch, model):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(epoch)))


def load_weights(epoch, model):
    model.load_weights(os.path.join(MODEL_DIR, 'weights.{}.h5'.format(100)))


def build_model(batch_size, seq_len, n_vocab,lstm_size=256,lstm_no=1,dropout_rate=0.2):
    '''
    Build the model--> Embedding the every character as neuron Dense Layer
                      --> 3 Layer of LSTM each of 256 units(cell) in it.
                      --> Each in between layer of LSTM Dropout with probabilty of 0.2
                          i.e 20 % of neurons connection is Drop to prevent the model
                          from overfitting.
                      -->
    '''

    model = Sequential()
    #model.add(Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_len)))
    if lstm_no==1:
        model.add(LSTM(lstm_size,return_sequences=False,input_shape=(seq_len,n_vocab)))
        lstm_no-=1
    else:
        for i in range(lstm_no-1):
            model.add(LSTM(lstm_size, return_sequences=True,input_shape=(seq_len,n_vocab)))
            model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_size,return_sequences=False))
    #model.add(TimeDistributed(Dense(n_vocab))) 
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    return model


def build_state_model(batch_size, seq_len, n_vocab,lstm_size=256,lstm_no=1,dropout_rate=0.2):
  
    model = Sequential()
    #model.add(Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_len)))
    if lstm_no==1:
        model.add(LSTM(lstm_size,return_sequences=False,batch_input_shape=(batch_size,seq_len,n_vocab),stateful=True))
        lstm_no-=1
    else:
        for i in range(lstm_no-1):
            model.add(LSTM(lstm_size, return_sequences=True))
            model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_size,return_sequences=False))
    #model.add(TimeDistributed(Dense(n_vocab))) 
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    return model


def create_simple_network_func(network_input_shape, n_vocab, load_weights=False, weights='',lstm_size=32,lstm_no=1,dropout_rate=0.2):
  inputs=Input(shape=(network_input_shape[1], network_input_shape[2])) #n_time_steps, n_features?
  #enc=Dense(32)(inputs)
  encoder1=LSTM(lstm_size, return_sequences=False)(inputs)
  drop1=Dropout(dropout_rate)(encoder1)
  #encoder2=LSTM(128, return_sequences=False)(drop1)
  #drop2=Dropout(0.5)(encoder2)
  predict=Dense(n_vocab, activation='softmax')(drop1)
  
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
  model=Model(inputs=inputs,outputs=predict)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
  #uncomment next line to load a weight file
  if load_weights:
    model.load_weights(weights)

  return model
