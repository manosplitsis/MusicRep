"""Module to quick-test various See RNN functionalities"""
import os
import tensorflow as tf
import numpy as np

os.environ['TF_KERAS'] = "1"  # configurable
os.environ['TF_EAGER'] = "0"  # configurable

print("TF version:", tf.__version__)

TF_2 = (tf.__version__[0] == '2')
eager_default = "1" if TF_2 else "0"
TF_EAGER = bool(os.environ.get('TF_EAGER', eager_default) == "1")
TF_KERAS = bool(os.environ.get('TF_KERAS', "0") == "1")

if TF_EAGER:
    if not TF_2:
        tf.enable_eager_execution()
    print("TF running eagerly")
else:
    if TF_2:
        tf.compat.v1.disable_eager_execution()
    print("TF running in graph mode")

if TF_KERAS:
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Input, LSTM, GRU, SimpleRNN, Bidirectional, Embedding, Dropout, TimeDistributed, Dense, Activation
    # from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU  # [1]
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.optimizers import Adam
else:
    import keras.backend as K
    from keras.layers import Input, LSTM, GRU, SimpleRNN, Bidirectional
    # from keras.layers import CuDNNLSTM, CuDNNGRU  # [1]: uncomment if using GPU
    from keras.models import Model
    from keras.optimizers import Adam

from see_rnn import get_gradients, get_outputs, get_rnn_weights
from see_rnn import features_0D, features_1D, features_2D
from see_rnn import rnn_heatmap, rnn_histogram


from model import build_model3
from train_midi import Data_Gen_Midi3

###############################################################################






def make_model(rnn_layer, batch_shape, units=8, bidirectional=False):
    ipt = Input(batch_shape=batch_shape)
    if bidirectional:
        x = Bidirectional(rnn_layer(units, return_sequences=True,))(ipt)
    else:
        x = rnn_layer(units, return_sequences=True)(ipt)
    out = rnn_layer(units, return_sequences=False)(x)

    model = Model(ipt, out)
    model.compile(Adam(lr=1e-2), 'mse')
    return model

def make_data(batch_shape, units):
    return (np.random.randn(*batch_shape),
            np.random.uniform(-1, 1, (batch_shape[0], units)),
            np.random.uniform(0, 2, batch_shape[0]))


def train_musrep(model,train_loader):
    
    for i in range(iterations):
        model.train_on_batch(train_loader[i][0],train_loader[i][1])
        print(end='.')

def train_model(model,batch_shape, iterations):
    #batch_shape = K.int_shape(model.input)
    units = model.layers[1].units
    x, y, sw = make_data(batch_shape, units)

    for i in range(iterations):
        model.train_on_batch(x, y, sw)
        print(end='.')  # progbar
        if i % 40 == 0:
            x, y, sw = make_data(batch_shape, units)

def viz_outs(model, idx=1):
    #x, y, _ = make_data(K.int_shape(model.input), model.layers[2].units)
    x=train_loader[0][0]
    outs = get_outputs(model, idx, x)

    features_1D(outs[:1], n_rows=8, show_borders=False)
    features_2D(outs,     n_rows=8, norm=(-1,1))

def viz_weights(model, idx=1):
    rnn_histogram(model, idx, mode='weights', bins=400)
    print('\n')
    rnn_heatmap(model,   idx, mode='weights', norm='auto')

def viz_outs_grads(model, idx=1):
    #x, y, _ = make_data(K.int_shape(model.input), model.layers[2].units)
    x,y=train_loader[0]
    grads = get_gradients(model, idx, x, y)
    kws = dict(n_rows=8, title='grads')

    features_1D(grads[0], show_borders=False, **kws)
    features_2D(grads,    norm=(-1e-4, 1e-4), **kws)

def viz_outs_grads_last(model, idx=2):  # return_sequences=False layer
    #x, y, _ = make_data(K.int_shape(model.input), model.layers[2].units)
    x,y=train_loader[0]
    
    grads = get_gradients(model, idx, x, y)
    features_0D(grads)

def viz_weights_grads(model, idx=1):
    #x, y, _ = make_data(K.int_shape(model.input), model.layers[2].units)
    x,y=train_loader[0]
    
    kws = dict(_id=idx, input_data=x, labels=y)

    rnn_histogram(model, mode='grads', bins=400, **kws)
    print('\n')
    rnn_heatmap(model,   mode='grads', cmap=None, absolute_value=True, **kws)

def viz_prefetched_data(model, data, idx=1):
    rnn_histogram(model, idx, data=data)
    rnn_heatmap(model,   idx, data=data)

###############################################################################

import numpy as np
import pandas as pd
from util import add_piece_start_stop,keep_dataset_notes



units = 32
layers=1
dropout=0.2
batch_size=32
seq_length=16
n_vocab=136
iterations=5
notes_path='notes/notes_event1_res8'
notes=pd.read_pickle(notes_path)
notes=notes[0:1600]
notes=add_piece_start_stop(notes)
notes=list(notes)
notes.sort(key=lambda x: len(x), reverse=True)
#notes=notes[batch_size*2:] #delete the first two (biggest) batches to save memory in gpu
dictionary=keep_dataset_notes(notes,zero_pad=True)

n_vocab=len(dictionary)

#val_split=0.1
#notes_train=notes[0:len(notes)-int(val_split*len(notes))]


train_loader=Data_Gen_Midi3(notes,dictionary,batch_size=batch_size)


###########################################################################33
units = 32
layers=1
dropout=0.2
batch_size=32
seq_length=16
n_vocab=136
batch_shape = (batch_size, seq_length)

#model = make_model(LSTM, batch_shape, units)
model=build_model3(batch_size, n_vocab,lstm_size=units,lstm_no=layers,dropout_rate=dropout)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
#train_model(model,batch_shape, 50)
train_musrep(model, train_loader)

viz_outs(model, 1)
viz_outs_grads(model, 1)
viz_outs_grads_last(model, 2)
viz_weights(model, 1)
viz_weights_grads(model, 1)

data = get_rnn_weights(model, 1)
viz_prefetched_data(model, data, 1)
