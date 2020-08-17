# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:25:48 2020

@author: incog
"""

import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Activation, Bidirectional, Flatten, AdditiveAttention
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence


from music21 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import pytz
from IPython.display import clear_output, Audio
from collections import Counter 
import glob
import pickle
import sys

from util import midi_to_onehot_dict, midi_to_onehot,load_doc,add_piece_start_stop
from extract_notes import note_length_event

def sample(preds, temperature=1.0):
    #print('sampling one')
    # helper function to sample an index from a probability array (from Keras library)
    preds = np.asarray(preds).astype('float64')
    #print (preds)
    preds = np.log(preds + 1e-8) / temperature  # Taking the log should be optional? add fudge factor to avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    #print (preds)
    probas = np.random.multinomial(1, preds, 1)
    
    #print(np.argmax(probas))
    return np.argmax(probas)




def generate_text(model_path,text_path,seed_ind,seq_length,gen_length):
    
    kern_files=glob.glob('data1/**/*.krn',recursive=True)
    piece=load_doc(kern_files[seed_ind])
    seed=''
    for line in piece.splitlines(True):
        if not (line.startswith('!!!') or line.startswith('!!')):
            seed+=line
    seed=seed[0:seq_length]
    print('Generating from seed: ')
    print(seed)
    print('----------')
    model=load_model(model_path)
    pattern=[]
    text=load_doc(text_path)
    
    chars = sorted(list(set(text)))
    n_vocab=len(chars)
    dictionary=dict((c, i) for i, c in enumerate(chars))
    rev_dict= dict((i, c) for i, c in enumerate(chars))
    
    for i in seed:
        pattern.append(dictionary[i])
    output_ind=pattern
    for i in range(gen_length):
        model_input=midi_to_onehot(pattern,dim=n_vocab)
        model_input=model_input.reshape(1, seq_length, n_vocab)
        preds=model.predict(model_input)
        char_ind=np.argmax(preds[0])
        pattern.append(char_ind)
        pattern=pattern[1:]
        output_ind.append(char_ind)
    output=''
    for i in output_ind:
        try:
            output+=rev_dict[i]
        except:
            print(i)
    print(output)
    return(output_ind,output)

def generate_abc_naive(model_path,text_path,seq_length,temp=1.):
    
    
    model=load_model(model_path)
    pattern=[]
    text=load_doc(text_path)
    chars = sorted(list(set(text)))
    #start_index = random.randint(0, len(text) - seq_length - 1)
    start_index=0
    pieces=text.split('\n\n')
    del text
    val_split=0.1
    pieces_train=pieces[0:len(pieces)-int(val_split*len(pieces))]
    pieces_validate=pieces[len(pieces)-int(val_split*len(pieces)):len(pieces)]
    del pieces
    sentence = pieces_validate[0].split()[start_index: start_index + seq_length-1]
    generated=[]
    
    #generated=np.array(generated)
    #sentence='M:9/8\nK:maj\n =G =E =E =E 2 =D =E =D =C | =G =E =E =E =F =G =A =B =c | =G =E =E =E 2 =D =E =D =C | =A =D =D =G =E =C =D 2 =A |'.split()
    experiment_path=os.path.dirname(os.path.dirname(model_path))
    dictionary=np.load(experiment_path+'/dictionary',allow_pickle=True)
    n_vocab=len(dictionary)
    char_indices=dictionary
    indices_char= {value:key for (key,value) in dictionary.items()}
    for i in sentence:
        generated.append(dictionary[i])
    #generated=[]
    #generated = sentence
    print('----- Generating with seed: "' + ''.join(sentence)+ '"')
    #sys.stdout.write(generated)
    
    for i in range(400):
        #x_pred = np.zeros((1, seq_length, n_vocab))
        #for t, char in enumerate(sentence):
        #    x_pred[0, t, char_indices[char]] = 1.
        x_pred=np.array(generated)

        preds = model.predict(x_pred, verbose=0)[-1][0]
        next_index = sample(preds, temp)
        next_char = indices_char[next_index]
        generated.append(next_index)
        generated=generated[1:]
        sentence.append(next_char)
        sentence = sentence[1:]

        sys.stdout.write(next_char)
        sys.stdout.flush()
    
    return(sentence)

#%%

  
if __name__=='__main__':
    model_path='experiments/folkrnn/ABC/data_v3_startstop_model_n2_s128_d0.5_bs64run_0/models/model-028-1.5035-1.3512'
    text_path='data/data_v3_startstop'
    tt=generate_abc_naive(model_path,text_path,2,temp=0.01)

