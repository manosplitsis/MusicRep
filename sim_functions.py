import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, TimeDistributed, LSTM, RepeatVector, Dropout, GRU
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

import argparse
import gc 
import math
import sys
import music21 as m21
import datetime as dt
import time
from ast import literal_eval


from scipy.ndimage.interpolation import shift
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

#import get_data as gd

def harmony_encode(harmony_bin):

    # print(np.count_nonzero(y_bin))
    #NOTE if it is 2D, it needs melody_encoded[1], if 1D melody_encoded[0]
    harmony_bin = harmony_bin.astype(int)
    if (np.count_nonzero(harmony_bin)!=0):
        harmony_encoded = np.where(harmony_bin == 1)
        # print(harmony_encoded)
        if len(harmony_bin.shape)>1:
            harmony_encoded = harmony_encoded[1]
        else:
            harmony_encoded = harmony_encoded[0]
    else:   
        harmony_encoded = 0

    # print(melody_encoded)

    return harmony_encoded

def harmony_decode(code):
    
    folder_path = "/media/datadisk/imim/existential_crisis/saved_data"
    f = open(folder_path + os.sep + "exp3_harmony_int2chord_dict.pkl",'rb')
    d_int2chord = pickle.load(f)
    f.close()

    print(code.shape)
    harmony_decoded = d_int2chord[int(code)]

    return harmony_decoded


def meloharmony_encode(melody_bin):

    # print(np.count_nonzero(y_bin))
    #NOTE if it is 2D, it needs melody_encoded[1], if 1D melody_encoded[0]
    melody_bin = melody_bin.astype(int)
    if (np.count_nonzero(melody_bin)!=0):
        melody_encoded = np.where(melody_bin == 1)
        if len(melody_bin.shape)>1:
            melody_encoded = melody_encoded[1]
        else:
            melody_encoded = melody_encoded[0]
    else:   
        melody_encoded = 0

    # print(melody_encoded)

    return melody_encoded

def meloharmony_decode(code):
    
    folder_path = "/media/datadisk/imim/existential_crisis/saved_data"
    f = open(folder_path + os.sep + "meloharmony_int2chord_dict.pkl",'rb')
    d_int2chord = pickle.load(f)
    f.close()

    print(code.shape)
    melody_decoded = d_int2chord[int(code)]

    return melody_decoded


def sampling_temp(output, temperature):
    value_index = -9
    # helper function to sample an index from a probability array (from Keras library)
    # output_temp = np.asarray(output[0]).astype('float64')
    # output_temp = np.log(output[0] + 1e-8) / temperature  # Taking the log should be optional? add fudge factor to avoid log(0)
    
    if temperature==0:
        print("\nError: Cannot devide by zero temperature!\n")
        exit()
    # elif temperature!=1:      
    # else:

    print("\nSampling: Temperature\n")
    output_temp = output[0] / temperature 
    exp_preds = np.exp(output_temp)
    output_temp = exp_preds / np.sum(exp_preds)
    # output_temp = output_temp / np.sum(output_temp)
    probas = np.random.multinomial(1, output_temp, 1)
    # print("Probs"+str(probas))
    # print("Output temp:"+str(output_temp))
    # print(output)

    probas = output[0]
    
    value_index = np.argmax(probas)
    value = output[0][value_index]

    print(value_index, value)

    return value_index


def sampling_top_k(output, top_k, filter_value = 0): #filter_value = -float('Inf')):
    print("\nSampling: Top_k\n")
    
    value_index = -9

    # Safety check
    top_k_nums = min(top_k, len(output[0]))  
    
    if top_k_nums > 0:
        # Remove all tokens with a probability less than the last token of the top-k

        # print(output[0])
        output_temp = output[0]
        max_elements = output_temp.argsort()[-top_k_nums:][::-1]
        # print(output[0])
        min_of_max_elements = min(output_temp[max_elements])
        # print(max_elements, min_of_max_elements)
        
        indices_to_zerofy, = np.where(output_temp < min_of_max_elements)
        # print(indices_to_zerofy)

        output_temp[indices_to_zerofy] = filter_value
        output_temp = output_temp.astype(float)
        output_temp_norm =  output_temp / np.sum(output_temp)
        # print(output_temp_norm.T)
        # print("%.50f" % np.sum(output_temp_norm[:-1]))
        probas = np.random.multinomial(1, output_temp_norm, 1)
        # print(probas)
        value_index = np.argmax(probas)
        value = output_temp[value_index]
        
        print(value_index, value)

    return value_index


def sampling_top_p(output, top_p, filter_value = 0):
    print("\nSampling: Top_p\n")
    
    value_index = -9

    # print(output[0])
    output_temp = output[0]

    elements_sorted = output_temp.argsort()[::-1][:]
    # print(elements_sorted)
    # print(output[0])
  
    # print(output[0])
    output_sorted = np.sort(output_temp)[::-1]
    # print(output_sorted)
    output_cumulprob = np.cumsum(output_sorted)
    # print(output_cumulprob)

    indices_to_zerofy, = np.where(output_cumulprob < top_p)
    # print(indices_to_zerofy)

    output_sorted[indices_to_zerofy] = filter_value
    output_sorted = output_sorted.astype(float)
    output_temp_norm = output_sorted / np.sum(output_sorted)
    probas = np.random.multinomial(1, output_temp_norm, 1)
    # print(probas)
    value_index = np.argmax(probas)
    value = output[0][elements_sorted[value_index]]
    value_index = elements_sorted[value_index]
    
    print(value_index, value)

    return value_index


def sampling(output, temperature = 1.0, top_k = 0, top_p = 0.0):
    value_index = -9    
    if temperature != 1.0:
        value_index = sampling_temp(output, temperature)
    
    elif top_k > 0:
        value_index = sampling_top_k(output, top_k)

    elif top_p > 0.0:
        value_index = sampling_top_p(output, top_p)



    return value_index   


def softmax_to_1hot(y_bin_original,temperature,top_k,top_p):
    y_bin = y_bin_original
    print(y_bin.shape)
    y_bin = y_bin.T
    # max_ind = np.where(y_bin == np.amax(y_bin))[0][0]
    # max_ind_test = np.argmax(y_bin)
    
    # print(max_ind,max_ind_test)
    # if max_ind == max_ind_test:
    #     print("SAME")
    # else:
    #     exit()

    #SAMPLING - NOTE
    #max_ind== index returned from the sampling function
    max_ind = sampling(y_bin,temperature,top_k,top_p)
    # print(y_bin.shape)
    # exit()

    y_bin_original[:,max_ind] = 1
    y_bin_original[y_bin_original!=1] = 0
    # if y_bin.shape[0]>y_bin.shape[1]:
    #     y_bin = y_bin.T
    # y_bin = y_bin.T
    # print(y_bin.shape)

    # y_bin[max_ind,:] = 1
    # y_bin[y_bin!=1] = 0

    return y_bin_original

def load_compile_model(model_path,model_type):
    
    #Load trained model
    final_model = load_model(model_path)
    
    if model_type == "seq":
        
        optimizer = Adam(lr=0.01)
        final_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    elif model_type == "non_seq20":
        
        optimizer = Adam(lr=0.01)
        final_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    elif model_type == "model_bot":

        optimizer = Adam(lr=0.01)
        final_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    final_model.summary()

    return final_model

def human_pred_step(human_pred_past_win,human_model,classes_num,temperature,top_k,top_p):
    #predict next timestep with trained human_model
    human_pred_past_win = human_pred_past_win.reshape(-1,human_pred_past_win.shape[0],human_pred_past_win.shape[1])
    pred_p0 = human_model.predict(human_pred_past_win)
    pred_p0 = pred_p0.reshape(-1,classes_num)
    pred_p0 = softmax_to_1hot(pred_p0,temperature,top_k,top_p)
    pred_p0 = meloharmony_encode(pred_p0)

    return int(pred_p0)

def bot_pred_step(bot_pred_past_win,bot_model,classes_num,temperature,top_k,top_p):
    #predict next timestep with trained bot_model
    human_pred_past_win = bot_pred_past_win.reshape(-1,bot_pred_past_win.shape[0],bot_pred_past_win.shape[1])
    pred_p1 = bot_model.predict(human_pred_past_win)
    pred_p1 = pred_p1.reshape(-1,classes_num)
    pred_p1 = softmax_to_1hot(pred_p1,temperature,top_k,top_p)
    pred_p1 = harmony_encode(pred_p1)

    return int(pred_p1)
