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

def generate_abc_naive(model_path,text_path,seq_length,temp=1.,no_exports=1):
    
    
    model=load_model(model_path)
    text=load_doc(text_path)
    chars = sorted(list(set(text)))
    #start_index = random.randint(0, len(text) - seq_length - 1)
    start_index=0
    pieces=text.split('\n\n')
    pieces_c=pieces[:22925]
    pieces_csharp=pieces[22925:]
    
    del text
    val_split=0.1
    
    
    pieces_train_c=pieces_c[0:len(pieces_c)-int(val_split*len(pieces_c))]
    pieces_validate_c=pieces_c[len(pieces_c)-int(val_split*len(pieces_c)):len(pieces_c)]
    pieces_train_csharp=pieces_csharp[0:len(pieces_csharp)-int(val_split*len(pieces_csharp))]
    pieces_validate_csharp=pieces_csharp[len(pieces_csharp)-int(val_split*len(pieces_csharp)):len(pieces_csharp)]
    pieces_train=pieces_train_c+pieces_train_csharp
    pieces_validate=pieces_validate_c+pieces_validate_csharp
    del pieces
    sentence = pieces_validate_c[0].split()[start_index: start_index + seq_length]
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
    if generate_pieces:
        for i in no_exports:
            abc=''
            stop=False
            count=0
            count_abc
            while stop==False and count<output_length:
                x_pred=np.array(generated)
        
                preds = model.predict(x_pred, verbose=0)[-1][0]
                #preds = model(x_pred, training = False)[-1][0]
                next_index = sample(preds, temp)
                next_char = indices_char[next_index]
                if next_char=='</s>':
                    stop=True
                    if len(abc)==0:
                        abc=process_abc(sentence,count_abc)
                    else:
                        abc+='\n\n'+process_abc(sentence,count_abc)
                generated.append(next_index)
                generated=generated[1:]
                sentence.append(next_char)
                sentence = sentence[1:]
        
                sys.stdout.write(next_char)
                sys.stdout.flush()
    else:
        for i in range(output_length):
            x_pred=np.array(generated)
    
            preds = model.predict(x_pred, verbose=0)[-1][0]
            #preds = model(x_pred, training = False)[-1][0]
            next_index = sample(preds, temp)
            next_char = indices_char[next_index]
            generated.append(next_index)
            generated=generated[1:]
            sentence.append(next_char)
            sentence = sentence[1:]
    
            sys.stdout.write(next_char)
            sys.stdout.flush()
    
    return(abc)

def make_abc_seeds(text_path,no_seeds,seq_length):
    text=load_doc(text_path)
    chars = sorted(list(set(text)))
    #start_index = random.randint(0, len(text) - seq_length - 1)
    start_index=0
    pieces=text.split('\n\n')
    pieces_c=pieces[:12117]
    pieces_csharp=pieces[12117:]
    
    del text
    val_split=0.1
    
    
    pieces_train_c=pieces_c[0:len(pieces_c)-int(val_split*len(pieces_c))]
    pieces_validate_c=pieces_c[len(pieces_c)-int(val_split*len(pieces_c)):len(pieces_c)]
    pieces_train_csharp=pieces_csharp[0:len(pieces_csharp)-int(val_split*len(pieces_csharp))]
    pieces_validate_csharp=pieces_csharp[len(pieces_csharp)-int(val_split*len(pieces_csharp)):len(pieces_csharp)]
    pieces_train=pieces_train_c+pieces_train_csharp
    pieces_validate=pieces_validate_c+pieces_validate_csharp
    del pieces
    inds=np.random.randint(0,len(pieces_validate_c),size=no_seeds)
    seeds=[]
    for i in inds:
        seeds.append(pieces_validate_c[i].split()[start_index: start_index + seq_length])
    return seeds

def generate_many_abc(model_path,seeds,seq_length,temp=1.,no_exports=1,generate_pieces=True,output_length=500):
    
    
    model=load_model(model_path)
    experiment_path=os.path.dirname(os.path.dirname(model_path))
    
    dictionary=np.load(experiment_path+'/dictionary',allow_pickle=True)
    n_vocab=len(dictionary)
    char_indices=dictionary
    indices_char= {value:key for (key,value) in dictionary.items()}
    sentences=[]
    seed_length=len(seeds[0])
    for seed in seeds:
        sentence=[]
        for i in seed:
            sentence.append(dictionary[i])
        sentences.append(np.array(sentence))
    #sentences=np.array(sentences,dtype='obj')
    #generated=[]
    #generated = sentence
    #print('----- Generating with seed: "' + ''.join(sentence)+ '"')
    #sys.stdout.write(generated)
    abc=''
    if generate_pieces:
        
        stop=False
        count=0
        count_abc=0
        delete_sentence=-1
        print('producing '+str(len(sentences))+' tunes')
        while len(sentences)>0 and count<output_length:
            
            if delete_sentence>=0:
                #print('deleting sentence ',delete_sentence, 'with length ',len(sentences[delete_sentence]))
                sentences.pop(delete_sentence)
                delete_sentence=-1
                if len(sentences)==0:
                    break
            x_pred=np.array(sentences,dtype='int16')
            preds_batches = model.predict(x_pred, verbose=0)
            for i,preds in enumerate(preds_batches):
                #preds = model(x_pred, training = False)[-1][0]
                next_index = sample(preds[-1], temp)
                next_char = indices_char[next_index]
                
                if next_char=='</s>':
                    #print('to be deleted:',i)
                    stop=True
                    
                    piece=[indices_char[ind] for ind in sentences[i] ]
                    if len(abc)==0:
                        abc=process_abc(piece,count_abc)
                        
                    else:
                        abc+='\n\n'+process_abc(piece,count_abc)
                        
                    count_abc+=1
                    delete_sentence=i
                    print('tunes made:',count_abc)
                    #sentences=np.delete(sentences,i,axis=0) #remove from seeds
                    #sentences.pop(i)
                    continue
                
                if len(sentences)>0:
                    sentences[i]=np.append(sentences[i],next_index)
        if count<output_length:
            for i in sentences:
                piece=[indices_char[ind] for ind in sentences[i] ]
                abc+='\n\n'+process_abc(piece,count_abc)
                count_abc+=1
                sentences=np.delete(sentences,i,axis=0)
        print(abc)        
        with open(experiment_path+'/output_seedlen_'+str(seed_length)+'.abc','w') as f:
            f.write(abc)
        print('saved at: '+experiment_path)
    else:
        for i in range(output_length):
            x_pred=np.array(generated)
    
            preds = model.predict(x_pred, verbose=0)[-1][0]
            #preds = model(x_pred, training = False)[-1][0]
            next_index = sample(preds, temp)
            next_char = indices_char[next_index]
            generated.append(next_index)
            generated=generated[1:]
            sentence.append(next_char)
            sentence = sentence[1:]
    
            sys.stdout.write(next_char)
            sys.stdout.flush()
    
    return(abc)

def process_abc(sentence,count_abc):
    abc='X:'+str(count_abc)+'\nM:4/4\nK:Cmaj\n'
    for i in sentence[1:]:
        abc+=i
    return abc

def process_abc2(sentence,count_abc):
    abc='X:'+count_abc+'\n'
    abc+=sentence[1]+'\n'+sentence[2]+'\n'
    for i in sentence[3:]:
        abc+=i
    return abc
#%%

  
if __name__=='__main__':
    model_path='experiments/seq_song/ABC/data_V3_nohead_model_n1_s32_d0.2_sl100_bs256_C_run_0/models/model-164-1.7136-1.6604'
    text_path='data/data_V3_nohead'
    no_seeds=5
    seq_length=10
    seeds=make_abc_seeds(text_path, no_seeds, seq_length)
    tt=generate_many_abc(model_path, seeds, seq_length)
