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

from util import midi_to_onehot_dict, midi_to_onehot,load_doc
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


def generate_notes_hot(model,network_input,smpl=True,temperature=1.0,output_length=500):
  """ Generate notes from the neural network based on a sequence of notes """
  # pick a random sequence from the input as a starting point for the prediction
  #start = np.random.randint(0, len(network_input)-1)

  pattern = network_input
  prediction_output = []


  # generate 500 notes
  for note_index in range(output_length):
      prediction_input = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))
      #prediction_input = prediction_input / float(n_vocab)

      prediction = model(prediction_input, training=False)
      if smpl:
        index=sample(prediction[0],temperature=temperature)
      else:
        index = np.argmax(prediction)
      
      #result = int_to_note[index]
      prediction_output.append(index)

      pattern=np.append(pattern,prediction,0)
      pattern = pattern[1:len(pattern)]
  return prediction_output

def create_midi(encoding,resolution,prediction_output,save_path='',name='test_output',return_stream=False):
    if encoding==1:
        s=create_midi1(prediction_output,save_path=save_path,name=name,resolution=resolution,return_stream=return_stream)
    if encoding==2:
        s=create_midi2(prediction_output,save_path=save_path,name=name,resolution=resolution,return_stream=return_stream)
    if encoding==4:
        s=create_midi4(prediction_output,save_path=save_path,name=name,resolution=resolution,return_stream=return_stream)
    if return_stream:
        return s


def create_midi1(prediction_output,resolution=8,save_path='',name='test_output',return_stream=False):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    if save_path=='':
      save_path=os.getcwd()
    offset = 0
    output_notes = []
    timestep=1/resolution #timestep quarterLength
    # create note and rest objects based on the values generated by the model
    for pattern in prediction_output:
      if pattern==128:
        new_rest=note.Rest()
        new_rest.offset=offset
        new_rest.duration.quarterLength=timestep
        offset+=timestep
        output_notes.append(new_rest)
      elif pattern==129:
        if len(output_notes)!=0:
          output_notes[-1].duration.quarterLength+=timestep
        offset+=timestep
      else:
        # pattern is a note
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.duration.quarterLength= timestep
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += timestep

    midi_stream = stream.Stream(output_notes)
    if return_stream:
        return midi_stream
    else:

        midi_stream.write('midi', fp=save_path+'/'+name+'.mid')
    
    
def create_midi2(prediction_output,save_path='',resolution=8,name='test_output',return_stream=False):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    if save_path=='':
      save_path=os.getcwd()
    offset = 0
    output_notes = []
    timestep=1/4 #one-sixteenth
    # create note and rest objects based on the values generated by the model
    old_pattern=132
    for pattern in prediction_output:
      if pattern==130 or pattern==131:
        continue
      if pattern==128:
        new_rest=note.Rest()
        new_rest.offset=offset
        new_rest.duration.quarterLength=timestep
        offset+=timestep
        output_notes.append(new_rest)
      elif pattern==129:
        if (old_pattern==128 or old_pattern==129):
          new_rest=note.Rest()
          new_rest.offset=offset
          new_rest.duration.quarterLength=timestep
          output_notes.append(new_rest)
        else:
          if len(output_notes)>0:
            old_note=output_notes[-1]
            old_note.duration.quarterLength+=timestep
        offset+=timestep
      else:
        # pattern is a note
        if (old_pattern==pattern):
          if len(output_notes)>0:
            old_note=output_notes[-1]
            old_note.duration.quarterLength+=timestep
        else:
          new_note = note.Note(pattern)
          new_note.offset = offset
          new_note.duration.quarterLength= timestep
          new_note.storedInstrument = instrument.Piano()
          output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += timestep
      old_pattern=pattern

    midi_stream = stream.Stream(output_notes)
    if return_stream:
        return midi_stream
    else:
        midi_stream.write('midi', fp=save_path+'/'+name+'.mid')

def create_midi4(prediction_output,resolution=8,save_path='',name='test_output',return_stream=False):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    if save_path=='':
      save_path=os.getcwd()
    offset = 0
    dur=0
    old={}
    output_notes = []
    timestep=1/4 #one-sixteenth
    # create note and rest objects based on the values generated by the model

    for pattern in prediction_output:
        if pattern<=127:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.duration.quarterLength=dur
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            dur=0
        
        elif pattern>=256:
            if len(output_notes)>0:
                dur+=(pattern-255)/resolution
                offset+=(pattern-255)/resolution
        else:
            if len(output_notes)>0:
                
            
                #if output_notes[-1].pitch.midi==pattern-128:
                output_notes[-1].duration.quarterLength=dur
            dur=0

    midi_stream = stream.Stream(output_notes)
    if return_stream:
        return midi_stream
    else:
        midi_stream.write('midi', fp=save_path+'/'+name+'.mid')

def generate(encoding,resolution,seed,seed_name,model_path='',dict=True, keep_seed=False,temperature=1):
    """ Generate a piano midi file """
    

    #loader= Data_Gen_Midi(batch_folder='npz/validate')
    #network_input = loader.__getitem__(0)[0][0]
    #seed=notes[0][0:64]
    
    experiment_path=os.path.dirname(os.path.dirname(model_path))
    output_path=experiment_path+'/output'
    filename=os.path.basename(model_path)
    filename=filename[6:23]+f'-temp_{temperature}'
    #resolution=int(experiment_path[-1])
    if dict:
      dictionary=np.load(experiment_path+'/dictionary',allow_pickle=True)
      hot_input=midi_to_onehot_dict(seed,dictionary)
    else:
      hot_input=midi_to_onehot(seed)


    model=load_model(model_path)
    prediction_output = generate_notes_hot(model, hot_input,temperature=temperature)
    
    if dict:
      dict_output=[]
      rev_dict={value:key for (key,value) in dictionary.items()}
      for i in prediction_output:
        dict_output.append(rev_dict[i])
      if keep_seed:
        dict_output=np.append(seed,dict_output)
        filename+='+seed'

      prediction_output=dict_output
    else:
      if keep_seed:
        prediction_output=np.append(seed,prediction_output)
        filename+=' + seed'
    #return prediction_output
    os.makedirs(f'{output_path}',exist_ok=True)
    create_midi(encoding,resolution,prediction_output,save_path=output_path+'/'+seed_name,name=filename)
    
def seed_to_midi(encoding,resolution,seed,seed_name,model_path=''):
    """ Generate a piano midi file """
    

    #loader= Data_Gen_Midi(batch_folder='npz/validate')
    #network_input = loader.__getitem__(0)[0][0]
    #seed=notes[0][0:64]
    
    experiment_path=os.path.dirname(os.path.dirname(model_path))
    output_path=experiment_path+'/output'+'/'+seed_name
    filename='seed_'+seed_name
    #resolution=int(experiment_path[-1])
   
    os.makedirs(f'{output_path}',exist_ok=True)
    create_midi(encoding,resolution,seed,save_path=output_path,name=filename)

def get_stats_dataset(notes_path,encoding,resolution,no_pieces,piece_length,output_path='stats'):
    notes=pd.read_pickle(notes_path)
    nnotes=[]
    for n in notes:
        if len(n)>=piece_length:
          nnotes.append(n)  
    piece_ind=np.random.choice(len(nnotes), size=no_pieces, replace=False)
    
    for i in range(no_pieces):
        if len(nnotes[piece_ind[i]])<piece_length:
            print('fuuuuuuu')
            extra=np.random.randint(0,len(notes),1)
            piece_ind=np.append(piece_ind,0)
        else:
            piece_name='piece'+str(piece_ind[i])    
            piece=nnotes[piece_ind[i]][0:piece_length]
            create_midi(encoding,resolution,piece,save_path=output_path,name=piece_name)

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




'''
if __name__=='__main__':
    notes_path='notes16/notes_event1_res8'
    encoding=4
    resolution=8
    no_pieces=100
    piece_length=200
    output_path='stats'
    get_stats_dataset(notes_path,encoding,resolution,no_pieces,piece_length,output_path='stats')
'''
'''    
if __name__=='__main__':
    model_path='experiments/max/25-07-20/text_nocomms_model_n3_s256_d0.5_sl40_bs256/models/model-010-0.4473-0.6069.h5'
    text_path='kern_text_nocomment.txt'
    tt=generate_text(model_path,text_path,0,40,100)
'''

if __name__=='__main__':
    model_path='C:/scripts/experiments/max/20-07-20/notes_event1_res8_model_n1_s32_d0.2/models/model-200-0.7731-0.8435.h5'
    notes_path='C:/scripts/notes16/notes_event1_res8'
    encoding=4
    #resolution=int(notes_path[-1])
    resolution=8
    notes=pd.read_pickle(notes_path)
    
    temperatures=[0.01,0.3,0.8,1]
    
    seed_ind=[0,1,3]
    seed_ind=range(5)
    seq_length=64
    for i,seed in enumerate ([notes_validate[i][0:seq_length] for i in seed_ind]):
        if len(seed)<seq_length:
            seed_ind.append(seed_ind[-1]+1)
            continue
        seed=seed_list[i]
        seed_name='seed'+str(seed_ind[i])  
        seed_to_midi(encoding,resolution,seed,seed_name,model_path=model_path)
        for t in temperatures:
          if t==0.01:
            generate(encoding,resolution,seed,seed_name,model_path,keep_seed=True, temperature=t,dict=True)
            continue
          generate(encoding,resolution,seed,seed_name,model_path,keep_seed=False, temperature=t,dict=True)
