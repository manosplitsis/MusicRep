# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 21:14:29 2020

@author: incog
"""

import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Activation, Bidirectional, Flatten, AdditiveAttention
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence

from music21 import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import pytz
from IPython.display import clear_output

import glob
import pickle

from train import get_notes
from extract_notes import get_notes_midi1,get_notes_midi2,get_notes_midi5,get_notes_event1

def get_streams(data_path='data',filetype='.krn',save=True,save_name='streams'):
    streams=[]
    file_list=glob.glob(data_path+'/**/*'+filetype,recursive=True)
    count=0
    for file in file_list:
        streams.append(converter.parse(file))
        print('Processed files:'+str(count))
        count+=1
    with open('streams/'+save_name, 'wb') as filepath:
        pickle.dump(streams, filepath)
    return streams

def grab_first_part(stream):
  try: # file has instrument parts
    s2 = instrument.partitionByInstrument(midi) #Change to only grab the piano???
    notes_to_parse = s2.parts[0].recurse() 
    if len(s2.parts)!=1:
      print('has more parts!')
  except: # file has notes in a flat structure
      print('no parts!')
      notes_to_parse = midi.flat.notes
  return notes_to_parse


def grab_first_parts(streams):
    parts=[]
    for stream in streams:
        parts.append(grab_first_part(stream))
    return parts

def transpose_streamsC(streams):
    '''
    Function that transposes all major pieces to the key of C major and minor pieces to A minor.
    To be used with Music21 streams that contain key information.

    '''
    keys=[]
    tstreams=[]
    for s in streams:
      temp=s.flat
      k=temp.getElementsByClass(key.Key)
      keys.append(k[0])
      iM = interval.Interval(k[0].tonic, pitch.Pitch('C4'))
      im = interval.Interval(k[0].tonic, pitch.Pitch('A3'))
      if k[0].mode=='minor':
        sNew = s.transpose(im)
      else:
        sNew = s.transpose(iM)
    
      tstreams.append(sNew)
    return tstreams  
      
def transpose_streams_step(streams,step):
    ttstreams=[]
    for s in tstreams:
        sNew=s.transpose(step)
        ttstreams.append(sNew)
    return ttstreams


def get_notes(encoding,data_dir='data',file_extension='.krn',resolution=8,streams=True):
    if not os.path.exists('notes'):
        os.mkdir('notes')
    path=data_dir
    if encoding==1:
        get_notes_midi1(path,resolution=resolution,streams=streams)
    elif encoding==2:
        get_notes_midi2(path,resolution=resolution,streams=streams)
    elif encoding==3:
        get_notes_midi5(path,resolution=resolution,streams=streams)    
    elif encoding==4:
        get_notes_event1(path,resolution=resolution,streams=streams)     

        
#%% 
if __name__=='__main__':
    streams=get_streams('data')
    streams=transpose_streamsC(streams)
    with open('streams/streams_C', 'wb') as filepath:
        pickle.dump(streams, filepath)
    
    
#%%         

for enc in [1,2,4]:
    for res in [4,8,16,24]:
        get_notes(enc,data_dir='C:/scripts/streams/streams_C',resolution=res,streams=True)
        
#%%

def transpose_notes_step(enc,notes,step=1):
    if enc not in [1,2,4]:
        print('Wrong encoding')
        return None
    tnotes=[]
    for song in notes:
        tsong=[]
        for n in song:
            if enc==1 or enc==2:
                if n<=(127-step):
                    tsong.append(n+step)
                else:
                    tsong.append(n)
            else:
                if n<=(255-step):
                    tsong.append(n+step)
                else:
                    tsong.append(n)
        tnotes.append(np.array(tsong))
        
    return np.array(tnotes)

notes_path='notes/tstep2'
notes_list=glob.glob(notes_path+'/*')
for n in notes_list:
    notes=pd.read_pickle(n)
    filename=os.path.basename(n)
    os.makedirs(notes_path+'/'+ filename+'_transposed',exist_ok=True)
    for step in range(12):
        tnotes=transpose_notes_step(4,notes,step)
        with open(notes_path+'/'+filename+'_transposed'+'/'+filename+'t'+str(step), 'wb') as filepath:
            pickle.dump(tnotes, filepath)
            
#%%

for res in [4,8,16,24]:
    get_notes(2,data_dir='data1',resolution=res,streams=False)
    
#%%

def get_pitch_count(stream):
    pc=0
    for n in stream:
        if isinstance(n, note.Note):
            pc+=1
    return pc

def get_note_count(piece):
    pc=0
    for n in piece:
        if n<128:
            pc+=1
    return pc

def check_notes_streams(streams,notes):
    bad=[]
    for i in range(len(streams)):
        temp=streams[i].parts[0].flat.notes
        if get_pitch_count(temp)!=get_note_count(notes[i]):
            print(i)
            bad.append(i)
    return bad
        