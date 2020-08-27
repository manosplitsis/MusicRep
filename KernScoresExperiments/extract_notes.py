# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:31:32 2020

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



# Extracts notes as vectors of midi indices
#Control quantization with the resolution variable (1 for quarters, 4 for 16ths, 8 for 32nd etc)
#0-127 : note onsets
#128 : rest onset
#129 : hold the previous state
#resolution: in how many notes we slice a quarter 
def get_notes_midi1(data_path='essen/europa/**/*.krn',save=True,streams=False,file_name='notes',resolution=4):
  
  
  notes=[]
  count=0
  if streams:
    data=pd.read_pickle(data_path)
  else:
    data=glob.glob(data_path,recursive=True)


  for file in data:
    clear_output(wait=True)
    print("Parsing %s" % file)
    
    if streams:
      midi=file
    else:
      midi=converter.parse(file)
    try: # file has instrument parts
        s2 = instrument.partitionByInstrument(midi) #Change to only grab the piano???
        notes_to_parse = s2.parts[0].recurse() 
    except: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
    
    first_note_at=notes_to_parse[0].offset
    dur = int((notes_to_parse[-1].offset+notes_to_parse[-1].quarterLength)*resolution)
    notes_list=np.ones(dur,dtype='int16')*128

    
    for n in notes_to_parse:
      if isinstance(n, note.Note):
        notes_list[int(n.offset*resolution)]=int(n.pitch.midi)          
        for d in range(1,int(n.quarterLength*resolution)):
          notes_list[int(n.offset*resolution)+d]=129
      elif isinstance(n,note.Rest):
        for d in range(int(n.quarterLength*resolution)):
          notes_list[int(n.offset*resolution)+d]=128
    # add to the list  
    notes.append(notes_list)
    count+=1
    print('Parsed files:'+str(count))

  notes=np.array(notes)
  
  if save:
    savepath=os.path.join('notes',file_name+'_tstep1_res'+str(resolution))
    with open(savepath, 'wb') as filepath:
      pickle.dump(notes, filepath)
  return notes



# Extracts notes as vectors of midi indices
# 0-127 : note onsets
# 128 : rest onset
# 129 : note-off
#resolution: in how many notes we slice a quarter 
def get_notes_midi2(data_path='essen/europa/**/*.krn',save=True,streams=False,file_name='notes',resolution=4):
  
    notes=[]
    count=0
    if streams:
      data=pd.read_pickle(data_path)
    else:
      data=glob.glob(data_path,recursive=True)
  
  
    for file in data:
        clear_output(wait=True)
        print("Parsing %s" % file)
      
        if streams:
          midi=file
        else:
          midi=converter.parse(file)
        
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi) #Change to only grab the piano???
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
       
        dur = int((notes_to_parse[-1].offset+notes_to_parse[-1].quarterLength)*resolution)
        notes_list=np.ones(dur,dtype='int16')*128
        prev=128
        
        for n in notes_to_parse:
          if isinstance(n, note.Note):
              if int(n.quarterLength*resolution)<1:
                  continue
              
                  
              if n.pitch.midi==prev: #if previous note is the same as the current one
                notes_list[int(n.offset*resolution)-1]=129 #add note-off event at the final timestep that the previous note is held          
                for d in range(0,int(n.quarterLength*resolution)):
                  notes_list[int(n.offset*resolution)+d]=int(n.pitch.midi)
              else:
                for d in range(0,int(n.quarterLength*resolution)):
                  notes_list[int(n.offset*resolution)+d]=int(n.pitch.midi)
              
              prev=n.pitch.midi
          elif isinstance(n,note.Rest):
            for d in range(int(n.quarterLength*resolution)):
              notes_list[int(n.offset*resolution)+d]=128
      
        #if notes_list[-1]!=128 and notes_list[-1]!=129:
        #  notes_list=np.append(notes_list,129)
        # add to the list  
        notes.append(notes_list)
        count+=1
        print('Parsed files:'+str(count))
    
    notes=np.array(notes)
    
    if save:
      savepath=os.path.join('notes',file_name+'_tstep2_res'+str(resolution))
      with open(savepath, 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes


# Extracts notes as vectors of midi indices
#Control quantization with the resolution variable (1 for quarters, 4 for 16ths, 8 for 32nd etc)
#0-127 : note onsets
#128 : rest onset
#resolution: in how many notes we slice a quarter 
def get_notes_midi3(data_path='essen/europa/**/*.krn',save=True,streams=False,file_name='notes',resolution=4):
  
    notes=[]
    count=0
    if streams:
        data=pd.read_pickle(data_path)
    else:
        data=glob.glob(data_path,recursive=True)
      
      
    for file in data:
        clear_output(wait=True)
        print("Parsing %s" % file)
        
        if streams:
            midi=file
        else:
            midi=converter.parse(file)
        
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi) #Change to only grab the piano???
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
           
        dur = int((notes_to_parse[-1].offset+notes_to_parse[-1].quarterLength)*resolution)
        notes_list=np.ones(dur,dtype='int16')*128
        
        
        for n in notes_to_parse:
            if isinstance(n, note.Note):
                notes_list[int(n.offset*resolution)]=int(n.pitch.midi)
        # add to the list  
        notes.append(notes_list)
        count+=1
        print('Parsed files:'+str(count))

    if save:
        savepath=os.path.join('notes',file_name+'_tstep3_res'+str(resolution))
        with open(savepath, 'wb') as filepath:
            pickle.dump(notes, filepath)
    return notes


# Extracts notes as vectors of midi indices
#Control quantization with the resolution variable (1 for quarters, 4 for 16ths, 8 for 32nd etc)
#0-127 : note onsets
#128 : rest onset
#Notes that are held for many timesteps are repeated
#resolution: in how many notes we slice a quarter 
def get_notes_midi4(data_path='essen/europa/**/*.krn',save=True,streams=False,file_name='notes_tstep4',resolution=4):
  
    notes=[]
    count=0
    if streams:
      data=pd.read_pickle(data_path)
    else:
      data=glob.glob(data_path,recursive=True)
      
      
    for file in data:
        clear_output(wait=True)
        print("Parsing %s" % file)
        
        if streams:
          midi=file
        else:
          midi=converter.parse(file)
          
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi) #Change to only grab the piano???
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
           
        dur = int((notes_to_parse[-1].offset+notes_to_parse[-1].quarterLength)*resolution)
        notes_list=np.ones(dur,dtype='int16')*128
        
        
        for n in notes_to_parse:
          if isinstance(n, note.Note):      
            for d in range(0,int(n.quarterLength*resolution)):
              notes_list[int(n.offset*resolution)+d]=int(n.pitch.midi)
          elif isinstance(n,note.Rest):
            for d in range(int(n.quarterLength*resolution)):
              notes_list[int(n.offset*resolution)+d]=128
        # add to the list  
        notes.append(notes_list)
        count+=1
        print('Parsed files:'+str(count))
  
    notes=np.array(notes)
    
    if save:
      savepath=os.path.join('notes',file_name+'_res'+str(resolution))
      with open(savepath, 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes

# Extracts notes as vectors of midi indices
# 0-127 : note onsets
# 128 : rest onset
# 129 : note-off
#resolution: in how many notes we slice a quarter 
def get_notes_midi5(data_path='essen/europa/**/*.krn',save=True,streams=False,file_name='notes_tstep5',resolution=4):
  
  
    notes=[]
    count=0
    if streams:
      data=pd.read_pickle(data_path)
    else:
      data=glob.glob(data_path,recursive=True)
      
      
    for file in data:
        clear_output(wait=True)
        print("Parsing %s" % file)
        
        if streams:
          midi=file
        else:
          midi=converter.parse(file)
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi) #Change to only grab the piano???
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
           
        dur = int((notes_to_parse[-1].offset+notes_to_parse[-1].quarterLength)*resolution)
        notes_list=np.ones(dur,dtype='int16')*128
        
        
        for n in notes_to_parse:
          if isinstance(n, note.Note):       
            notes_list[int(n.offset*resolution)+int(n.quarterLength*resolution)-1]=int(n.pitch.midi)+129 #add note-off event at the final timestep that the note is held          
            for d in range(0,int(n.quarterLength*resolution)-1):
              notes_list[int(n.offset*resolution)+d]=int(n.pitch.midi) 
          elif isinstance(n,note.Rest):
            for d in range(int(n.quarterLength*resolution)):
              notes_list[int(n.offset*resolution)+d]=128
        # add to the list  
        notes.append(notes_list)
        count+=1
        print('Parsed files:'+str(count))

    notes=np.array(notes)
    
    if save:
      savepath=os.path.join('notes',file_name+'_res'+str(resolution))
      with open(savepath, 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes

# Extracts notes as vectors of midi indices
#Control quantization with the resolution variable (1 for quarters, 4 for 16ths, 8 for 32nd etc)
#0-127 : note onsets
#128 : rest onset
#129 : hold the previous state


def get_notes_event1(data_path='essen/europa/**/*.krn',save=True,streams=False,file_name='notes', resolution=8):
  
    notes=[]
    count=0
    if streams:
      data=pd.read_pickle(data_path)
    else:
      data=glob.glob(data_path,recursive=True)
      
      
    for file in data:
        clear_output(wait=True)
        print("Parsing %s" % file)
        
        if streams:
          midi=file
        else:
          midi=converter.parse(file)
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi) #Change to only grab the piano???
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
           
        dur = int((notes_to_parse[-1].offset+notes_to_parse[-1].quarterLength))
        notes_list=[]
        
        prev=0
        for n in notes_to_parse:
            
          #if isinstance(n, bar.Barline):
                #notes_list.append(500)
          if isinstance(n, note.Note):
              if n.offset>prev:
                  notes_list.append(note_length_event(n.offset-prev,resolution))
              if n.offset>=prev:
                  notes_list.append(int(n.pitch.midi))
                  notes_list.append(note_length_event(n.quarterLength,resolution))        
                  notes_list.append(int(n.pitch.midi)+128)
                  prev=n.offset+n.quarterLength
          if isinstance(n,note.Rest):
              if n.offset>prev:
                  notes_list.append(note_length_event(n.offset-prev,resolution))
              if n.offset>=prev:
                  notes_list.append(int(n.pitch.midi))
                  notes_list.append(note_length_event(n.quarterLength,resolution))        
                  notes_list.append(int(n.pitch.midi)+128)
                  prev=n.offset+n.quarterLength
          if isinstance(n, chord.Chord):
            if n.offset>prev:
              notes_list.append(note_length_event(n.offset-prev,resolution))
            for p in n:
              notes_list.append(int(p.pitch.midi))
            notes_list.append(note_length_event(n.quarterLength,resolution))
            for p in n:
              notes_list.append(int(p.pitch.midi)+128)
            prev=n.offset
        # add to the list  
        notes.append(np.array(notes_list, dtype='int16'))
        count+=1
        print('Parsed files:'+str(count))

    notes=np.array(notes)
    
    if save:
      savepath=os.path.join('notes',file_name+'_event1_res'+str(resolution))
      with open(savepath, 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes

def get_notes_event2(data_path='essen/europa/**/*.krn',save=True,streams=False,file_name='notes', resolution=8):
    """Only one note-off (129), only works with monophonic data. Also adding barlines"""
    notes=[]
    count=0
    if streams:
      data=pd.read_pickle(data_path)
    else:
      data=glob.glob(data_path,recursive=True)
      
      
    for file in data:
        clear_output(wait=True)
        print("Parsing %s" % file)
        
        if streams:
          midi=file
        else:
          midi=converter.parse(file)
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi) #Change to only grab the piano???
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
           
        dur = int((notes_to_parse[-1].offset+notes_to_parse[-1].quarterLength))
        notes_list=[]
        
        prev=0
        for n in notes_to_parse:
            if isinstance(n, note.Note):
                if n.offset>prev:
                    notes_list.append(note_length_event(n.offset-prev,resolution))
                if n.offset>=prev:
                    notes_list.append(int(n.pitch.midi))
                    notes_list.append(note_length_event(n.quarterLength,resolution))        
                    notes_list.append(129)
                    prev=n.offset+n.quarterLength
            
            if isinstance(n, chord.Chord):
                if n.offset>prev:
                  notes_list.append(note_length_event(n.offset-prev,resolution))
                for p in n:
                  notes_list.append(int(p.pitch.midi))
                notes_list.append(note_length_event(n.quarterLength,resolution))
                for p in n:
                  notes_list.append(129)
                prev=n.offset
        # add to the list  
        notes.append(np.array(notes_list, dtype='int16'))
        count+=1
        print('Parsed files:'+str(count))

    notes=np.array(notes)
    
    if save:
      savepath=os.path.join('notes',file_name+'_event1_res'+str(resolution))
      with open(savepath, 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes


def note_length_event(quarterLength,resolution=4,maxlen=10):
  quantize=1/resolution
  if quarterLength<0:
    print('oops')
  if quarterLength<=quantize:
    return 256
  if quarterLength>quantize and quarterLength<=maxlen:
    return 255+np.ceil(quarterLength/quantize)
  if quarterLength>10:
    return 255+np.ceil(maxlen/quantize)+1

def get_notes(encoding,data_dir='data',file_extension='.krn',resolution=8,streams=False):
    if not os.path.exists('notes'):
        os.mkdir('notes')
    #path=data_dir+'/**/*'+file_extension
    path=data_dir
    if encoding==1:
        get_notes_midi1(path,resolution=resolution,streams=streams)
    elif encoding==2:
        get_notes_midi2(path,resolution=resolution,streams=streams)
    elif encoding==3:
        get_notes_midi5(path,resolution=resolution,streams=streams)    
    elif encoding==4:
        get_notes_event1(path,resolution=resolution,streams=streams)     

if __name__=='__main__':
    get_notes(2,data_dir='streams/st',resolution=8,streams=True)

'''
if __name__=='__main__':
    for enc in [1,2,4]:
        for res in [4,8,16,24]:
            get_notes(enc,data_dir='data1',resolution=res,streams=False)
'''