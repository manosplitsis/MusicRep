

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
from generate import generate_notes_hot,create_midi,create_midi1,create_midi4,create_midi2

def get_stats_dataset(notes_path,encoding,resolution,inds,piece_length,seq_length,late=True,model_path='',save_path='stats/choice1',keep_seed=False,temperature =0.01):
    notes=pd.read_pickle(notes_path)
    
    
    #notes_name=os.path.basename(notes_path)
    experiment_path=os.path.dirname(os.path.dirname(model_path))
    notes_name=os.path.basename(experiment_path)
    save_path+='/'+notes_name
    for i in inds:
        seed=notes[i][0:seq_length]
        
        
        filename=os.path.basename(model_path)
        filename=filename[6:23]+f'-temp_{temperature}'
        name=filename+'_'+str(i)
        #resolution=int(experiment_path[-1])
      
        dictionary=np.load(experiment_path+'/dictionary',allow_pickle=True)
        hot_input=midi_to_onehot_dict(seed,dictionary)
        model=load_model(model_path)
        
        prediction_output = generate_notes_hot(model, hot_input,temperature=temperature,smpl=True)
        dict_output=[]
        rev_dict={value:key for (key,value) in dictionary.items()}
        for i in prediction_output:
          dict_output.append(rev_dict[i])
        if keep_seed:
          dict_output=np.append(seed,dict_output)
          name='seed+'+name
        
        prediction_output=dict_output
        os.makedirs(save_path,exist_ok=True)
        st=create_midi(encoding,resolution,prediction_output,return_stream=True,save_path=save_path,name=filename)
        if late:
            count_start=32
        else:
            count_start=0
        cut_st=cut_stream(st,piece_length)
        #cut_st=stream.Stream(cut_st)
        stream_to_midi(cut_st,save_path,name)
        
        
def get_indices1(notes_tstep,resolution,no_pieces,piece_length):
    #samples from the pieces that are at least as long as the required length (e.g. 8 bars)
    #can choose a biased sample depending on dataset
    nnotes=[]
    for i,n in enumerate(notes_tstep):
        if len(n)>=piece_length:
            nnotes.append(i)
    print('Choosing from '+str(len(nnotes))+' pieces')
    piece_ind=np.random.choice(len(nnotes), size=no_pieces, replace=False)
    piece_no=0
    i=0
    output=[]
    while(piece_no<no_pieces):
        if len(notes_tstep[nnotes[piece_ind[i]]])<piece_length:
            print('fuuuuuuu')
            extra=np.random.randint(0,len(notes_tstep),1)
            piece_ind=np.append(piece_ind,extra)
        else:
            output.append(nnotes[piece_ind[i]])
            piece_no+=1
        i+=1
    return output

def get_indices2(notes_tstep,resolution,no_pieces,piece_length):
    #TODO
    #when piece is too small, concatenate with another random piece
    #also problematic as some features may be skewed a little
    piece_ind=np.random.choice(len(nnotes), size=no_pieces, replace=False)
    piece_no=0
    i=0
    output=[]
    while(piece_no<no_pieces):
        if len(notes[nnotes[piece_ind[i]]])<piece_length:
            print('fuuuuuuu')
            extra=np.random.randint(0,len(notes),1)
            piece_ind=np.append(piece_ind,extra)
        else:
            piece_no+=1
            output.append(nnotes[piece_ind[i]])
        i+=1
    return output

def cut_stream(s,piece_quarter_Length,cut_start=0):
    
    count_end=0
    count_start=0
    if cut_start>0:
        count_start=1
    s=s.flat.notes
    for obj in s:
        if not obj.offset+obj.quarterLength>cut_start:
            count_start+=1
        if obj.offset>piece_quarter_Length:
            if s[count_end-1].offset+s[count_end-1].quarterLength>piece_quarter_Length:
                s[count_end-1].quarterLength=piece_quarter_Length-s[count_end-1].offset
            return stream.Stream(s[count_start:count_end])
        count_end+=1
    return stream.Stream(s[count_start:count_end])

def stream_to_midi(stream,save_path,name):
    
    sf=stream.parts[0].flat
    for el in sf:
        if 'Instrument' in el.classes: # or 'Piano'
            sf.replace(el, instrument.Piano(),recurse=True,allDerived=True)
    sf.write('midi', fp=save_path+'/'+name+'.mid')
            
def streams_to_midi(streams,inds,save_path,length=32,cut=True):
    os.makedirs(save_path,exist_ok=True)
    for i in inds:
        if cut:
            
            st=cut_stream(streams[i],length)
        else:
            st=streams[i]
        stream_to_midi(st,save_path,'stream'+str(i))
    with open(os.path.dirname(save_path)+'/inds','wb') as filepath:
        pickle.dump(inds,filepath)

'''     
notes=pd.read_pickle('notes16/notes_tstep1_res8')
notes_tstep1=notes[len(notes)-int(0.1*len(notes)):len(notes)]
res=8
no_pieces=200
piece_len=32
inds=get_indices1(notes_tstep1,res,no_pieces,piece_len*res)
inds=np.array(inds)+len(notes)-int(0.1*len(notes))

#streams=pd.read_pickle('streams/streams')
save_path='stats/choice6'
streams_midi_path=save_path+'/streams_to_midi'
os.makedirs(streams_midi_path, exist_ok=True)
streams_to_midi(streams,inds,streams_midi_path)
'''





 #%%
save_path='stats/choice6/late'
inds=pd.read_pickle('stats/choice6/inds')
notes_path='notes/notes_tstep2_res8'
enc=2
res=8
piece_length=32
sequence_length=64
temperature=1.
keep_seed=False
model_path='experiments/max/21-07-20/notes_tstep2_res8_model_n1_s32_d0.2/models/model-200-0.4818-0.6122.h5'
get_stats_dataset(notes_path,enc,res,inds,piece_length,sequence_length,model_path=model_path,save_path=save_path,keep_seed=keep_seed,temperature=temperature)