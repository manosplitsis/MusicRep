# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:43:37 2020

@author: incog
"""

import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Activation, Bidirectional, Flatten, AdditiveAttention,TimeDistributed
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.utils import Sequence


#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

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


from util import midi_to_onehot_dict, midi_to_onehot, set_callbacks, keep_dataset_notes, preprocess, glue_notes, add_piece_start_stop,load_doc
from model import create_simple_network_func, build_model,build_model2,build_model2_emb
from extract_notes import get_notes_midi1,get_notes_midi2,get_notes_midi5,get_notes_event1


def get_samples(notes,seq_length=64):
    samples=[]
    try:
        notes[0][0]==0
    except IndexError:
        notes=[notes]
    for piece in notes:
        piece_length=piece.shape[0]
        x=[]
        
        if piece_length<=seq_length:
            print('smol')
            continue
        for i in range(0, piece_length - seq_length, 1):
            sequence_in = piece[i:i + seq_length+1]
            #sequence_out = piece[i + seq_length]
      
            x.append(np.array(sequence_in,dtype='int16'))
            
        samples.append(np.array(x))
    return np.array(samples)

def get_fsamples(notes,seq_length=64):
    samples=[]
    try:
        notes[0][0]==0
    except IndexError:
        notes=[notes]
    for piece in notes:
        
        piece_length=len(piece)
        x=[]
        
        if piece_length<=seq_length:
            continue
        for i in range(0, piece_length - seq_length, 1):
            sequence_in = piece[i:i + seq_length+1]
            #sequence_out = piece[i + seq_length]
      
            samples.append(np.array(sequence_in,dtype='int16'))
    print('nb sequences:', len(samples))            
    return np.array(samples)



def samples_to_batches(samples,batch_size):
    batches=[]
    batch=[]
    count=0
    for sample in samples:
        batch.append(sample)
        count+=1
        if count>=batch_size:
            count=0
            batches.append(np.array(batch))
            batch=[]
    return np.array(batches)

def samples_to_batches2(samples,batch_size):
    batches=[]
    batch=[]
    count=0
    for index,sample in enumerate(samples):
        batch.append(index)
        count+=1
        if count>=batch_size:
            count=0
            batches.append(np.array(batch))
            batch=[]
    return np.array(batches)



class Data_Gen_Midi(Sequence):
    
    def __init__(self,notes, batch_size=64,seq_length=64, to_fit=True,shuffle=True, one_hot=True,dict=True,n_vocab=130):
    
        #self.list_IDs = sorted(glob.glob(f"{batch_folder}/**/*.krn"))
        self.samples=get_fsamples(notes,seq_length=seq_length)
        #self.batches=samples_to_batches(self.samples,batch_size)
        self.batch_size = batch_size
        self.seq_length=seq_length
        self.shuffle=shuffle
        self.to_fit=to_fit
        self.one_hot=one_hot
        self.dict=dict
        self.dictionary=keep_dataset_notes(notes)
        self.n_vocab=n_vocab
    def __len__(self):
        # print(self.type + ' - len : ' + str(int(np.ceil(self.x.shape[0] / self.batch_size))))
        return int(len(self.samples)/self.batch_size)
    
    def __getitem__(self, idx):
        batch=self.samples[self.batch_size*idx:self.batch_size*(idx+1)]
        #batch=self.batches[idx]
        batch_x_midi =np.array([batch[i][0:self.seq_length] for i in range(self.batch_size)])
        batch_y_midi =np.array([batch[i][self.seq_length] for i in range(self.batch_size)])
        if self.one_hot:
          batch_x=[]
          batch_y=[]
          if self.dict:
            for seq in batch_x_midi:
              batch_x.append(midi_to_onehot_dict(seq,self.dictionary))
            batch_y=midi_to_onehot_dict(batch_y_midi,self.dictionary)
          else:
            for seq in batch_x_midi:
              batch_x.append(midi_to_onehot(seq,dim=self.n_vocab))
            batch_y=midi_to_onehot(batch_y_midi,dim=self.n_vocab)
          batch_x=np.asarray(batch_x)
        else:
          batch_x=batch_x_midi
          batch_y=batch_y_midi
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
      if self.shuffle == True:
            np.random.shuffle(self.samples)
            #self.batches=samples_to_batches(self.samples,self.batch_size)
  
class Data_Gen_Midi2(Sequence):

    def __init__(self,notes, batch_size=64,seq_length=64, to_fit=True,shuffle=True, one_hot=False,dict=True,n_vocab=130,glue_notes=False):
        
        #self.list_IDs = sorted(glob.glob(f"{batch_folder}/**/*.krn"))
        
        if glue_notes:
            self.samples=get_fsamples(notes,seq_length=seq_length)
        else:
            self.samples=get_fsamples(notes,seq_length=seq_length)
        #self.batches=samples_to_batches(self.samples,batch_size)
        self.batch_size = batch_size
        self.seq_length=seq_length
        self.shuffle=shuffle
        self.to_fit=to_fit
        self.one_hot=one_hot
        self.dict=dict
        self.dictionary=keep_dataset_notes(notes)
        self.n_vocab=n_vocab
        
    def __len__(self):
        # print(self.type + ' - len : ' + str(int(np.ceil(self.x.shape[0] / self.batch_size))))
        return int(len(self.samples)/self.batch_size)

    def __getitem__(self, idx):
        batch=self.samples[self.batch_size*idx:self.batch_size*(idx+1)]
        #batch=self.batches[idx]
        batch_x_midi =np.array([batch[i][0:self.seq_length] for i in range(self.batch_size)])
        batch_y_midi =np.array([batch[i][1:] for i in range(self.batch_size)])
        if self.one_hot:
            batch_x=[]
            batch_y=[]
            if self.dict:
                for seq in batch_x_midi:
                    batch_x.append(midi_to_onehot_dict(seq,self.dictionary))
                for seq in batch_y_midi:
                    batch_y.append(midi_to_onehot_dict(seq,self.dictionary))
            else:
                for seq in batch_x_midi:
                    batch_x.append(midi_to_onehot(seq,dim=self.n_vocab))
                for seq in batch_y_midi:
                    batch_y.append(midi_to_onehot(seq,dim=self.n_vocab))
            
            batch_x=np.asarray(batch_x)
            batch_y=np.asarray(batch_y)
        else:
            batch_x=np.array([[self.dictionary[i] for i in batch_x_midi[j]] for j in range(self.batch_size)])
            batch_y=np.array([[self.dictionary[i] for i in batch_y_midi[j]] for j in range(self.batch_size)])
    
        return batch_x, batch_y    

    def on_epoch_end(self):
        if self.shuffle == True:
              np.random.shuffle(self.samples)
              #self.batches=samples_to_batches(self.samples,self.batch_size)


def train_with_loader(notes_path,batch_size,seq_length,load=False,all_notes=False,model_path='',desc='',dict=True,lstm_size=32,lstm_no=1,dropout=0.2,learning_rate=0.001):
  #date to be used for archiving model and training history
  date=datetime.datetime.utcnow()
  gdate=date.astimezone(pytz.timezone('Europe/Athens'))
  fdate=gdate.strftime('%d-%m-%y %H:%M')
  fday=gdate.strftime('%d-%m-%y')
  ftime=gdate.strftime('%H_%M')
  print( fday)
  print(ftime)
  #os.mkdir('/experiments/{fday}')
  #os.mkdir('/experiments/{fday}/{ftime} - {desc}')
  notes_name=os.path.basename(notes_path)
  notes=pd.read_pickle(notes_path)
  notes=notes[0:int(len(notes)/10)]
  notes=add_piece_start_stop(notes)
  print('Notes read')
  
  model_info=f'_model_n{lstm_no}_s{lstm_size}_d{dropout}_sl{seq_length}_bs{batch_size}'
  experiment_path=os.path.join('experiments',fday,notes_name+model_info,'')
  logdir=os.path.join(experiment_path,'logs','')

  os.makedirs(experiment_path+'/models',exist_ok=True)
  os.makedirs(experiment_path+'/logs',exist_ok=True)

  input_shape=np.array([batch_size,seq_length,130])
  n_vocab=130
  
  val_split=0.1
  notes_train=notes[0:len(notes)-int(val_split*len(notes))]
  notes_validate=notes[len(notes)-int(val_split*len(notes)):len(notes)]
  if all_notes:
      notes_train=glue_notes(notes_train,add_marks=True)
      notes_validate=glue_notes(notes_validate,add_marks=True)
  print('Notes glued')

  train_loader=Data_Gen_Midi2(notes_train,batch_size=batch_size,seq_length=seq_length,shuffle=True,n_vocab=n_vocab,dict=dict,glue_notes=glue_notes)
  val_loader=Data_Gen_Midi2(notes_validate,batch_size=batch_size,seq_length=seq_length,shuffle=True,n_vocab=n_vocab,dict=dict,glue_notes=glue_notes)

  if dict:
    tdict=train_loader.dictionary
    vdict=val_loader.dictionary
    tdict.update(vdict)
    val_loader.dictionary=tdict
    dictionary=tdict
    n_vocab=len(dictionary)
    input_shape[2]=n_vocab
    with open(experiment_path+'/dictionary', 'wb') as filepath:
      pickle.dump(dictionary, filepath)
    

  if load:
    model=load_model(model_path)
  else:
    #model=create_simple_network_func(input_shape,n_vocab=n_vocab,lstm_size=lstm_size)
    model=build_model2_emb(input_shape[0], input_shape[1], n_vocab, lstm_no=lstm_no,lstm_size=lstm_size,dropout_rate=dropout)
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

  model.summary()
  filepath = os.path.abspath(experiment_path+'/models/model-{epoch:03d}-{loss:.4f}-{val_loss:.4f}')
  checkpoint = ModelCheckpoint(
      filepath,
      save_weights_only=False,
      period=2, #Every 10 epochs
      monitor='loss',
      verbose=2,
      save_best_only=False,
      mode='min'
  )

#define callbacks
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=0.000001)
  #plot_losses=TrainingPlot()
  
  csvlog=tf.keras.callbacks.CSVLogger(experiment_path+'/logs.csv', separator=",", append=False)
  
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir,
                                                histogram_freq = 1)
  
  callbacks_list = [checkpoint,csvlog,tensorboard_callback]
  
  
  model.fit(train_loader,validation_data=val_loader,initial_epoch=0, epochs=200, callbacks=callbacks_list)
  

#%%

if __name__=='__main__':
    notes_path='notes/notes_tstep1_res8'
    res=8
    enc=1
    batch_size=256
    seq_length=64
    train_with_loader(notes_path, batch_size, seq_length,lstm_no=3,lstm_size=256,dropout=0.5,all_notes=False)
