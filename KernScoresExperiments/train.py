# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 17:07:02 2020

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


from util import midi_to_onehot_dict, midi_to_onehot, set_callbacks, keep_dataset_notes, preprocess, glue_notes, add_piece_start_stop
from model import create_simple_network_func
from extract_notes import get_notes_midi1,get_notes_midi2,get_notes_midi5,get_notes_event1

class Data_Gen_Midi(Sequence):

  def __init__(self,batch_folder='npz_midi', batch_size=64, to_fit=True,shuffle=True, one_hot=True,dict=False,n_vocab=130):
      self.list_IDs = glob.glob(f"{batch_folder}/*.npz")
      self.batch_size = batch_size
      self.shuffle=shuffle
      self.to_fit=to_fit
      self.one_hot=one_hot
      self.dict=dict
      self.dictionary=np.load(f'{batch_folder}/notes_dict.npy', allow_pickle=True).item()
      self.n_vocab=n_vocab
  def __len__(self):
      # print(self.type + ' - len : ' + str(int(np.ceil(self.x.shape[0] / self.batch_size))))
      return int(np.ceil(len(self.list_IDs) ))

  def __getitem__(self, idx):
      batch_file = self.list_IDs[idx]
      batch=np.load(batch_file)
      batch_x_midi = batch['network_input']
      batch_y_midi = batch['network_output']
      if self.one_hot:
        batch_x=[]
        batch_y=[]
        if self.dict:
          for batch in batch_x_midi:
            batch_x.append(midi_to_onehot_dict(batch,self.dictionary))
          batch_y=midi_to_onehot_dict(batch_y_midi,self.dictionary)
        else:
          for batch in batch_x_midi:
            batch_x.append(midi_to_onehot(batch,dim=self.n_vocab))
          batch_y=midi_to_onehot(batch_y_midi,dim=self.n_vocab)
        batch_x=np.asarray(batch_x)
      else:
        batch_x=batch_x_midi
        batch_y=batch_y_midi

      return batch_x, batch_y

  def on_epoch_end(self):
      if self.shuffle == True:
            np.random.shuffle(self.list_IDs)
      
def get_notes(encoding,data_dir='data',file_extension='.krn',resolution=8,streams=True):
    if not os.path.exists('notes'):
        os.mkdir('notes')
    path=data_dir+'/**/*'+file_extension
    if encoding==1:
        get_notes_midi1(path,resolution=resolution,streams=streams)
    elif encoding==2:
        get_notes_midi2(path,resolution=resolution,streams=streams)
    elif encoding==3:
        get_notes_midi5(path,resolution=resolution,streams=streams)    
    elif encoding==4:
        get_notes_event1(path,resolution=resolution,streams=streams)     



def pretrain(notes_path,batch_size=256,seq_length=64,desc='',val_split=0.1, all_notes=False):
    
    notes=pd.read_pickle(notes_path)
    
    if all_notes:
        nnotes=add_piece_start_stop(notes)
        notes=glue_notes(nnotes)
        
    batch_folder=f'batches/sl{seq_length}_bs{batch_size}'+'_'+desc
    try:
        os.makedirs(batch_folder, exist_ok=True)
        os.makedirs(batch_folder+'/train', exist_ok=True)
        os.makedirs(batch_folder+'/validate', exist_ok=True)
    except:
        print('Batch folder already exists')
    
    durations=np.empty(0)
    for piece in notes:
        durations=np.append(durations,piece.shape[0])
    #notes=np.array(notes)
    inds=durations.argsort()
    durations=durations[inds]
    notes_sorted=notes[inds]
    
    notes=notes_sorted[durations>64]
    np.random.shuffle(notes)
    notes_train=notes[0:len(notes)-int(val_split*len(notes))]
    notes_validate=notes[len(notes)-int(val_split*len(notes)):len(notes)]
        

    preprocess(notes_train,batch_folder=batch_folder+'/train',sequence_length=seq_length,batch_size=batch_size)
    preprocess(notes_validate,batch_folder=batch_folder+'/validate',sequence_length=seq_length,batch_size=batch_size)
    
    return batch_folder



def train_with_loader(batch_folder='npz',one_hot=True,load=False,model_path='',desc='',dict=True):
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
  
  experiment_path='experiments/'+fday+'/'+ftime+' - '+desc
  os.makedirs(experiment_path+'/models')
  logdir=experiment_path+'/logs'
  os.makedirs(experiment_path+'/models',exist_ok=True)

  input_shape=np.load(f'{batch_folder}/train/input_shape.npy')
  batch_size=input_shape[0]
  n_vocab=130

  train_loader=Data_Gen_Midi(batch_folder=batch_folder+'/train',batch_size=batch_size,shuffle=True,one_hot=one_hot,n_vocab=n_vocab,dict=dict)
  val_loader=Data_Gen_Midi(batch_folder=batch_folder+'/validate',batch_size=batch_size,shuffle=True,one_hot=one_hot,n_vocab=n_vocab,dict=dict)

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
    model=create_simple_network_func(input_shape,n_vocab=n_vocab)

  filepath = os.path.abspath(experiment_path+'/models/model-{epoch:03d}-{loss:.4f}-{val_loss:.4f}')
  checkpoint = ModelCheckpoint(
      filepath,
      save_weights_only=False,
      period=10, #Every 10 epochs
      monitor='loss',
      verbose=2,
      save_best_only=False,
      mode='min'
  )
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=0.000001)
  plot_losses=TrainingPlot()
  callbacks_list = [checkpoint,plot_losses]
  herstory=model.fit(train_loader,validation_data=val_loader,initial_epoch=0, epochs=200, callbacks=callbacks_list)
  with open(experiment_path+'/history', 'wb') as filepath:
    pickle.dump(herstory, filepath)
  return herstory  


def train_with_loader2(batch_folder='npz',one_hot=True,load=False,model_path='',desc='',dict=True,lstm_size=32):
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
  experiment_path=os.path.join('experiments',fday,desc,'')
  logdir=os.path.join(experiment_path,'logs','')

  os.makedirs(experiment_path+'/models',exist_ok=True)
  os.makedirs(experiment_path+'/logs',exist_ok=True)

  input_shape=np.load(f'{batch_folder}/train/input_shape.npy')
  batch_size=input_shape[0]
  n_vocab=130

  train_loader=Data_Gen_Midi(batch_folder=batch_folder+'/train',batch_size=batch_size,shuffle=True,one_hot=one_hot,n_vocab=n_vocab,dict=dict)
  val_loader=Data_Gen_Midi(batch_folder=batch_folder+'/validate',batch_size=batch_size,shuffle=True,one_hot=one_hot,n_vocab=n_vocab,dict=dict)

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
    model=create_simple_network_func(input_shape,n_vocab=n_vocab,lstm_size=lstm_size)

  filepath = os.path.abspath(experiment_path+'/models/model-{epoch:03d}-{loss:.4f}-{val_loss:.4f}.h5')
  checkpoint = ModelCheckpoint(
      filepath,
      save_weights_only=False,
      period=20, #Every 10 epochs
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
                                                histogram_freq = 1,
                                                profile_batch = '500,520')
  
  callbacks_list = [checkpoint,csvlog,tensorboard_callback]
  
  #try:
      #tf.profiler.experimental.start(logdir)

  
  model.fit(train_loader,validation_data=val_loader,initial_epoch=0, epochs=200, callbacks=callbacks_list)
  #with open(experiment_path+'/history', 'wb') as filepath:
  #  pickle.dump(herstory, filepath)
  #return herstory 
 # except KeyboardInterrupt:
      
      #tf.profiler.experimental.stop()
  #tf.profiler.experimental.stop()
  """

    '''
   """

if __name__=='__main__':

    #get_notes(4,resolution=4)
    #get_notes(2,resolution=8)
    
        for notes_path in glob.glob('notes/*'):
            try:
                batch_size=256
                batch_folder=pretrain(notes_path,desc=os.path.basename(notes_path), batch_size=batch_size)
                train_with_loader2(batch_folder=batch_folder,desc=os.path.basename(batch_folder))
            except KeyboardInterrupt:
                continue