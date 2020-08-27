# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:43:37 2020

@author: incog
"""

import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Activation, Bidirectional, Flatten, AdditiveAttention
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
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


from util import midi_to_onehot_dict, midi_to_onehot, set_callbacks, keep_dataset_notes, preprocess, glue_notes, add_piece_start_stop,load_doc,removeComments, get_kern_text
from model import create_simple_network_func, build_model
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

def get_samples_text(text,seq_length,step=1):
    sentences = []
    #next_chars = []
    for i in range(0, len(text) - seq_length, step):
        sentences.append(text[i: i + seq_length+1])
        #next_chars.append(text[i + seq_length])
    print('nb sequences:', len(sentences))
    return sentences


class Data_Gen_Midi(Sequence):

  def __init__(self,notes, batch_size=64,seq_length=64, to_fit=True,shuffle=True, one_hot=True,dict=True,n_vocab=130):
      
      #self.list_IDs = sorted(glob.glob(f"{batch_folder}/**/*.krn"))
      self.samples=get_fsamples(notes,seq_length=seq_length)
      self.batches=samples_to_batches(self.samples,batch_size)
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
      
      batch=self.batches[idx]
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
            self.batches=samples_to_batches(self.samples,self.batch_size)




class Data_Gen_Text(Sequence):

  def __init__(self,text, batch_size=64,seq_length=64, to_fit=True,shuffle=True, one_hot=True,n_vocab=130):
      
      #self.list_IDs = sorted(glob.glob(f"{batch_folder}/**/*.krn"))
      self.sentences=get_samples_text(text,seq_length)
      #self.batches=samples_to_batches2(self.sentences,batch_size)
      self.batch_size = batch_size
      self.seq_length=seq_length
      self.shuffle=shuffle
      chars = sorted(list(set(text)))
      self.one_hot=one_hot
      
      self.dictionary=dict((c, i) for i, c in enumerate(chars))
      self.n_vocab=n_vocab
  def __len__(self):
      # print(self.type + ' - len : ' + str(int(np.ceil(self.x.shape[0] / self.batch_size))))
      return int(len(self.sentences)/self.batch_size)

  def __getitem__(self, idx):
      
      batch=self.sentences[self.batch_size*idx:self.batch_size*(idx+1)]
      batch_x_text =np.array([batch[i][0:self.seq_length] for i in range(self.batch_size)])
      batch_y_text =np.array([batch[i][self.seq_length] for i in range(self.batch_size)])
      if self.one_hot:
        batch_x=[]
        batch_y=[]
        
        for seq in batch_x_text:
          batch_x.append(midi_to_onehot_dict(seq,self.dictionary))
        batch_y=midi_to_onehot_dict(batch_y_text,self.dictionary)
        batch_x=np.array(batch_x)
      else:
        batch_x=batch_x_text
        batch_y=batch_y_text

      return batch_x, batch_y

  def on_epoch_end(self):
      if self.shuffle == True:
            np.random.shuffle(self.sentences)
            #self.batches=samples_to_batches2(self.sentences,self.batch_size)
      
def get_notes(encoding,data_dir='data',file_extension='.krn',resolution=8,streams=False):
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



def train_with_loader(batch_folder='npz',load=False,model_path='',desc='',dict=True):
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

  train_loader=Data_Gen_Midi(batch_folder=batch_folder+'/train',batch_size=batch_size,shuffle=True,n_vocab=n_vocab,dict=dict)
  val_loader=Data_Gen_Midi(batch_folder=batch_folder+'/validate',batch_size=batch_size,shuffle=True,n_vocab=n_vocab,dict=dict)

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


def train_with_loader2(notes_path,batch_size,seq_length,load=False,all_notes=False,model_path='',desc='',dict=True,lstm_size=32,lstm_no=1,dropout=0.2,learning_rate=0.001):
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
  print('Notes read')
  if all_notes:
      notes=glue_notes(notes,add_marks=True)
  print('Notes glued')
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

  train_loader=Data_Gen_Midi(notes_train,batch_size=batch_size,seq_length=seq_length,shuffle=True,n_vocab=n_vocab,dict=dict)
  val_loader=Data_Gen_Midi(notes_validate,batch_size=batch_size,seq_length=seq_length,shuffle=True,n_vocab=n_vocab,dict=dict)

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
    model=build_model(input_shape[0], input_shape[1], n_vocab, lstm_no=lstm_no,lstm_size=lstm_size,dropout_rate=dropout)
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

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
                                                profile_batch = 10000)
  
  callbacks_list = [checkpoint,csvlog,tensorboard_callback]
  
  
  model.fit(train_loader,validation_data=val_loader,initial_epoch=0, epochs=200, callbacks=callbacks_list)
  
def train_stateful(notes_path,batch_size,seq_length,load=False,all_notes=False,model_path='',desc='',dict=True,lstm_size=32,lstm_no=1,dropout=0.2):
    notes_name=os.path.basename(notes_path)
    notes=pd.read_pickle(notes_path)
    notes=glue_notes(notes,add_marks=True)
    dictionary=keep_dataset_notes(notes)
    n_vocab=len(dictionary)
    input_shape[2]=n_vocab
    with open(experiment_path+'/dictionary', 'wb') as filepath:
      pickle.dump(dictionary, filepath)
    input_shape=np.array([batch_size,seq_length,n_vocab])
    samples=get_fsamples(notes,seq_length=seq_length)
    network_input=np.reshape(samples,(len(samples),seq_length,n_vocab))
    
    if load:
        model=load_model(model_path)
    else:
        #model=create_simple_network_func(input_shape,n_vocab=n_vocab,lstm_size=lstm_size)
        model=build_state_model(input_shape[0], input_shape[1], n_vocab, lstm_no=lstm_no,lstm_size=lstm_size,dropout_rate=dropout)
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    model.fit(network_input,initial_epoch=0, epochs=200)


def train_text(text_path,batch_size,seq_length,load=False,model_path='',lstm_no=1,lstm_size=32,dropout=0.2):
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
    text=load_doc(text_path)
    
    val_split=0.1
    text_train=text[0:len(text)-int(val_split*len(text))]
    text_validate=text[len(text)-int(val_split*len(text)):len(text)]
    
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    n_vocab=len(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    
    # cut the text in semi-redundant sequences of seq_length characters
    
    train_loader=Data_Gen_Text(text_train,batch_size=batch_size,seq_length=seq_length,n_vocab=n_vocab)
    val_loader=Data_Gen_Text(text_validate,batch_size=batch_size,seq_length=seq_length,n_vocab=n_vocab)

    tdict=train_loader.dictionary
    vdict=val_loader.dictionary
    tdict.update(vdict)
    val_loader.dictionary=tdict
    dictionary=tdict
    n_vocab=len(dictionary)
    
    def on_epoch_end(epoch, _):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)
    
        start_index = random.randint(0, len(text) - seq_length - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)
    
            generated = ''
            sentence = text[start_index: start_index + seq_length]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)
    
            for i in range(400):
                x_pred = np.zeros((1, seq_length, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.
    
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
    
                sentence = sentence[1:] + next_char
    
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    
    model=build_model(batch_size, seq_length, n_vocab,lstm_size=lstm_size,lstm_no=lstm_no,dropout_rate=dropout)
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    
    model.fit(train_loader,validation_data=val_loader,initial_epoch=0, epochs=200,callbacks=[print_callback])

if __name__=='__main__':
    get_kern_text('data1','kern_text.txt')
    removeComments('kern_text.txt', 'kern_text_nocomment.txt')
    text_path='C:/scripts/kern_text_nocomment.txt'
    train_text(text_path,256,35,lstm_no=3,lstm_size=256,dropout=0.5)