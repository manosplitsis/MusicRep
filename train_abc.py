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






def get_samples_text(text,seq_length,step=1):
    sentences = []
    #next_chars = []
    for i in range(0, len(text) - seq_length, step):
        sentences.append(text[i: i + seq_length+1])
        #next_chars.append(text[i + seq_length])
    print('nb sequences:', len(sentences))
    return sentences

def get_samples_text_pieces(notes,seq_length,step=1):
    samples=[]
    small_pieces=0
    try:
        notes[0][0]==0
    except IndexError:
        notes=[notes]
    for piece in notes:
        
        piece_length=len(piece)
        if piece_length<=seq_length:
            small_pieces+=1
            continue
        for i in range(0, piece_length - seq_length, 1):
            sequence_in = piece[i:i + seq_length+1]
            #sequence_out = piece[i + seq_length]
      
            samples.append(sequence_in)
    print('nb sequences:', len(samples))
    print('too small pieces: ',small_pieces)            
    return samples


class Data_Gen_Text(Sequence):

  def __init__(self,text, batch_size=64,seq_length=64, to_fit=True,shuffle=True, one_hot=False,n_vocab=130,dictionary={}):
      
      #self.list_IDs = sorted(glob.glob(f"{batch_folder}/**/*.krn"))
      self.sentences=get_samples_text(text,seq_length)
      
      print('nuber of sequences: ',len(self.sentences))
      #self.batches=samples_to_batches2(self.sentences,batch_size)
      self.batch_size = batch_size
      self.seq_length=seq_length
      self.shuffle=shuffle
      chars = sorted(list(set(text)))
      self.one_hot=one_hot
      
      #self.dictionary=dict((c, i) for i, c in enumerate(chars))
      self.dictionary=dictionary
      self.n_vocab=n_vocab
  def __len__(self):
      # print(self.type + ' - len : ' + str(int(np.ceil(self.x.shape[0] / self.batch_size))))
      return int(len(self.sentences)/self.batch_size)

  def __getitem__(self, idx):
      
      batch=self.sentences[self.batch_size*idx:self.batch_size*(idx+1)]
      batch_x_text =[batch[i][0:self.seq_length] for i in range(self.batch_size)]
      batch_y_text =[batch[i][1:] for i in range(self.batch_size)]
      if self.one_hot:
        batch_x=[]
        batch_y=[]
        
        for seq in batch_x_text:
          batch_x.append(midi_to_onehot_dict(seq,self.dictionary))
        for seq in batch_y_text:
          batch_y.append(midi_to_onehot_dict(seq,self.dictionary))
        
        batch_x=np.array(batch_x)
        batch_y=np.array(batch_y)
      else:
        batch_x=[[self.dictionary[i] for i in batch_x_text[j]] for j in range(self.batch_size)]
        batch_y=[[self.dictionary[i] for i in batch_y_text[j]] for j in range(self.batch_size)]

      return np.array(batch_x),np.array( batch_y)

  def on_epoch_end(self):
      if self.shuffle == True:
            np.random.shuffle(self.sentences)
            #self.batches=samples_to_batches2(self.sentences,self.batch_size)
            
            
class Data_Gen_Text_Pieces(Sequence):

  def __init__(self,pieces, batch_size=64,seq_length=64, to_fit=True,shuffle=True, one_hot=False,n_vocab=130,dictionary={}):
      
      #self.list_IDs = sorted(glob.glob(f"{batch_folder}/**/*.krn"))
      
      self.sentences=get_samples_text_pieces(pieces,seq_length)
      
      print('nuber of sequences: ',len(self.sentences))
      #self.batches=samples_to_batches2(self.sentences,batch_size)
      self.batch_size = batch_size
      self.seq_length=seq_length
      self.shuffle=shuffle
      self.one_hot=one_hot
      self.dictionary=dictionary
      self.n_vocab=n_vocab
      
  def __len__(self):
      # print(self.type + ' - len : ' + str(int(np.ceil(self.x.shape[0] / self.batch_size))))
      return int(len(self.sentences)/self.batch_size)

  def __getitem__(self, idx):
      
      batch=self.sentences[self.batch_size*idx:self.batch_size*(idx+1)]
      batch_x_text =[batch[i][0:self.seq_length] for i in range(self.batch_size)]
      batch_y_text =[batch[i][1:] for i in range(self.batch_size)]
      if self.one_hot:
        batch_x=[]
        batch_y=[]
        
        for seq in batch_x_text:
          batch_x.append(midi_to_onehot_dict(seq,self.dictionary))
        for seq in batch_y_text:
          batch_y.append(midi_to_onehot_dict(seq,self.dictionary))
        
        batch_x=np.array(batch_x)
        batch_y=np.array(batch_y)
      else:
        batch_x=[[self.dictionary[i] for i in batch_x_text[j]] for j in range(self.batch_size)]
        batch_y=[[self.dictionary[i] for i in batch_y_text[j]] for j in range(self.batch_size)]

      return np.array(batch_x),np.array( batch_y)

  def on_epoch_end(self):
      if self.shuffle == True:
            np.random.shuffle(self.sentences)
            #self.batches=samples_to_batches2(self.sentences,self.batch_size)
      





def train_abc(text_path,batch_size,seq_length,load=False,model_path='',lstm_no=1,lstm_size=32,dropout=0.2,epochs=200):
    #date to be used for archiving model and training history
    date=datetime.datetime.utcnow()
    gdate=date.astimezone(pytz.timezone('Europe/Athens'))
    fdate=gdate.strftime('%d-%m-%y %H:%M')
    fday=gdate.strftime('%d-%m-%y')
    ftime=gdate.strftime('%H_%M')
    print( fday)
    print(ftime)
    model_info=f'_model_n{lstm_no}_s{lstm_size}_d{dropout}_sl{seq_length}_bs{batch_size}'
    text_name=os.path.basename(text_path)
    experiment_path=os.path.join('experiments',fday,text_name+model_info,'')
    logdir=os.path.join(experiment_path,'logs','')
    os.makedirs(experiment_path+'/models',exist_ok=True)
    os.makedirs(experiment_path+'/logs',exist_ok=True)
    text=load_doc(text_path)
    #text=text[0:int(len(text)/4)]
    text=text.split()
    #pieces=text.split('\n\n')
    #for i,piece in enumerate(pieces):
    #    pieces[i]=piece.split()
    
    val_split=0.1
    #pieces_train=pieces[0:len(pieces)-int(val_split*len(pieces))]
    #pieces_validate=pieces[len(pieces)-int(val_split*len(pieces)):len(pieces)]
    text_train=text[0:len(text)-int(val_split*len(text))]
    text_validate=text[len(text)-int(val_split*len(text)):len(text)]
    
    chars = sorted(list(set(text.split())))
    print('total chars:', len(chars))
    n_vocab=len(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    
    # cut the text in semi-redundant sequences of seq_length characters
    
    train_loader=Data_Gen_Text(text_train,batch_size=batch_size,seq_length=seq_length,n_vocab=n_vocab,dictionary=char_indices)
    val_loader=Data_Gen_Text(text_validate,batch_size=batch_size,seq_length=seq_length,n_vocab=n_vocab,dictionary=char_indices)

    
    dictionary=char_indices
    n_vocab=len(dictionary)
    with open(experiment_path+'/dictionary', 'wb') as filepath:
      pickle.dump(dictionary, filepath)
    
    
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
    # learning rate scheduler
    def schedule(epoch):
        if epoch < 5:
             new_lr = .003
        elif epoch >= 5:
             new_lr = 0.003 * (epoch-4) ** 0.97
        
        
        print("\nLR at epoch {} = {}  \n".format(epoch,new_lr))
        return new_lr
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

    csvlog=tf.keras.callbacks.CSVLogger(experiment_path+'/logs.csv', separator=",", append=False)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir,
                                                  histogram_freq = 1,
                                                  profile_batch = '2,2000')
    callbacks_list=[checkpoint,tensorboard_callback,csvlog,lr_scheduler]
    
    model=build_model2_emb(batch_size, seq_length, n_vocab,lstm_size=lstm_size,lstm_no=lstm_no,dropout_rate=dropout)
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    model.summary()
    
    model.fit(train_loader,validation_data=val_loader,initial_epoch=0, epochs=epochs,callbacks=callbacks_list)
    
def train_abc2(text_path,batch_size,seq_length,load=False,model_path='',lstm_no=1,lstm_size=32,dropout=0.2,epochs=200):
    #date to be used for archiving model and training history
    date=datetime.datetime.utcnow()
    gdate=date.astimezone(pytz.timezone('Europe/Athens'))
    fdate=gdate.strftime('%d-%m-%y %H:%M')
    fday=gdate.strftime('%d-%m-%y')
    ftime=gdate.strftime('%H_%M')
    print( fday)
    print(ftime)
    model_info=f'_model_n{lstm_no}_s{lstm_size}_d{dropout}_sl{seq_length}_bs{batch_size}'
    text_name=os.path.basename(text_path)
    experiment_path=os.path.join('experiments',fday,text_name+model_info,'')
    logdir=os.path.join(experiment_path,'logs','')
    os.makedirs(experiment_path+'/models',exist_ok=True)
    os.makedirs(experiment_path+'/logs',exist_ok=True)
    text=load_doc(text_path)
    
    
    chars = sorted(list(set(text.split())))
    print('total chars:', len(chars))
    n_vocab=len(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    
    
    #text=text[0:int(len(text)/4)]
    #text=text.split()
    pieces=text.split('\n\n')
    del text
    for i,piece in enumerate(pieces):
        pieces[i]=piece.split()
    
    val_split=0.1
    pieces_train=pieces[0:len(pieces)-int(val_split*len(pieces))]
    pieces_validate=pieces[len(pieces)-int(val_split*len(pieces)):len(pieces)]
    #text_train=text[0:len(text)-int(val_split*len(text))]
    #text_validate=text[len(text)-int(val_split*len(text)):len(text)]
    
    
    
    # cut the text in semi-redundant sequences of seq_length characters
    
    train_loader=Data_Gen_Text_Pieces(pieces_train,batch_size=batch_size,seq_length=seq_length,n_vocab=n_vocab,dictionary=char_indices)
    val_loader=Data_Gen_Text_Pieces(pieces_validate,batch_size=batch_size,seq_length=seq_length,n_vocab=n_vocab,dictionary=char_indices)

    
    dictionary=char_indices
    n_vocab=len(dictionary)
    with open(experiment_path+'/dictionary', 'wb') as filepath:
      pickle.dump(dictionary, filepath)
    

    
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
    # learning rate scheduler
    def schedule(epoch):
        if epoch < 5:
             new_lr = .003
        elif epoch >= 5:
             new_lr = 0.003 * (epoch-4) ** 0.97
        
        
        print("\nLR at epoch {} = {}  \n".format(epoch,new_lr))
        return new_lr
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)


    csvlog=tf.keras.callbacks.CSVLogger(experiment_path+'/logs.csv', separator=",", append=False)
    
    
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir,
                                                  histogram_freq = 1,
                                                  profile_batch = '200,400')
    callbacks_list=[checkpoint,tensorboard_callback,csvlog,lr_scheduler]
    if load:
        print('loading model')
        model=load_model(model_path)
        model.layers[0].trainable=False
    else:
        model=build_model2_emb(batch_size, seq_length, n_vocab,lstm_size=lstm_size,lstm_no=lstm_no,dropout_rate=dropout)
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    model.summary()
    model.fit(train_loader,validation_data=val_loader,initial_epoch=0, epochs=epochs,callbacks=callbacks_list)
    
def train_abc3(text_path,batch_size,seq_length,load=False,model_path='',lstm_no=1,lstm_size=32,dropout=0.2,epochs=200,split_pieces=False):
    #date to be used for archiving model and training history
    date=datetime.datetime.utcnow()
    gdate=date.astimezone(pytz.timezone('Europe/Athens'))
    fdate=gdate.strftime('%d-%m-%y %H:%M')
    fday=gdate.strftime('%d-%m-%y')
    ftime=gdate.strftime('%H_%M')
    print( fday)
    print(ftime)
    model_info=f'_model_n{lstm_no}_s{lstm_size}_d{dropout}_sl{seq_length}_bs{batch_size}'
    text_name=os.path.basename(text_path)
    experiment_path=os.path.join('experiments',fday,text_name+model_info,'')
    logdir=os.path.join(experiment_path,'logs','')
    os.makedirs(experiment_path+'/models',exist_ok=True)
    os.makedirs(experiment_path+'/logs',exist_ok=True)
    text=load_doc(text_path)
    chars = sorted(list(set(text.split())))
    print('total chars:', len(chars))
    n_vocab=len(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    #indices_char = dict((i, c) for i, c in enumerate(chars))
    with open(experiment_path+'/dictionary', 'wb') as filepath:
      pickle.dump(char_indices, filepath)
    val_split=0.1

    if split_pieces:
    
        pieces=text.split('\n\n')
        del text
        for i,piece in enumerate(pieces):
            pieces[i]=piece.split()
        
        pieces_train=pieces[0:len(pieces)-int(val_split*len(pieces))]
        pieces_validate=pieces[len(pieces)-int(val_split*len(pieces)):len(pieces)]
        train_loader=Data_Gen_Text_Pieces(pieces_train,batch_size=batch_size,seq_length=seq_length,n_vocab=n_vocab,dictionary=char_indices)
        val_loader=Data_Gen_Text_Pieces(pieces_validate,batch_size=batch_size,seq_length=seq_length,n_vocab=n_vocab,dictionary=char_indices)

    else:
        #text=text[0:int(len(text)/4)]
        text=text.split()
        text_train=text[0:len(text)-int(val_split*len(text))]
        text_validate=text[len(text)-int(val_split*len(text)):len(text)]
        train_loader=Data_Gen_Text(text_train,batch_size=batch_size,seq_length=seq_length,n_vocab=n_vocab,dictionary=char_indices)
        val_loader=Data_Gen_Text(text_validate,batch_size=batch_size,seq_length=seq_length,n_vocab=n_vocab,dictionary=char_indices)

    
    
    
    

    
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
    # learning rate scheduler
    def schedule(epoch):
        if epoch < 5:
             new_lr = .003
        elif epoch >= 5:
             new_lr = 0.003 * 0.97 ** (epoch-4) 
        
        
        print("\nLR at epoch {} = {}  \n".format(epoch,new_lr))
        return new_lr
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)


    csvlog=tf.keras.callbacks.CSVLogger(experiment_path+'/logs.csv', separator=",", append=False)
    
    
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir,
                                                  histogram_freq = 1,
                                                  profile_batch = '200,400')
    callbacks_list=[checkpoint,tensorboard_callback,csvlog,lr_scheduler]
    if load:
        print('loading model')
        model=load_model(model_path)
    else:
        model=build_model2_emb(batch_size, seq_length, n_vocab,lstm_size=lstm_size,lstm_no=lstm_no,dropout_rate=dropout)
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    model.summary()
    model.fit(train_loader,validation_data=val_loader,initial_epoch=0, epochs=epochs,callbacks=callbacks_list)
#%%


if __name__=='__main__':
    #get_kern_text('data1','kern_text.txt')
    #removeComments('kern_text.txt', 'kern_text_nocomment.txt')
    text_path='data_v3_startstop'
    model_path='experiments/13-08-20/data_v3_startstop_model_n3_s256_d0.2_sl64_bs256/models/model-006-0.9513-1.2259'
    train_abc3(text_path,256,64,lstm_no=1,lstm_size=32,dropout=0.2,epochs=100)
    
    