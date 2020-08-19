# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:43:37 2020

@author: Manos Plitsis
"""

import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Activation, Bidirectional, Flatten, AdditiveAttention,TimeDistributed
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


#from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()

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
from model import create_simple_network_func, build_model,build_model2,build_model2_emb, build_model3
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
    print(len(samples))
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
        self.dictionary=keep_dataset_notes(notes)
        self.samples=get_fsamples(notes,seq_length=seq_length)
        del notes
        self.sample_ids=np.arange(len(self.samples))
        self.batch_size = batch_size
        self.seq_length=seq_length
        self.shuffle=shuffle
        self.to_fit=to_fit
        self.one_hot=one_hot
        self.dict=dict
        
        self.n_vocab=n_vocab
    def __len__(self):
        return int(len(self.samples)/self.batch_size)
    
    def __getitem__(self, idx):
        batch_ids=self.sample_ids[self.batch_size*idx:self.batch_size*(idx+1)]
        batch=self.samples[batch_ids]
        #batch=self.samples[self.batch_size*idx:self.batch_size*(idx+1)]
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
          np.random.shuffle(self.sample_ids)
          #np.random.shuffle(self.samples)
  
class Data_Gen_Midi2(Sequence):

    def __init__(self,notes, dictionary,batch_size=64,seq_length=64,to_fit=True,shuffle=True, one_hot=False,glue_notes=False):
        
        self.samples=get_fsamples(notes,seq_length=seq_length)
        del notes
        self.sample_ids=np.arange(len(self.samples))
        self.batch_size = batch_size
        self.seq_length=seq_length
        self.shuffle=shuffle
        self.to_fit=to_fit
        self.one_hot=one_hot
        self.dictionary=dictionary
        self.n_vocab=len(dictionary)
        
    def __len__(self):
        return int(len(self.samples)/self.batch_size)

    def __getitem__(self, idx):
        batch_ids=self.sample_ids[self.batch_size*idx:self.batch_size*(idx+1)]
        batch=self.samples[batch_ids]
        #batch=self.samples[self.batch_size*idx:self.batch_size*(idx+1)]
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
            np.random.shuffle(self.sample_ids)
            #np.random.shuffle(self.samples)
  
class Data_Gen_Midi3(Sequence):

    def __init__(self,notes,dictionary, batch_size=64,shuffle=False):
        self.notes=notes
        self.note_ids=np.arange(len(notes))        
        if not shuffle:
            self.notes.sort(key=lambda x: len(x), reverse=True)
        else:
            np.random.shuffle(self.note_ids)
        self.no_batches=int(len(notes)/batch_size)
        #self.batches=[notes[i:i+batch_size] for i in range(self.no_batches)]
        self.shuffle=shuffle
        self.batch_size = batch_size
        self.dictionary=dictionary
        
    def __len__(self):
        return self.no_batches

    def __getitem__(self, idx):
        batch_ids=self.note_ids[self.batch_size*idx:self.batch_size*(idx+1)]
        batch=self.notes[batch_ids]
        #batch=self.notes[self.batch_size*idx:self.batch_size*(idx+1)]
        batch_x_midi =np.array([batch[i][:-1] for i in range(self.batch_size)])
        batch_y_midi =np.array([batch[i][1:] for i in range(self.batch_size)])
        batch_x_midi=[[self.dictionary[i] for i in piece]for piece in batch_x_midi]
        batch_y_midi=[[self.dictionary[i] for i in piece]for piece in batch_y_midi]
        batch_x_midi=pad_sequences(batch_x_midi, maxlen=None, dtype="int16", padding="post", value=0)
        batch_y_midi=pad_sequences(batch_y_midi, maxlen=None, dtype="int16", padding="post", value=0)
        
        return batch_x_midi, batch_y_midi    

    def on_epoch_end(self):
        if self.shuffle == True:
              np.random.shuffle(self.note_ids)


def schedule2(epoch):
            if epoch <= 20:
                 new_lr = .003
            elif epoch > 20:
                 new_lr = .003 * 0.97 **(epoch-20)
            
            
            print("\nLR at epoch {} = {}  \n".format(epoch,new_lr))
            return new_lr


def train_with_loader(notes_path,batch_size,seq_length,epochs=50,load=False,all_notes=False,model_path='',lstm_size=32,lstm_no=1,dropout=0.2,learning_rate=0.0001):
    #date to be used for archiving model and training history
    date=datetime.datetime.utcnow()
    gdate=date.astimezone(pytz.timezone('Europe/Athens'))
    fdate=gdate.strftime('%d-%m-%y %H:%M')
    fday=gdate.strftime('%d-%m-%y')
    ftime=gdate.strftime('%H_%M')
    print( fday)
    print(ftime)
    notes_name=os.path.basename(notes_path)
    notes=pd.read_pickle(notes_path)
    notes=add_piece_start_stop(notes)
    
    print('Notes read')
    
    model_info=f'_model_n{lstm_no}_s{lstm_size}_d{dropout}_sl{seq_length}_bs{batch_size}'
    if all_notes:
        experiment_path=os.path.join('experiments','seq_corpus','MIDI',notes_name+model_info+'run_0')
    else:
        experiment_path=os.path.join('experiments','seq_song','MIDI',notes_name+model_info+'run_0')
    run=0
    while os.path.exists(experiment_path):
        run+=1
        experiment_path=experiment_path[:-1]+str(run)
    logdir=os.path.join(experiment_path,'logs','')
    os.makedirs(experiment_path+'/models',exist_ok=True)
    os.makedirs(experiment_path+'/logs',exist_ok=True)
    

    dictionary=keep_dataset_notes(notes,zero_pad=False)
    n_vocab=len(dictionary)
    with open(experiment_path+'/dictionary', 'wb') as filepath:
          pickle.dump(dictionary, filepath)
    val_split=0.1
    
    if all_notes:
        notes_train=notes[0:len(notes)-int(val_split*len(notes))]
        notes_validate=notes[len(notes)-int(val_split*len(notes)):len(notes)]
        del notes
        notes_train=glue_notes(notes_train,add_marks=True)
        notes_validate=glue_notes(notes_validate,add_marks=True)
        print('Notes glued')
    else:
        notes_train=notes[0:len(notes)-int(val_split*len(notes))]
        notes_validate=notes[len(notes)-int(val_split*len(notes)):len(notes)]
        del notes
    
    train_loader=Data_Gen_Midi2(notes_train,dictionary,batch_size=batch_size,seq_length=seq_length,shuffle=True,glue_notes=glue_notes)
    val_loader=Data_Gen_Midi2(notes_validate,dictionary,batch_size=batch_size,seq_length=seq_length,shuffle=True,glue_notes=glue_notes)
    del notes_train
    del notes_validate

    

    

    
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
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                patience=5, min_lr=0.000001)
    
    csvlog=tf.keras.callbacks.CSVLogger(experiment_path+'/logs.csv', separator=",", append=False)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir,
                                                  histogram_freq = 0)
    
    callbacks_list = [checkpoint,csvlog,tensorboard_callback]
    
    if load:
        model=load_model(model_path)
    else:
        #model=create_simple_network_func(input_shape,n_vocab=n_vocab,lstm_size=lstm_size)
        model=build_model2_emb(batch_size, n_vocab, lstm_no=lstm_no,lstm_size=lstm_size,dropout_rate=dropout)
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        model.summary()
    model.fit(train_loader,validation_data=val_loader,initial_epoch=0, epochs=epochs, callbacks=callbacks_list,verbose=1)
    
def train_with_loader2(notes_path,batch_size,epochs=50,load=False,all_notes=False,model_path='',lstm_size=32,lstm_no=1,dropout=0.2,learning_rate=0.0001,lr_schedule=False,shuffle=False):
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
    pieces=pd.read_pickle(notes_path)
    #notes=notes[0:19200]
    pieces=add_piece_start_stop(pieces)
    pieces=list(pieces)
    #notes.sort(key=lambda x: len(x), reverse=True)
    #notes=notes[batch_size*2:] #delete the first two (biggest) batches to save memory in gpu
    dictionary=keep_dataset_notes(pieces,zero_pad=True)
    
    n_vocab=len(dictionary)
    print('Notes read')
    
    model_info=f'_model_n{lstm_no}_s{lstm_size}_d{dropout}_bs{batch_size}'
    if shuffle:
        model_info+='_shuffle'
    experiment_path=os.path.join('experiments','folkrnn','MIDI',notes_name+model_info+'run_0')
    
    run=0
    while os.path.exists(experiment_path):
        run+=1
        experiment_path=experiment_path[:-1]+str(run)
    logdir=os.path.join(experiment_path,'logs','')
      
    os.makedirs(experiment_path+'/models',exist_ok=True)
    os.makedirs(experiment_path+'/logs',exist_ok=True)
    
    
    with open(experiment_path+'/dictionary', 'wb') as filepath:
      pickle.dump(dictionary, filepath)
    
    
    pieces_c=pieces[:22925]
    pieces_csharp=pieces[22925:]
    del pieces
    
    val_split=0.1
    pieces_train_c=pieces_c[0:len(pieces_c)-int(val_split*len(pieces_c))]
    pieces_validate_c=pieces_c[len(pieces_c)-int(val_split*len(pieces_c)):len(pieces_c)]
    pieces_train_csharp=pieces_csharp[0:len(pieces_csharp)-int(val_split*len(pieces_csharp))]
    pieces_validate_csharp=pieces_csharp[len(pieces_csharp)-int(val_split*len(pieces_csharp)):len(pieces_csharp)]
    pieces_train=pieces_train_c+pieces_train_csharp
    pieces_validate=pieces_validate_c+pieces_validate_csharp
    del pieces_c,pieces_csharp
    
    #notes_train=notes[0:len(notes)-int(val_split*len(notes))]
    #notes_validate=notes[len(notes)-int(val_split*len(notes)):len(notes)]
    #del notes
    
    train_loader=Data_Gen_Midi3(pieces_train,dictionary,batch_size=batch_size,shuffle=shuffle)
    val_loader=Data_Gen_Midi3(pieces_validate,dictionary,batch_size=batch_size,shuffle=shuffle)
    #del notes_train
    #del notes_validate

    if load:
      model=load_model(model_path)
      model.layers[0].trainable=False
    else:
      model=build_model3(batch_size,n_vocab, lstm_no=lstm_no,lstm_size=lstm_size,dropout_rate=dropout)
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
      #optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate,clipnorm=5)
      
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
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                                patience=5, min_lr=0.000001)
    
    csvlog=tf.keras.callbacks.CSVLogger(experiment_path+'/logs.csv', separator=",", append=False)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir,
                                                  histogram_freq = 1,
                                                  profile_batch = '20,40',
                                                  write_graph=False,
                                                  embeddings_freq=0,
                                                  write_images=True)
    
    earlystop=tf.keras.callbacks.EarlyStopping(patience=2)
    callbacks_list = [checkpoint,csvlog,earlystop,tensorboard_callback]
    
    if lr_schedule:
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule2)
        callbacks_list=[checkpoint,csvlog,tensorboard_callback,lr_scheduler,earlystop]
    
    #model.fit(train_loader,validation_data=val_loader,initial_epoch=0, epochs=100, callbacks=callbacks_list,verbose=2)
    #model.fit(train_loader,initial_epoch=0, epochs=100, callbacks=callbacks_list,verbose=1)
    model.fit(train_loader,validation_data=val_loader,initial_epoch=0,callbacks=callbacks_list, epochs=epochs,verbose=1)
    
    

#%%
'''
if __name__=='__main__':
    notes_path='notes/notes_tstep1_res8'
    model_path='experiments/15-08-20/notes_event1_res8_model_n1_s32_d0.2_sl64_bs64run_4/models/model-020-2.3585-2.6278'
    for batch_size in [128,64,32]:
        for lstm_no in [2,3]:
            for lstm_size in [64]:
                try:
                    train_with_loader2(notes_path, batch_size,epochs=200,lstm_no=lstm_no,lstm_size=lstm_size,dropout=0.5,all_notes=False,lr_schedule=False,shuffle=False)
                except:
                    continue
    
    #train_with_loader(notes_path, batch_size,seq_length=seq_length,lstm_no=1,lstm_size=32,dropout=0.2,all_notes=False)
    #train_with_loader(notes_path, batch_size,seq_length=seq_length,lstm_no=2,lstm_size=256,dropout=0.5,all_notes=False)
    
'''
if __name__=='__main__':
    notes_path='notes/notes_event1_res8_44'
    train_with_loader(notes_path, 256, 100,epochs=200,lstm_size=32,lstm_no=1,dropout=0.2)
    notes_path='notes/notes_tstep1_res8_44'
    train_with_loader(notes_path, 256, 100,epochs=200,lstm_size=32,lstm_no=1,dropout=0.2)
    notes_path='notes/notes_tstep1_res8_44'
    train_with_loader(notes_path, 256, 100,epochs=200,lstm_size=32,lstm_no=1,dropout=0.2,all_notes=True)
    notes_path='notes/notes_event1_res8_44'
    train_with_loader(notes_path, 256, 100,epochs=200,lstm_size=32,lstm_no=1,dropout=0.2,all_notes=True)
