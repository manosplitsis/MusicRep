# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:29:57 2020

@author: incog
"""
import tensorflow.keras as keras

import numpy as np
import glob
from IPython.display import clear_output

class TrainingPlot(keras.callbacks.Callback):
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()

#add this:
#plot_losses = TrainingPlot()

def midi_to_onehot(notes_midi,dim=130):
  '''transform a vector(iterable) of midi notes in a one-hot representation'''
  notes_hot=[]
  for note in notes_midi:
    temp=np.zeros(dim)
    temp[int(note)]=1
    notes_hot.append(temp)
  return np.asarray(notes_hot)

def midi_to_onehot_dict(notes_midi,dict):
  '''transform a vector(iterable) of midi notes in a one-hot representation'''
  notes_hot=[]
  dim=len(dict)
  for note in notes_midi:
    temp=np.zeros(dim)
    temp[dict[note]]=1
    notes_hot.append(temp)
  return np.asarray(notes_hot)

def set_callbacks(verbose, use_tensorboard, checkpoint_dir = "checkpoints"):
    '''Set callbacks for Keras model.
       Args:
         - use_tensorboard: (int) Add TensorBoard callback if use_tensorboard == 1
       Returns:
         - callbacks: (list) list of callbacks for model'''        
    root_dir = '..'
    checkpoint_dir = os.path.join(root_dir,
                                  checkpoint_dir, 
                                  'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    callbacks = [ModelCheckpoint(checkpoint_dir, verbose=verbose)]
    if use_tensorboard:
        log_dir = os.path.join('..', 'logs')
        tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=0.01,
                              write_images=True)
        callbacks.append(tb_callback)  

    return callbacks

#breaks the gathered notes in one .npz file per batch, to be used with a data generator
#saves 
def preprocess(notes,batch_folder='npz',batch_size=256,sequence_length=64, pad=False):
  notes_dict=keep_dataset_notes(notes)
  network_input = []
  network_output = []
  batch_no=0
  batch_ind=0
  batches=[]

  for piece in notes:
    if pad: #pads rests at the beginning and end of each piece
      start=np.zeros(sequence_length)+128
      piece=np.append(start,piece)
      piece=np.append(piece,start)
    piece_length=piece.shape[0]
    #sequence_length=int(piece_length/3)
    if piece_length<=sequence_length:
        continue
    for i in range(0, piece_length - sequence_length, 1):
      sequence_in = piece[i:i + sequence_length]
      sequence_out = piece[i + sequence_length]

      network_input.append(sequence_in)
      network_output.append(sequence_out)
      batch_ind+=1
      
      if batch_ind>=batch_size:
        batch_no+=1
        n_patterns = len(network_input)
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        network_output=np.array(network_output)
        np.savez(f'{batch_folder}/batch{batch_no:06}.npz',network_input=network_input,network_output=network_output)
        batches.append((network_input,network_output))
        batch_ind=0
        network_input = []
        network_output = []
  input_shape=np.array([batch_size,sequence_length,130])
  np.save(f'{batch_folder}/input_shape',input_shape)
  np.save(f'{batch_folder}/notes_dict',notes_dict)
  return batches

#breaks the gathered notes in one .npz file per batch, to be used with a data generator
#saves 
def preprocess_all(piece,batch_folder='npz',batch_size=256,sequence_length=64, pad=False):
  x=set(piece)
  temp=zip(x,range(len(sorted(x))))
  notes_dict=dict(temp)
  network_input = []
  network_output = []
  batch_no=0
  batch_ind=0
  batches=[]



  if pad: #pads rests at the beginning and end of each piece
    start=np.zeros(sequence_length)+128
    piece=np.append(start,piece)
    piece=np.append(piece,start)
  piece_length=piece.shape[0]
  #sequence_length=int(piece_length/3)
  for i in range(0, piece_length - sequence_length, 1):
    sequence_in = piece[i:i + sequence_length]
    sequence_out = piece[i + sequence_length]

    network_input.append(sequence_in)
    network_output.append(sequence_out)
    batch_ind+=1
    
    if batch_ind>=batch_size:
      batch_no+=1
      n_patterns = len(network_input)
      network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
      network_output=np.array(network_output)
      np.savez(f'{batch_folder}/batch{batch_no:06}.npz',network_input=network_input,network_output=network_output)
      batches.append((network_input,network_output))
      batch_ind=0
      network_input = []
      network_output = []
  input_shape=np.array([batch_size,sequence_length,132])
  np.save(f'{batch_folder}/input_shape',input_shape)
  np.save(f'{batch_folder}/notes_dict',notes_dict)
  return batches

def add_piece_start_stop(notes):
    """Add an event at the start and at the end of the piece """ 
    nnotes=[]
    for n in notes:
      n=np.append(np.array([500]),n)
      n=np.append(n,np.array([501]))
      nnotes.append(n)
    return np.array(nnotes,dtype=object)

def glue_notes(notes, add_marks=True):
  """ glues together all pieces in a big note array, if add_marks it adds start and stop marks before and after each piece"""
  all_notes=[]
  for n in notes:
    if add_marks:
        all_notes=all_notes+[350]+list(n)+[351]    
    else:
        all_notes=all_notes+list(n)
  return np.array(all_notes)

def keep_dataset_notes(notes,zero_pad=False):  
    """Returns a dictionary of all notes that appear in the dataset"""
    try:
        notes[0][0]==0
    except IndexError:
        notes=[notes]
    x=set()
    for piece in notes:
        x=x.union(set(piece))
    if zero_pad:
        temp=zip(x,range(1,len(sorted(x))+1))
    else:
        temp=zip(x,range(len(sorted(x))))
    return dict(temp)

def transpose_notes_step(enc,notes,step=1):
    '''
    transpose encoded notes a number of semitones (steps)

    '''
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

def load_doc(filename):
    """load a .txt file in memory as raw text"""
    # open the file as read only
    file = open(filename, 'r', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def removeComments(inputFileName, outputFileName):

    input = open(inputFileName, "r", encoding='latin-1')
    output = open(outputFileName, "w", encoding='latin-1')


    for line in input:
        if not (line.lstrip().startswith("!!!") or line.lstrip().startswith("!!")):
            output.write(line)

    input.close()
    output.close()
    
def get_kern_text(data_path,save_path):
    filenames=glob.glob(data_path+'/**/*.krn',recursive=True)
    with open(save_path, 'w', encoding='ANSI') as outfile:
        for fname in filenames:
            with open(fname,encoding='ANSI') as infile:
                outfile.write(infile.read())