# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 18:56:04 2020

@author: incog
"""
import pandas as pd
notes_path='notes16/notes_event1_res24'
notes=pd.read_pickle(notes_path)

#notes2=pd.read_pickle('notes16/notes_event1_res8')

pcount8=0
durs=[]

for piece in notes:
    durs.append(len(piece))
    for n in piece:
        if n<=127:
            pcount8+=1
            
print(notes_path)
print(len(durs))
print(pcount8)

#%%

from music21 import *

pc=0

for s in streams:
    for n in s.parts[0].flat:
        if isinstance(n,note.Note):
            pc+=1
print(pc)

#%%

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:41:33 2020

@author: incog
"""
from pandas.core.common import flatten
import numpy as np

def get_indices(notes):
    dup=np.zeros((len(notes),len(notes)))
    for ind,i in enumerate(notes):
        for ind2,j in enumerate(notes):
            if len(i)!=len(j):
                continue
            
            if all(i==j):
                dup[ind,ind2]+=1
    return dup

dup=get_indices(notes)  
dup=dup-np.eye(len(notes))          
rmv=[]
for ind,i in enumerate(dup):
    a=np.argwhere(i==1)
    if len(a)>0:
        print (ind,i,a)
    rmv+=list(flatten(a[a>ind]))

#%%

dupll=[]
for i in dup:
    if sum(i)>1:
        dupll.append(list(dup[i].nonzero()))
        
#%%

sums=np.sum(dup,axis=0)
pos=np.argwhere(sums>1)

#%%
poss=[]
for ind,row in enumerate(dup):
    if np.sum(row)>1:
        poss.append(list(np.argwhere(row==1).flatten()))
        
rmv=np.unique(np.array(poss))
rmvv=[]
for i in rmv:
    rmvv.append(i[1:])