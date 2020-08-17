# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 19:37:27 2020

@author: incog
"""
from util import load_doc
def abc_addCounters(text,save_path):
    newtxt=''
    count=1
    newtxt+='X:0\n'
    for line in text.splitlines():
        newtxt+=line+'\n'
        if len(line)==0:
            newtxt+='X:'+str(count)+'\n'
            count+=1
    with open(save_path) as infile:
        infile.write(newtxt)
    return newtxt


def count_char(text,tokenized=True):
    count = {}
    if tokenized:
        text=text.split()
    for ch in text:
      
        # If char already in dictionary increment count
        # otherwise add char as key and 1 as value
        if ch in count:
          count[ch] += 1
        else:
          count[ch] = 1
    for k, v in count.items():
      print('Charcater {} occurs {} times'.format(k,v))
    return {k: v for k, v in sorted(count.items(), key=lambda item: item[1])}

def add_trackStartStop(text,save_path):
    newtxt=''
    count=0
    chars_in_tune=0
    char=[]
    next_tune=1
    newtxt+='<s>\n'
    for line in text.splitlines():
        for ch in line.split():
            chars_in_tune+=1
        if len(line)==0:
            newtxt+='</s>\n\n<s>'
            count+=1
            char.append(chars_in_tune)
            chars_in_tune=0
        newtxt+=line+'\n'
    print('Songs in file:',count)
    print('Average chars in tune:', sum(char)/len(char))
    with open(save_path, 'w') as infile:
        infile.write(newtxt)
    return newtxt,char

def tokenized_toNormal(text):
    newtxt=''
    for line in text.splitlines():
        temp=''.join(line.split())
        newtxt+=temp+'\n'
    return newtxt

def count_timeSigs(text):
    count = {}
    for line in text.splitlines():
        if line.startswith('M:'):
            if line in count:
                count[line]+=1
            else:
                count[line]=1
    for k, v in count.items():
      print('Time Signature {} occurs {} times'.format(k,v))
    return count

def count_keys(text):
    count = {}
    for line in text.splitlines():
        if line.startswith('K:'):
            if line in count:
                count[line]+=1
            else:
                count[line]=1
    for k, v in count.items():
      print('Key/Mode {} occurs {} times'.format(k,v))
    return count

def keep_only_with(text,string):
    pieces=text.split('\n\n')
    del text
    keepers=[]
    for piece in pieces:
        for line in piece.splitlines():
            if line.startswith(string):
                keepers.append(piece)
                continue
    return keepers

def glue_pieces(pieces):
    text=''
    for i in pieces:
        text+=i+'\n\n'
    return text

    
def correct_keys(text):
    newtxt=''
    for line in text.splitlines():
        if line.startswith('K:maj'):
            newline='K:Cmaj'
        elif line.startswith('K:dor'):
            newline='K:Ddor'
        elif line.startswith('K:min'):
            newline='K:Amin'
        elif line.startswith('K:mix'):
            newline='K:Gmix'
        else:
            newline=line
        newtxt+=newline+'\n'
    return newtxt
#text=load_doc('data_v3_startstop')
#cc=correct_keys(text)

def correct_hornpipe(text):
    newtext=''
    for line in text.splitlines():
        newline=''
        for ch in line.split():
            if ch in ['<','>','<s>','</s>']:
                newline+=' '+ch+' '
            elif '<' in ch:
                temp=ch.split('<')
                newline+=temp[0]+' '
                newline+=' '+'<'+' '
            elif '>' in ch:
                temp=ch.split('>')
                newline+=' '+ temp[0]+' '
                newline+=' ' +'>'+' '
            else:
                newline+=' '+ch+' '
        newtext+=newline[1:]+'\n'
    return newtext
            