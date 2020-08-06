# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 19:41:35 2020

@author: incog
"""
import midi
import glob
import numpy as np
import pretty_midi
import seaborn as sns
import matplotlib.pyplot as plt
from mgeval import core, utils
from sklearn.model_selection import LeaveOneOut

#Compute features for train and generated set
set1 = glob.glob('../stats/choice6/streams_to_midi/*.mid')
num_samples = 200

set1_eval = {'total_used_pitch':np.zeros((num_samples,1)), 
             'bar_used_pitch':np.zeros((num_samples,8,1)),
             'total_used_note':np.zeros((num_samples,1)),
             'bar_used_note':np.zeros((num_samples,8,1)),
             'total_pitch_class_histogram':np.zeros((num_samples,12)),
             'bar_pitch_class_histogram':np.zeros((num_samples,8,12)),
             'pitch_class_transition_matrix':np.zeros((num_samples,12,12)),
             'pitch_range':np.zeros((num_samples,1)),
             'avg_pitch_shift':np.zeros((num_samples,1)),
             'avg_IOI':np.zeros((num_samples,1)),
             'note_length_hist':np.zeros((num_samples,12)),
             'note_length_transition_matrix':np.zeros((num_samples,12,12))
             }
metrics_list = list(set1_eval.keys())
count=0
for i in range(0, num_samples):
    feature = core.extract_feature(set1[i])
    for m in metrics_list:
        set1_eval[m][i] = getattr(core.metrics(), m)(feature)
    
set2 = glob.glob('../stats/choice6/late/notes_tstep1_res8_model_n1_s32_d0.2/*.mid')
set2_eval = {'total_used_pitch':np.zeros((num_samples,1)), 
             'bar_used_pitch':np.zeros((num_samples,8,1)),
             'total_used_note':np.zeros((num_samples,1)),
             'bar_used_note':np.zeros((num_samples,8,1)),
             'total_pitch_class_histogram':np.zeros((num_samples,12)),
             'bar_pitch_class_histogram':np.zeros((num_samples,8,12)),
             'pitch_class_transition_matrix':np.zeros((num_samples,12,12)),
             'pitch_range':np.zeros((num_samples,1)),
             'avg_pitch_shift':np.zeros((num_samples,1)),
             'avg_IOI':np.zeros((num_samples,1)),
             'note_length_hist':np.zeros((num_samples,12)),
             'note_length_transition_matrix':np.zeros((num_samples,12,12))
             }
for i in range(0, num_samples):
    feature = core.extract_feature(set2[i])
    for m in metrics_list:
        set2_eval[m][i] = getattr(core.metrics(), m)(feature)
    
#%%

print('Absolute Metrics')
print('------------------------')
for i in range(0, len(metrics_list)):
    print (metrics_list[i] + ':')
    print ('------------------------')
    print (' dataset')
    print ('  mean: ', np.mean(set1_eval[metrics_list[i]], axis=0))
    print ('  std: ', np.std(set1_eval[metrics_list[i]], axis=0))

    print ('------------------------')
    print (' event1')
    print ('  mean: ', np.mean(set2_eval[metrics_list[i]], axis=0))
    print ('  std: ', np.std(set2_eval[metrics_list[i]], axis=0))

#%%


#Cross validation for intra-set distances
loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
for i in range(len(metrics_list)):
    for train_index, test_index in loo.split(np.arange(num_samples)):
        set1_intra[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
        set2_intra[test_index[0]][i] = utils.c_dist(set2_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]][train_index])

#Cross validation for inter-set distances
loo = LeaveOneOut()
loo.get_n_splits(np.arange(num_samples))
sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))

for i in range(len(metrics_list)):
    for train_index, test_index in loo.split(np.arange(num_samples)):
        sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]])
        
#Plotting the distributions
plot_set1_intra = np.transpose(set1_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
plot_set2_intra = np.transpose(set2_intra,(1, 0, 2)).reshape(len(metrics_list), -1)
plot_sets_inter = np.transpose(sets_inter,(1, 0, 2)).reshape(len(metrics_list), -1)
for i in range(0,len(metrics_list)):
    sns.kdeplot(plot_set1_intra[i], label='intra_set1')
    sns.kdeplot(plot_sets_inter[i], label='inter')
    sns.kdeplot(plot_set2_intra[i], label='intra_set2')

    plt.title(metrics_list[i])
    plt.xlabel('Euclidean distance')
    plt.ylabel('Density')
    plt.show()

#%%


for i in range(0, len(metrics_list)):
    print (metrics_list[i] + ':')
    print ('------------------------')
    print (' tstep1')
    print ('  Kullback–Leibler divergence:',utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i]))
    print ('  Overlap area:', utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i]))
    
    print (' event1')
    print ('  Kullback–Leibler divergence:',utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i]))
    print ('  Overlap area:', utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i]))
    
#%%
kloa=np.zeros((2,2,12))
for i in range(0, len(metrics_list)):
    kloa[0,0,i]=utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])
    kloa[0,1,i]=utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
    kloa[1,0,i]=utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i])
    kloa[1,1,i]=utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i])