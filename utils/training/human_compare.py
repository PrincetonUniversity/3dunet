#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:38:18 2018

@author: wanglab
"""

from __future__ import division
import os
os.chdir("/jukebox/wang/zahra/lightsheet_copy")
from tools.conv_net.functions.bipartite import pairwise_distance_metrics
from tools.utils.io import load_dictionary

if __name__ == "__main__":
    
    #load points dict
    points_dict = load_dictionary("/jukebox/wang/zahra/conv_net/annotations/prv/filename_points_dictionary_quality_test.p")
    
    #get dsets to compare
    dsets = ["JGANNOTATION_20180305_jg_bl6f_prv_12_647_010na_7d5um_250msec_10povlp_ch00_C00_400-440_01.npy",
             "JGANNOTATION_20180305_jg_bl6f_prv_12_647_010na_7d5um_250msec_10povlp_ch00_C00_400-440_02.npy",
             "JGANNOTATION_20180306_jg_bl6f_prv_16_647_010na_7d5um_250msec_10povlp_ch00_C00_Z0450-0500_01.npy",
             "JGANNOTATION_20180305_jg_bl6f_prv_12_647_010na_7d5um_250msec_10povlp_ch00_C00_400-440_03.npy",
             "JGANNOTATION_20180306_jg_bl6f_prv_16_647_010na_7d5um_250msec_10povlp_ch00_C00_Z0450-0500_02.npy"
                ]
    
    #separate annotators - will have to modify conditions accordinaly
    ann1_dsets = dsets
    ann2_dsets = [xx for xx in points_dict.keys() if xx not in ann1_dsets and any(xx in yy for yy in ann1_dsets)]
    
    #initialise empty vectors
    tps = []; fps = []; fns = []   
    
    for dset in ann2_dsets:
    
        #set ground truth
        ann1_ground_truth = points_dict["JGANNOTATION_"+ dset]
        ann2_ground_truth = points_dict[dset]
        
        paired,tp,fp,fn = pairwise_distance_metrics(ann2_ground_truth, ann1_ground_truth, cutoff = 30) #returns true positive = tp; false positive = fp; false negative = fn
           
        tps.append(tp); fps.append(fp); fns.append(fn) #append matrix to save all values to calculate f1 score
        
    tp = sum(tps); fp = sum(fps); fn = sum(fns) #sum all the elements in the lists
    precision = tp/(tp+fp); recall = tp/(tp+fn) #calculating precision and recall
    f1 = 2*( (precision*recall)/(precision+recall) ) #calculating f1 score
    
    print ('\n   Finished calculating statistics for set params\n\n\nReport:\n***************************\n\
    F1 score: {}% \n\
    true positives, false positives, false negatives: {} \n\
    precision: {}% \n\
    recall: {}%\n'.format(round(f1*100, 2), (tp,fp,fn), round(precision*100, 2), round(recall*100, 2)))

