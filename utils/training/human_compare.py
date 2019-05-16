#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:38:18 2018

@author: wanglab
"""

from __future__ import division
import os, numpy as np, zipfile
os.chdir("/jukebox/wang/zahra/lightsheet_copy")
from tools.conv_net.functions.bipartite import pairwise_distance_metrics
from tools.utils.io import listdirfull, load_np, makedir, load_dictionary, save_dictionary
from tools.conv_net.input.read_roi import read_roi_zip

def human_compare_with_raw_rois(ann1roipth, ann2roipth, cutoff = 30):
            
    #format ZYX, and remove any rois missaved
    ann1_zyx_rois = np.asarray([[int(yy) for yy in xx.replace(".roi", "").split("-")] for xx in read_roi_zip(ann1roipth, include_roi_name=True)])
    ann2_zyx_rois = np.asarray([[int(yy) for yy in xx.replace(".roi", "").split("-")] for xx in read_roi_zip(ann2roipth, include_roi_name=True)])
        
    paired,tp,fp,fn = pairwise_distance_metrics(ann1_zyx_rois, ann2_zyx_rois, cutoff) #returns true positive = tp; false positive = fp; false negative = fn
        
    precision = tp/(tp+fp); recall = tp/(tp+fn) #calculating precision and recall
    f1 = 2*( (precision*recall)/(precision+recall) ) #calculating f1 score
    
    print ("\n   Finished calculating statistics for set params\n\n\nReport:\n***************************\n\
    Cutoff: {} \n\
    F1 score: {}% \n\
    true positives, false positives, false negatives: {} \n\
    precision: {}% \n\
    recall: {}%\n".format(cutoff, round(f1*100, 2), (tp,fp,fn), round(precision*100, 2), round(recall*100, 2)))

    return tp, fp, fn, f1

if __name__ == "__main__":
    
    #load points dict
    points_dict = load_dictionary("/home/wanglab/Documents/prv_inputs/neocortex/filename_points_dictionary.p")   
        
    print(points_dict.keys())
    #separate annotators - will have to modify conditions accordinaly
    ann1_dsets = ["zd_ann_prv_jg05_neocortex_z310-449_01.npy", 
                  "zd_ann_prv_jg24_neocortex_z300-400_01.npy", 
                  "zd_ann_prv_jg29_neocortex_z300-500_01.npy", 
                  "zd_ann_prv_jg32_neocortex_z650-810_01.npy"]

    ann2_dsets = ["cj_ann_prv_jg05_neocortex_z310-449_01.npy", 
                  "cj_ann_prv_jg24_neocortex_z300-400_01.npy", 
                  "cj_ann_prv_jg29_neocortex_z300-500_01.npy", 
                  "cj_ann_prv_jg32_neocortex_z650-810_01.npy"]

    
    #initialise empty vectors
    tps = []; fps = []; fns = []   
    
    #set voxel cutoff value
    cutoff = 30
    
    for i in range(len(ann2_dsets)):
    
        #set ground truth
        print(ann1_dsets[i])
        ann1_ground_truth = points_dict[ann1_dsets[i]]
        ann2_ground_truth = points_dict[ann2_dsets[i]]
        
        paired,tp,fp,fn = pairwise_distance_metrics(ann2_ground_truth, ann1_ground_truth, cutoff = 30) #returns true positive = tp; false positive = fp; false negative = fn
        
        #f1 per dset
        precision = tp/(tp+fp); recall = tp/(tp+fn) #calculating precision and recall
        f1 = 2*( (precision*recall)/(precision+recall) ) #calculating f1 score
        tps.append(tp); fps.append(fp); fns.append(fn) #append matrix to save all values to calculate f1 score
        
    tp = sum(tps); fp = sum(fps); fn = sum(fns) #sum all the elements in the lists
    precision = tp/(tp+fp); recall = tp/(tp+fn) #calculating precision and recall
    f1 = 2*( (precision*recall)/(precision+recall) ) #calculating f1 score
    
    print ("\n   Finished calculating statistics for set params\n\n\nReport:\n***************************\n\
    Cutoff: {} \n\
    F1 score: {}% \n\
    true positives, false positives, false negatives: {} \n\
    precision: {}% \n\
    recall: {}%\n".format(cutoff, round(f1*100, 2), (tp,fp,fn), round(precision*100, 2), round(recall*100, 2)))
