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
    points_dict = load_dictionary("/home/wanglab/Documents/prv_inputs/filename_points_dictionary.p")   
    
    #separate annotators - will have to modify conditions accordinaly
    ann1_dsets = [xx for xx in points_dict.keys() if xx[:2] == "cj"]; ann1_dsets.sort()
    ann2_dsets = [xx for xx in points_dict.keys() if xx not in ann1_dsets]; ann2_dsets.sort()
    
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
        print(f1)
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
#    
#    annpth = "/home/wanglab/mounts/wang/zahra/conv_net/annotations/prv/volumes_zd_made/all_annotations"
#    
#    anns = listdirfull(annpth)
#    
#    ann1pths = [xx for xx in anns if os.path.basename(xx)[:2] == "cj" and "neocortex" in os.path.basename(xx)]; ann1pths.sort()
#    
#    ann2pths = [xx for xx in anns if xx not in ann1pths and "neocortex" in os.path.basename(xx)]; ann2pths.sort()
#    
#    tps = []; fps = []; fns = []
#    
#    #set voxel cutoff value
#    cutoff = 30
#    
#    for i in range(len(ann1pths)):
#        
#        ann1roipth = ann1pths[i]; ann2roipth = ann2pths[i]
#        
#        tp, fp, fn = human_compare_with_raw_rois(ann1roipth, ann2roipth)
#        
#        tps.append(tp); fps.append(fp); fns.append(fn) #append matrix to save all values to calculate f1 score
#        
#    tp = sum(tps); fp = sum(fps); fn = sum(fns) #sum all the elements in the lists
#    precision = tp/(tp+fp); recall = tp/(tp+fn) #calculating precision and recall
#    f1 = 2*( (precision*recall)/(precision+recall) ) #calculating f1 score
#    
#    print ("\n   Finished calculating FINAL statistics for set params\n\n\nReport:\n***************************\n\
#    Cutoff: {} \n\
#    F1 score: {}% \n\
#    true positives, false positives, false negatives: {} \n\
#    precision: {}% \n\
#    recall: {}%\n".format(cutoff, round(f1*100, 2), (tp,fp,fn), round(precision*100, 2), round(recall*100, 2)))
#    
    
    
    
    