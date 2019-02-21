#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:39:37 2018

@author: wanglab
"""

from __future__ import division
import os, numpy as np, sys, multiprocessing as mp, time, matplotlib.pyplot as plt, pandas as pd
from scipy import ndimage
from scipy.integrate import simps
from skimage.external import tifffile
from scipy.ndimage.morphology import generate_binary_structure
import h5py
os.chdir("/jukebox/wang/zahra/lightsheet_copy")
from tools.utils.io import load_dictionary, load_np
from tools.conv_net.functions.bipartite import pairwise_distance_metrics

    
def probabiltymap_to_centers_thresh(src, threshold = (0.1,1), numZSlicesPerSplit = 200, overlapping_planes = 40, cores = 4, return_pixels = False, verbose = False, structure_rank_order = 2):
    '''
    by tpisano
    
    Function to take probabilty maps generated by run_cnn and output centers based on center_of_mass

    Inputs:
    --------------
    memmapped_paths: (optional, str, or list of strs) pth(s) to memmapped array
    threshold: (tuple) lower and upper bounds to keep. e.g.: (0.1, 1) - note this assumes input from a CNN and thus data will
    numZSlicesPerSplit: chunk of zplanes to process at once. Adjust this and cores based on memory constraints.
    cores: number of parallel jobs to do at once. Adjust this and numZSlicesPerSplit based on memory constraints
    overlapping_planes: number of planes on each side to overlap by, this should be comfortably larger than the maximum z distances of a single object
    structure_rank_order: Optional. If true provides the structure element to used in ndimage.measurements.labels, 2 seems to be the most specific
    save (optional) 'True', 'False', str of path and file name to save with extension .p. If multiple cell /jukebox/LightSheetTransfer/cnn/zmd/20180929_395000chkpnt_xy160z20/channels.
    return_pixels, if True return centers and all pixels associated with that center

    Returns single list of
    ------------
    centers: list of zyx coordinates of centers of mass
    (IF RETURN PIXELS = True, dictionary consisting of k=centers, v=indices determined by cnn with k's center)
    save_location (if saving)

    OUTPUTS ZYX

    '''
    #handle inputs
    if type(src) == str:
        if src[-2:] == 'h5':
            f = h5py.File(src)
            src = f['/main'].value
            f.close()
        elif src[-3:] == 'tif': src = tifffile.imread(src)
        elif src[-3:] == 'npy': src = load_np(src)
        
    src = np.squeeze(src)    
    zdim, ydim, xdim = src.shape

    #run
    if cores > 1: 
        start = time.time()
        if verbose: sys.stdout.write('\n   Thesholding, determining connected pixels, identifying center of masses\n\n'); sys.stdout.flush()
        p = mp.Pool(cores)
        iterlst=[(src, z, numZSlicesPerSplit, overlapping_planes, threshold, return_pixels, structure_rank_order) for z in range(0, zdim, numZSlicesPerSplit)]
        centers = p.map(helper_labels_centerofmass_thresh, iterlst)
        p.terminate()
    else:
        start = time.time()
        if verbose: sys.stdout.write('\n   Thesholding, determining connected pixels, identifying center of masses\n\n'); sys.stdout.flush()
        iterlst=[(src, z, numZSlicesPerSplit, overlapping_planes, threshold, return_pixels, structure_rank_order) for z in range(0, zdim, numZSlicesPerSplit)]
        centers = []
        for i in iterlst: 
            centers.append(helper_labels_centerofmass_thresh(i))
        

    #unpack
    if not return_pixels: centers = [zz for xx in centers for zz in xx]
    if return_pixels:
        center_pixels_dct = {}; [center_pixels_dct.update(xx[1]) for xx in centers]
        centers = [zz for xx in centers for zz in xx[0]]
        

    if verbose: print ('Total time {} minutes'.format(round((time.time() - start) / 60)))
    if verbose: print('{} objects found.'.format(len(centers)))

    if return_pixels: return center_pixels_dct
    return centers


def helper_labels_centerofmass_thresh((array, start, numZSlicesPerSplit, overlapping_planes, threshold, return_pixels, structure_rank_order)):
    '''
    by tpisano
    
    '''
    zdim, ydim, xdim = array.shape

    structure = generate_binary_structure(array.ndim, structure_rank_order) if structure_rank_order else None

    #process
    if start == 0:
        arr = array[:numZSlicesPerSplit+overlapping_planes]
        #thresholding
        arr[arr<threshold[0]] = 0
        arr[arr>threshold[1]] = 0
        #find labels
        labels = ndimage.measurements.label(arr, structure); lbl_len = labels[1]
        centers = ndimage.measurements.center_of_mass(arr, labels[0], range(1, labels[1]+1)); 
        #return pixels associated with a center
        if return_pixels: dct = return_pixels_associated_w_center(centers, labels)
        del labels, arr
        assert lbl_len == len(centers), 'Something went wrong, center of mass missed labels'
        #filter such that you only keep centers in first half
        centers = [center for center in centers if (center[0] <= numZSlicesPerSplit)]
        if return_pixels: dct = {c:dct[c] for c in centers}

    else: #cover 3x
        arr = array[start - overlapping_planes : np.min(((start + numZSlicesPerSplit + overlapping_planes), zdim))]
        #thresholding
        arr[arr<threshold[0]] = 0
        arr[arr>threshold[1]] = 0
        #find labels
        labels = ndimage.measurements.label(arr)
        centers = ndimage.measurements.center_of_mass(arr, labels[0], range(1, labels[1]))
        #return pixels associated with a center
        if return_pixels: dct = return_pixels_associated_w_center(centers, labels)
        del labels, arr
        #filter such that you only keep centers within middle third
        centers = [center for center in centers if (center[0] > overlapping_planes) and (center[0] <= np.min(((numZSlicesPerSplit + overlapping_planes), zdim)))]
        if return_pixels: dct = {c:dct[c] for c in centers}
        
    #adjust z plane to accomodate chunking
    centers = [(xx[0]+start, xx[1], xx[2]) for xx in centers]
    if return_pixels: 
        dct = {tuple((kk[0]+start, kk[1], kk[2])):v for kk,v in dct.iteritems()}
        return centers, dct

    return centers


def return_pixels_associated_w_center(centers, labels, size = (15,100,100)):
    '''
    by tpisano
    
    Function to return dictionary of k=centers, v=pixels of a given label
    size is the search window from center - done to speed up computation
    ''' 
    dct = {}
    zz,yy,xx = size
    for cen in centers:
        z,y,x = [aa.astype('int') for aa in cen]
        dct[cen] = np.asarray(np.where(labels[0][z-zz:z+zz+1, y-yy:y+yy+1, x-xx:x+xx+1]==labels[0][z,y,x])).T    
    return dct

def calculate_true_negatives(impth, tp, fp, fn):
    
    """ 
    uses formula: TN = total voxels-TP-FP-FN
    calculates TN for roc curve from bipartite mapping output
    #FIXME: check TN formula
    """
    
    #read image    
    img = tifffile.imread(impth)
    imgshp = img.shape
    
    #calculate total voxels
    total_voxels = (imgshp[0]*imgshp[1]*imgshp[2])/(32*20*20)
    
    tn = total_voxels-tp-fp-fn
    
    return tn

def calculate_f1_score(pth, points_dict, threshold = 0.6, cutoff = 30, verbose = False):
    """ 
    simple function to manually calculate F1 scores using human annotations 
    inputs:
        pth = path to datasets
        points_dict = grouth truth dictionary of points/dataset(made before training)
        threshold = desired threshold to test, from 0 to 1
    """
    
    #initialise empty vectors
    tps = []; fps = []; fns = []
    #iterates through forward pass output
    for dset in os.listdir(pth):
        impth = os.path.join(pth, dset)
        predicted = probabiltymap_to_centers_thresh(impth, threshold = (threshold, 1))        
        if verbose: print("\n   Finished finding centers for {}, calculating statistics\n".format(dset))        
        ground_truth = points_dict[dset[:-23]+".npy"] #modifying file names so they match with original data        
        paired, tp, fp, fn = pairwise_distance_metrics(ground_truth, predicted, cutoff = cutoff, verbose = False) #returns true positive = tp; false positive = fp; false negative = fn        
        
        tps.append(tp); fps.append(fp); fns.append(fn)#append matrix to save all values to calculate f1 score and roc curve
    
    tp = sum(tps); fp = sum(fps); fn = sum(fns)
    precision = tp/(tp+fp); recall = tp/(tp+fn) #calculating precision and recall
    f1 = 2*( (precision*recall)/(precision+recall) ) #calculating f1 score
    
    if verbose: print ("\n   Finished calculating statistics for set params\n\n\nReport:\n***************************\n\
                        Threshold: {} \n\
                        Cutoff: {} \n\
                        F1 score: {}% \n\
                        true positives, false positives, false negatives: {} \n\
                        precision: {}% \n\
                        recall: {}%\n".format(threshold, cutoff, round(f1*100, 2), (tp,fp,fn), round(precision*100, 2), round(recall*100, 2)))    
    
    return f1, precision, recall

def generate_precision_recall_curve(precisions, recalls):
    """ plots ROC curve based on contingency table measures obtained from calculate_f1_scores function """
#    
    #calculate
    roc_auc = simps(precisions, dx = 0.002)
    
    plt.figure()
    plt.plot([1-xx for xx in recalls], precisions, color="darkorange", lw=1 , label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("1 - Recall") 
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower right")
    plt.savefig("/jukebox/wang/zahra/conv_net/training/h129/experiment_dirs/20181115_zd_train/precision_recall_curve.pdf")
    
#    return roc_auc

#%%
if __name__ == "__main__":
    
    #set relevant paths
    src = "/jukebox/wang/zahra/conv_net/training/prv/experiment_dirs/20190130_zd_transfer_learning/forward/iters_552460"
    points_dict = load_dictionary("/jukebox/wang/zahra/conv_net/annotations/prv/screened_inputs/filename_points_dictionary.p")
    
    #which thresholds are being evaluated
    thresholds = [0.7, 0.72, 0.75, 0.8,0.85,0.9]#np.arange(0.002, 1, 0.002)
    cutoff = 30
    f1s = []; precisions = []; recalls = []
    
    #generate precision recall list
    for threshold in thresholds:
        f1, precision, recall = calculate_f1_score(src, points_dict, threshold, cutoff, verbose = True)
        f1s.append(f1); precisions.append(precision); recalls.append(recall)
#%%    
    #save
    pth = "/jukebox/wang/zahra/conv_net/training/h129/experiment_dirs/20181115_zd_train/roc_curve_295590.csv"
    generate_precision_recall_curve(precisions, recalls)
    stats_dict = {}
    stats_dict["threshold"] = [(xx, 1) for xx in thresholds]
    stats_dict["f1 score"] = f1s
    stats_dict["precision"] = precisions
    stats_dict["recall"] = recalls
    pd.DataFrame(stats_dict, index = None).to_csv(pth)
    