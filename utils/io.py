#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:42:10 2018

@author: wanglab
"""

import os, csv, h5py, cv2, ast
from subprocess import check_output
import numpy as np
from skimage.external import tifffile
from utils.postprocessing.cell_stats import consolidate_cell_measures

#function to run
def sp_call(call):
    """ command line call function """ 
    print(check_output(call, shell=True)) 
    return


def make_inference_output_folder(pth):
    """ needed to start inference correctly so chunks aren't missing from output folder """
    
    if not os.path.exists(os.path.join(pth, "output_chnks")): os.mkdir(os.path.join(pth, "output_chnks"))
    print("output folder made for :\n {}".format(pth))
    
    return

def consolidate_cell_measures_bulk(pth):
    
    fls = [xx for xx in os.listdir(pth) if "reconstructed_array.npy" in os.listdir(os.path.join(pth, xx))]
    
    for fl in fls:
        src = os.path.join(os.path.join(pth, fl), "cnn_param_dict.csv")
        params = csv_to_dict(src)
        consolidate_cell_measures(**params)
    
    return
    
def resize(pth, dst, resizef = 6):
    """ 
    resize function using cv2
    inputs:
        pth = 3d tif stack or memmap array
        dst = folder to save each z plane
    """
    #make sure dst exists
    if not os.path.exists(dst): os.mkdir(dst)
    
    #read file
    if pth[-4:] == ".tif": img = tifffile.imread(pth)
    elif pth[-4:] == ".npy": img = np.lib.format.open_memmap(pth, dtype = "float32", mode = "r")
    
    z,y,x = img.shape
    
    for i in range(z):
        #make the factors
        xr = img[i].shape[1] / resizef; yr = img[i].shape[0] / resizef
        im = cv2.resize(img[i], (xr, yr), interpolation=cv2.INTER_LINEAR)
        tifffile.imsave(os.path.join(dst, "zpln{}.tif".format(str(i).zfill(12))), im.astype("float32"), compress=1)
    
    return dst

def resize_stack(pth, dst):
    
    """
    runs with resize
    inputs:
        pth = folder with resized tifs
        dst = folder
    """
    #make sure dst exists
    if not os.path.exists(dst): os.mkdir(dst)
    
    #get all tifs
    fls = [os.path.join(pth, xx) for xx in os.listdir(pth) if xx[0] != "." and "~" not in xx and "Thumbs.db" not in xx and ".tif" in xx]; fls.sort()
    y,x = tifffile.imread(fls[0]).shape
    dims = (len(fls),y,x)
    stack = np.zeros(dims)
    
    for i in range(len(fls)):
        stack[i] = tifffile.imread(fls[i])
    
    #save stack
    tifffile.imsave(os.path.join(dst, "resized_stack.tif"), stack.astype("float32"))
    
    return os.path.join(dst, "resized_stack.tif")
    
def check_dim(pth):
    """ 
    find all dimensions of imgs in the direccvtory 
    usefull to check training inputs before setting window size
    i.e. window size should not be larger than input dimensions 
    e.g. pth = "/jukebox/wang/pisano/conv_net/annotations/all_better_res/h129/otsu/inputRawImages"
    only h5 files
    """
    for i, fn in enumerate(os.listdir(pth)):
        f = h5py.File(os.path.join(pth,fn))
        d = f["/main"].value
        f.close()
        print(fn, d.shape, np.nonzero(d)[0].shape)

def sample_reconstructed_array(pth, zstart, zend):
    """ check to make sure reconstruction worked
    pth = path to cnn output folder (probably in scratch) that has the reconstructed array
    """

    flds = os.listdir(pth)
    if "reconstructed_array.npy" in flds: 
        #read memory mapped array
        chunk = np.lib.format.open_memmap(os.path.join(pth, "reconstructed_array.npy"), dtype = "float32", mode = "r")
        print(chunk.shape)
        
        #save tif
        tifffile.imsave(os.path.join(pth, "sample.tif"), chunk[zstart:zend, :, :])
        
        print("chunk saved as: {}".format(os.path.join(pth, "sample.tif")))
    
def csv_to_dict(csv_pth):
    """ 
    reads csv and converts to dictionary
    1st column = keys
    2nd column = values
    """
    csv_dict = {}
    
    with open(csv_pth) as csvf:
        f = csv.reader(csvf)
        for r in f:
            if r[1][:9] == "/jukebox/" or r[0] == "dtype" or r[0] == "expt_name":
                csv_dict[r[0]] = r[1]
            else:
                csv_dict[r[0]] = ast.literal_eval(r[1]) #reads integers, tuples, and lists as they were entered
    
    return csv_dict

