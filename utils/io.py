#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:42:10 2018

@author: wanglab
"""
import os, random, csv, h5py
from subprocess import check_output
import numpy as np
from skimage.external import tifffile

#function to run
def sp_call(call):
    """ command line call function """ 
    print(check_output(call, shell=True)) 
    return

def check_dim(pth):
    """ 
    find all dimensions of imgs in the directory 
    usefull to check training inputs before setting window size
    i.e. window size should not be larger than input dimensions 
    e.g. pth = '/jukebox/wang/pisano/conv_net/annotations/all_better_res/h129/otsu/inputRawImages'
    only h5 files
    """
    for i, fn in enumerate(os.listdir(pth)):
        f = h5py.File(os.path.join(pth,fn))
        d = f["/main"].value
        f.close()
        print fn, d.shape, np.nonzero(d)[0].shape

def sample_reconstructed_array(pth):
    """ check to make sure reconstruction worked
    pth = path to cnn output folder (probably in scratch) that has the reconstructed array
    """

    flds = os.listdir(pth)
    if 'reconstructed_array.npy' in flds: 
        #read memory mapped array
        chunk = np.lib.format.open_memmap(os.path.join(pth, 'reconstructed_array.npy'), dtype = 'float32', mode = 'r')
        print(chunk.shape)
        
        #save tif
        tifffile.imsave(os.path.join(pth, 'sample.tif'), chunk[500:520, :, :])
        
        print("chunk of z500-520 saved as: {}".format(os.path.join(pth, 'sample.tif')))
    
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
            csv_dict[r[0]] = r[1]
            
    return csv_dict

def find_imgs_to_process(scratch_dir, tracing_fld, call = False):
    """ find 10 random samples to process from tom"s tracing folder,
        submit preprocessing en masse when call = True """
    
    #repo = "/jukebox/wang/zahra/python/3dunet"
    #run/running thorugh cnn
    done = [xx for xx in os.listdir(scratch_dir) if "checked_logs" not in xx 
            and "for_tom" not in xx and "logs" not in xx and "slurm_scripts" not in xx]
    
    print("\n {} are done\n".format(len(done))) 
    
    #all in tracing folder
    brains = os.listdir(tracing_fld)
    
    #find ones not yet processed
    left = [xx for xx in brains if xx not in done]
    print("\n {} are left\n".format(len(left)))
    
    #get a random sample
    to_process = [left[random.randrange(len(left))] for i in range(10)]
    
    print("\n 10 brain samples: \n*************************************************************************\n {}".format(to_process))
    
    pths = [os.path.join(tracing_fld, xx) for xx in to_process]
    print("\n their paths: \n*************************************************************************\n {}".format(pths))
    
    if call:
        for pth in pths:  
            #submit preprocessing jobs
            call = "sbatch cnn_preprocess.sh {}".format(pth)
            print(call)
            sp_call(call)

def submit_reconstruction(scratch_dir, tracing_fld, to_reconstruct = False):
    """ submit reconstruction en masse """

    if not to_reconstruct:
        to_reconstruct = [xx for xx in os.listdir(scratch_dir) if "reconstructed_array.npy"
                      not in os.listdir(os.path.join(scratch_dir, xx)) and "output_chnks" in os.listdir(os.path.join(scratch_dir, xx))]   
    #call
    for pth in to_reconstruct:
        call = "sbatch --array=0-100 slurm_scripts/cnn_step2.sh {}".format(os.path.join(tracing_fld, pth))
        print(call)
        sp_call(call)
        
if __name__ == "__main__":
    
    scratch_dir = "/jukebox/scratch/"
    tracing_fld = "/jukebox/wang/pisano/tracing_output/antero_4x"
    
    find_imgs_to_process(scratch_dir, tracing_fld, call = False)
    
#    to_reconstruct = ["20170308_tp_bl6f_lob6a_2x_01",
#                        "20170116_tp_bl6_lob7_1000r_10",
#                        "20170116_tp_bl6_lob45_ml_11",
#                        "20170115_tp_bl6_lob6a_1000r_02",
#                        "20170204_tp_bl6_cri_1750r_03",
#                        "20180410_jg49_bl6_lob45_02",
#                        "20170207_db_bl6_crii_rlat_03"]
#    
#    submit_reconstruction(scratch_dir, tracing_fld, to_reconstruct)