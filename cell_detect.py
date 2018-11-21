#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 15:42:12 2018

@author: wanglab
"""

import os, numpy as np, sys, multiprocessing as mp, time, shutil
from scipy import ndimage
from skimage.external import tifffile
from scipy.ndimage.morphology import generate_binary_structure
from skimage.util import view_as_windows, regular_grid
import argparse                   

def load_np(src, mode='r'):
    '''Function to handle .npy and .npyz files. Assumes only one k,v pair in npz file
    '''
    if str(type(src)) == "<type 'numpy.ndarray'>" or str(type(src)) == "<class 'numpy.core.memmap.memmap'>":
        return src
    elif src[-4:]=='.npz':
        fl = np.load(src)
        #unpack ASSUMES ONLY SINGLE FILE
        arr = [fl[xx] for xx in fl.keys()][0]
        return arr
    elif src[-4:]=='.npy':
        try:
            arr=load_memmap_arr(src, mode)
        except:
            arr = np.load(src)
        return arr

def makedir(path):
    '''Simple function to make directory if path does not exists'''
    if os.path.exists(path) == False:
        os.mkdir(path)
    return

def generate_patch(input_arr, patch_dst, patchlist, stridesize, patchsize, mode = 'folder', verbose = True):
    '''Function to patch up data and make into memory mapped array
    
    Inputs
    -----------
    src = folder containing tiffs
    patch_dst = location to save memmap array
    patchlist = list of patches generated from make_indices function
    stridesize = (90,90,30) - stride size in 3d ZYX
    patchsize = (180,180,60) - size of window ZYX
    mode = 'folder' #'folder' = list of files where each patch is a file, 'memmap' = 4D array of patches by Z by Y by X

    Returns
    ------------
    location of patched memory mapped array of shape (patches, patchsize_z, patchsize_y, patchsize_x)

    '''
    #load array
    input_arr = load_np(input_arr)
        
    if mode == 'memmap':
        print('Mode == memmap')
        #init patch array:
        inputshape = (len(patchlist), patchsize[0], patchsize[1], patchsize[2])
        patch_array = load_memmap_arr(patch_dst, mode='w+', shape = inputshape, dtype = 'float32')
            
        #patch
        for i,p in enumerate(patchlist):
            v = input_arr[p[0]:p[0]+patchsize[0], p[1]:p[1]+patchsize[1], p[2]:p[2]+patchsize[2]]
            patch_array[i, :v.shape[0], :v.shape[1], :v.shape[2]] = v
            if i%2==0: 
                patch_array.flush()
                if verbose: print('{} of {}'.format(i, len(patchlist)))
        patch_array.flush()
        
    if mode == 'folder':
        print('Mode == folder')
        if patch_dst[-4:]=='.npy': patch_dst = patch_dst[:-4]
        if not os.path.exists(patch_dst): os.mkdir(patch_dst)
        #patch
        for i,p in enumerate(patchlist):
            if i >= len(os.listdir(patch_dst)-1): #so that you can re-run a job if it was killed halfway (bc of occassional memory issues)
                v = input_arr[p[0]:p[0]+patchsize[0], p[1]:p[1]+patchsize[1], p[2]:p[2]+patchsize[2]]
                tifffile.imsave(os.path.join(patch_dst, 'patch_{}.tif'.format(str(i).zfill(10))), v.astype('float32'), compress=1)
                if i%10==0 and verbose: print('{} of {}'.format(i, len(patchlist))); del v
    #return
    return patch_dst
   
    
def get_dims_from_folder(src):    
    '''Function to get dims from folder (src)
    '''
    
    fls = listdirfull(src, keyword = '.tif')
    y,x = tifffile.imread(fls[0]).shape
    return (len(fls),y,x)
    
def make_indices(inputshape, stridesize):
    '''Function to collect indices
    inputshape = (500,500,500)
    stridesize = (90,90,30)
    '''    
    zi, yi, xi = inputshape
    zs, ys, xs = stridesize
    
    lst = []
    z = 0; y = 0; x = 0
    while z<zi:
        while y<yi:
            while x<xi:
                lst.append((z,y,x))
                x+=xs
            x=0
            y+=ys
        x=0
        y=0
        z+=zs
    return lst

def make_memmap_from_tiff_list(src, dst, cores=1, dtype=False, verbose=True):
    '''Function to make a memory mapped array from a list of tiffs
    '''
    
    if type(src) == str and os.path.isdir(src): 
        src = listdirfull(src, keyword = '.tif')
        src.sort()
    im = tifffile.imread(src[0])
    if not dtype: dtype = im.dtype
    
    #init
    memmap=load_memmap_arr(dst, mode='w+', dtype=dtype, shape=tuple([len(src)]+list(im.shape)))
    
    #run
    if cores<=1:
        for i,s in enumerate(src):
            memmap[i,...] = tifffile.imread(s)
            memmap.flush()
    else:
        iterlst = [(i,s, dst, verbose) for i,s in enumerate(src)]    
        p = mp.Pool(cores)
        p.map(make_memmap_from_tiff_list_helper, iterlst)
        p.terminate

    return dst

def make_memmap_from_tiff_list_helper((i, s, memmap_pth, verbose)):
    '''
    '''
    #load
    arr=load_np(memmap_pth, mode='r+')
    arr[i,...] = tifffile.imread(s)
    arr.flush(); del arr
    if verbose: sys.stdout.write('\ncompleted plane {}'.format(i)); sys.stdout.flush()
    return

def listdirfull(x, keyword=False):
    '''might need to modify based on server...i.e. if automatically saving a file called 'thumbs'
    '''
    if not keyword:
        return [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != '.' and '~' not in xx and 'Thumbs.db' not in xx]
    else:
        return [os.path.join(x, xx) for xx in os.listdir(x) if xx[0] != '.' and '~' not in xx and 'Thumbs.db' not in xx and keyword in xx]

def load_memmap_arr(pth, mode='r', dtype = 'float32', shape = False):
    '''Function to load memmaped array.
    
    by @tpisano

    '''
    if shape:
        assert mode =='w+', 'Do not pass a shape input into this function unless initializing a new array'
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode, shape = shape)
    else:
        arr = np.lib.format.open_memmap(pth, dtype = dtype, mode = mode)
    return arr
    
def reconstruct_memmap_array_from_tif_dir(cnn_src, recon_dst, inputshape, patchlist, patchsize, verbose=True):
    '''Function to take CNN probablity map tifs (patches, patchsize_z, patchsize_y, patchsize_x) and build into single 3d volume
    
    Inputs
    ---------------
    src = cnn_memory_mapped array of shape (patches, patchsize_z, patchsize_y, patchsize_x)
    recon_dst = path to generate numpy array
    inputshape = (Z,Y,X) shape of original input array
    patchlist = list of patches generated from make_indices function
    stridesize = (90,90,30) - stride size in 3d ZYX
    patchsize = (180,180,60) - size of window ZYX
    
    Returns
    ------------
    location of memory mapped array of inputshape
    '''
    
    #load
    cnn_fls = os.listdir(cnn_src); cnn_fls.sort()
    
    #init new array
    recon_array = load_memmap_arr(recon_dst, mode='w+', shape = inputshape, dtype = 'float32')
    
    #patchsize
    zps, yps, xps = patchsize
    
    #iterate
    for i,p in enumerate(patchlist):    
        b = tifffile.imread(os.path.join(cnn_src,cnn_fls[i])).astype('float32')
        a = recon_array[p[0]:p[0]+b.shape[0], p[1]:p[1]+b.shape[1], p[2]:p[2]+b.shape[2]]
        if not a.shape == b.shape: b = b[:a.shape[0], :a.shape[1], :a.shape[2]]
        nvol = np.maximum(a,b)
        recon_array[p[0]:p[0]+b.shape[0], p[1]:p[1]+b.shape[1], p[2]:p[2]+b.shape[2]] = nvol
        recon_array.flush(); del b
        if verbose: print('{} of {}'.format(i, len(patchlist)))

    return recon_dst
    
    
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
        if src[-3:] == 'tif': src = tifffile.imread(src)
        if src[-3:] == 'npy': src = load_np(src)

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

#%%
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('expt_name',
                        help='Tracing output directory (aka registration output)')
    parser.add_argument('stepid', type=int,
                        help='Step ID to run patching, reconstructing, or cell counting')
    args = parser.parse_args()
    
    #setup
    fsz = os.path.join(args.expt_name, 'full_sizedatafld')
    vols = os.listdir(fsz); vols.sort()
    src = os.path.join(fsz, vols[len(vols)-1]) #hack - try to load param_dict instead?
    if not os.path.isdir(src): src = os.path.join(fsz, vols[len(vols)-2]) #hack - try to load param_dict instead?
    sys.stdout.write('\n preprocessing tif directory of: \n{}'.format(src)); sys.stdout.flush()
    
    #set params to use for reconstruction
    patchsz = (64,3840,3328) #cnn window size for lightsheet = typically 20, 192, 192 #patchsize = (64,3840,3136)
    stridesz = (44,3648,3136) #stridesize = (44,3648,2944)

    #set params    
    dtype = 'float32'    
    cores = 8
    verbose = True 
    cleanup = True #if True, files will be deleted when they aren't needed. Keep false while testing
    mode = 'folder' #'folder' = list of files where each patch is a file, 'memmap' = 4D array of patches by Z by Y by X (not recommended)
    
    #make patches
    inputshape = get_dims_from_folder(src)
    patchlist = make_indices(inputshape, stridesz)
    
    #set scratch directory    
    dst = os.path.join('/jukebox/scratch/zmd', os.path.basename(os.path.abspath(args.expt_name))); makedir(dst)
    in_dst = os.path.join(dst, 'input_memmap_array.npy') 
    
    #recover step id from command line args      
    stepid = args.stepid
    
    if stepid == 0:
        #######################################PRE-PROCESSING FOR CNN INPUT --> MAKING INPUT ARRAY######################################################
            
        #convert full size data folder into memmap array
        input_arr = make_memmap_from_tiff_list(src, in_dst, cores, dtype = dtype)
            
    if stepid == 1:
        #######################################PRE-PROCESSING FOR CNN INPUT --> PATCHING######################################################
        
        #generate memmap array of patches
        patch_dst = os.path.join(dst, 'input_chnks')
        sys.stdout.write('\n making patches...\n'); sys.stdout.flush()
        patch_dst = generate_patch(in_dst, patch_dst, patchlist, stridesz, patchsz, mode = mode, verbose = verbose)

    elif stepid == 2:
        #######################################POST CNN --> RECONSTRUCTION AFTER RUNNING INFERENCE ON TIGER2#################################

        #set cnn patch directory
        cnn_src = os.path.join(dst, 'cnn_output')
  
        #reconstruct
        sys.stdout.write('\n starting reconstruction...\n'); sys.stdout.flush()
        recon_dst = os.path.join(args.cnn_src, 'reconst_array.npy')
        reconstruct_memmap_array_from_tif_dir(cnn_src, recon_dst, inputshape, patchlist, patchsz, verbose = verbose)
        if cleanup: shutil.rmtree(cnn_src)

    elif stepid == 3:
        ##############################################POST CNN --> FINDING CELL CENTERS#####################################################
        
        #FIXME: untested
        #load cnn_src to find shape and iterate
        arr = load_np(cnn_src)    
        
        #find cell centers    
        predicted_cell_centers = load_memmap_arr(os.path.join(cnn_src, 'cells.npy'), mode = 'w+', dtype = 'float32', shape = arr.shape[0])
        
        #iterates through forward pass output
        #FIXME: find radius and max intensity and save it in same array too
        for i in range(arr.shape[0]):
            
            predicted_cell_centers[i] = probabiltymap_to_centers_thresh(arr[i,:,:,:], threshold = (0.4, 1))
            
            print '\n   Finished finding centers for patch # {}\n'.format(i+1)
            
            predicted_cell_centers.flush()