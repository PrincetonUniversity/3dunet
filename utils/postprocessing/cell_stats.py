#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:13:16 2018

@author: wanglab

by Tom Pisano (tpisano@princeton.edu, tjp77@gmail.com) & Zahra D (zmd@princeton.edu, zahra.dhanerawala@gmail.com)

"""
import numpy as np, cv2, sys, os
import time
from skimage.external import tifffile

from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure
import pandas as pd


def calculate_cell_measures(**params):
    
    #calculate cell measures
    df = probabiltymap_to_cell_measures(params["reconstr_arr"], threshold = params["threshold"], numZSlicesPerSplit = params["zsplt"], 
                                                            overlapping_planes = params["ovlp_plns"], cores = 1, 
                                                            verbose = params["verbose"])
    
    df.to_csv(os.path.join(params["output_dir"], "cells/cell_measures.csv"))
    
    #return csv path
    return os.path.join(params["output_dir"], "cells/cell_measures.csv")


#%%
def probabiltymap_to_cell_measures(src, threshold = (0.6,1), numZSlicesPerSplit = 100, overlapping_planes = 30, cores = 10, 
                                    verbose = False, structure_rank_order = 2):
    '''
    by tpisano
    
    Function to take probabilty maps generated by run_cnn and output centers based on center_of_mass, as well as sphericties and perimeter of cell

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
        if src[-3:] == 'npy': src = np.lib.format.open_memmap(src, dtype = 'float32', mode = 'r')

    zdim, ydim, xdim = src.shape
    start = time.time()
    if verbose: sys.stdout.write('\n   thresholding, determining connected pixels, identifying center of masses\n\n')
    sys.stdout.flush() 
    
    iterlst=[(src, z, numZSlicesPerSplit, overlapping_planes, threshold, structure_rank_order) for z in range(0, zdim, numZSlicesPerSplit)]    
    
    #run
    df = []
    for i in iterlst: 
        #i = iterlst[9]
        #(array, start, numZSlicesPerSplit, overlapping_planes, threshold, structure_rank_order) = iterlst[5]
        df.append(find_labels_centerofmass_cell_measures(i[0], i[1], i[2], i[3], i[4], i[5]))
        
    df = pd.concat(df)
                
    print ('total time {} minutes'.format(round((time.time() - start) / 60)))

    return df

def find_labels_centerofmass_cell_measures(array, start, numZSlicesPerSplit, overlapping_planes, threshold, structure_rank_order):
    '''
    by tpisano
    
    '''
    zdim, ydim, xdim = array.shape

    structure = generate_binary_structure(array.ndim, structure_rank_order) if structure_rank_order else None
    
    #get array
    if start==0:
        arr = array[:numZSlicesPerSplit+overlapping_planes]
    else:
        arr = array[max(start - overlapping_planes,0) : np.min(((start + numZSlicesPerSplit + overlapping_planes), zdim))]
    
    #thresholding
    a = arr>=threshold[0]
    a = a.astype("bool") #thanks to ben - reduces size of arr 4 fold!
    #del arr
    
    #find labels
    labels = ndimage.measurements.label(a, structure)
    centers = ndimage.measurements.center_of_mass(a, labels[0], range(1, labels[1]+1)); 
    
    #filter
    if start==0:
        #such that you only keep centers in first chunk
        centers = [center for center in centers if (center[0] <= numZSlicesPerSplit)]
    else:
        #such that you only keep centers within middle third
        centers = [center for center in centers if (center[0] > overlapping_planes) and (center[0] <= np.min(((numZSlicesPerSplit + overlapping_planes), zdim)))]
        
    #get perimeter of cell and sphericities
    perimeters, sphericities, zspans = find_perimeter_sphericity(tuple((labels, centers, start)))
    del a, labels
    
    #get radii
    intensities = find_intensity(centers, start, arr, zyx_search_range=(5,10,10))
        
    #adjust z plane to accomodate chunking
    if start!=0:
        centers = [(xx[0]+(start-overlapping_planes), xx[1], xx[2]) for xx in centers] #think this might have been the error on my part. It didn't have the start minus overlapping planes
    
    #put into df
    data = [[centers[i][0].astype("uint16"), centers[i][1].astype("uint16"), centers[i][2].astype("uint16"), intensities[i], sphericities[i], perimeters[i], zspans[i]] for i in range(len(centers))]
    df = pd.DataFrame(data = data, columns = ["z", "y", "x", "intensity", "sphericity", "maximum perimeter", "length of cell (in z)" ])
    
    return df

#%%
def perimeter_sphericity(src, dims = 3):
    """
    src = 3d
    
    looks at it from two perspectives and then takes the total average
    dims=(2,3) number of dimensions to look at
    
    ball(9) = .895
    cube(9) = .785
    cylinder(9,9) = .770
    np.asarray([star(9) for xx in range(9)]) = .638
    
    sometime two contours are found on a zplane after labels - in this case take min, but could take average?
    """
    #initialise
    sphericities = []; perimeters = []
    
    if 0 in src.shape: 
        return 0, 0 #projects against empty labels    
    else:
        #in z dimension
        for z in src:
            try:
                contours = findContours(z.astype("uint8"))
                circ = circularity(contours)
                perimeter = cv2.arcLength(contours, True)
                sphericities.append(circ); perimeters.append(perimeter)
            except:
                "no cell in plane"
                
        #look in y dimension
        if dims == 3:
            for z in np.swapaxes(src, 0, 1):
                try:
                    contours = findContours(z.astype("uint8"))
                    circ = circularity(contours)
                    perimeter = cv2.arcLength(contours, True)
                    sphericities.append(circ); perimeters.append(perimeter)
                except:
                    "no cell in plane"
        
        #return - maximum perimeter and mean sphericity per cell
        #TODO: add z plane span of a cell
        if len(perimeters) > 0:
            perimeter = np.max(perimeters)
            zspan = len(perimeters)
        else:
            perimeter = perimeters
        if len(sphericities) > 0: 
            sphericity = np.mean(sphericities) 
        else: 
            sphericity = sphericities
            
        #return values
        return perimeter, sphericity, zspan


def circularity(contours):
    """
    A Hu moment invariant as a shape circularity measure, Zunic et al, 2010
    """
    #moments = [cv2.moments(c.astype(float)) for c in contours]
    #circ = np.array([(m['m00']**2)/(2*np.pi*(m['mu20']+m['mu02'])) if m['mu20'] or m['mu02'] else 0 for m in moments])
    circ = [ (4*np.pi*cv2.contourArea(c))/(cv2.arcLength(c,True)**2) for c in contours]

    return np.asarray(circ)

def findContours(z):
    """
    Function to handle compatiblity of opencv2 vs 3
    """
    if str(cv2.__version__)[0] == '3':
        cim,contours,hierarchy = cv2.findContours(z, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #if need more than two values to unpack error here upgrade to cv2 3+
    elif str(cv2.__version__)[0] == '2':
        contours,hierarchy = cv2.findContours(z, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #if need more than two values to unpack error here upgrade to cv2 3+
    contours = np.asarray([c.squeeze() for c in contours if cv2.contourArea(c)>0])
    return contours


def find_perimeter_sphericity(args): #(labels, centers):
    """
    Function that takes in from scipy.ndimage.labels
    
    1 is more spherical
    dims=(2,3) number of dimensions to look at
    box_size: 1/2 size of bounding box (should be comfortably larger than cell). 50 = 50pixels +/- center, meaning a 100pixel length cube

    """    
    #set variables
    timer = time.time()
    labels = args[0]; centers = args[1]
    
    #initialise dataframes
    perimeters = []; sphericities = []; zspans = []
    
    if labels[1]==0: #np array of zeros
        return labels
    else:
        vals = range(1, labels[1]+1) #don't need zero and account for zero indexing
        for val, center in zip(vals, centers):   
            vol = bounding_box_from_center_array(labels[0], val, center) #FIXME: make start a variable
            perimeter, sphericity, zspan = perimeter_sphericity(vol)      
            perimeters.append(perimeter); sphericities.append(sphericity); zspans.append(zspan)
    
    print ('Total time for measurements {} minutes'.format(round((time.time() - timer) / 60)))
    
    return perimeters, sphericities, zspans

def bounding_box_from_center_array(src, val, center, ovrlp = 10, box_size=(20, 50, 50)):
    """
    Faster version of _array as it makes a box rather than calculating
    """
    z,y,x = [int(xx) for xx in center]
    
    out = np.copy(src[max(0,z-box_size[0]):z+box_size[0], max(0, y-box_size[1]):y+box_size[1], max(x-box_size[2],0):x+box_size[2]]) #copy is critical
    
    a = (out==val).astype(int)
    return a.astype("uint8")

    
def find_intensity(centers, start, recon_dst, zyx_search_range=(5,10,10)):
    """
    function to return maximum intensity of a determined center src_raw given a zyx point and search range
    zyx_search_range=(4,10,10)
    zyx = (345, 3490, 3317)
    """
    
    #handle input
    if type(recon_dst) == str:
        #load reconstructed memmap array
        cnn_src = np.lib.format.open_memmap(recon_dst, dtype = "float32", mode = "r")
    else:
        cnn_src = recon_dst
    
    #initialise
    a = []
    for i in centers:
        #setting the proper ranges
        z,y,x = i[0],i[1],i[2]
        zr,yr,xr = zyx_search_range
        
        #making sure ranges are not negative
        rn = []
        for xx,yy in zip((z,y,x), zyx_search_range):
            if yy < xx:
                rn.append((int(xx-yy), int(xx+yy+1)))
            else:
                rn.append((int(xx), int(xx+yy+1)))
    
        
        #find the maximum part of cell 
        mx = np.max(cnn_src[rn[0][0]:rn[0][1], rn[1][0]:rn[1][1], rn[2][0]:rn[2][1]])
        
        #append the initialised list
        a.append(mx)
        
    return np.array(a)
     
