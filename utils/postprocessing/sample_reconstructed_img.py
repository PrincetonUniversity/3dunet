# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:30:01 2018

@author: zahramansoor
"""

import os
import tifffile as tif
import numpy as np

dst = '/jukebox/scratch/'

#find brains
fls = [xx for xx in os.listdir(dst) if xx not in 'for_tom' and xx not in 'stride_test']
tracing_fld = '/jukebox/wang/pisano/tracing_output/antero_4x'

pths = [os.path.join(dst, fl) for fl in fls]

for pth in pths:
    flds = os.listdir(pth)
    if 'reconstructed_array.npy' in flds: 
        chunk = np.lib.format.open_memmap(os.path.join(pth, 'reconstructed_array.npy'), dtype = 'float32', mode = 'r')
        print(chunk.shape)
        tif = tif.imsave(os.path.join(pth, 'sample_output_seam_check.tif'), chunk[500:520, :, :])
        print(os.path.join(pth, 'sample_output_seam_check.tif'))