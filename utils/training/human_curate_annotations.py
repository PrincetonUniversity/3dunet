#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:09:41 2019

@author: wanglab
"""

import cv2, numpy as np, os
from skimage.external import tifffile
from tools.conv_net.input.read_roi import read_roi_zip
from tools.registration.transform import swap_cols

roifld = "/home/wanglab/Documents/prv_inputs/hypothalamus/rois_to_rm"

lblfld = "/home/wanglab/Documents/prv_inputs/hypothalamus/otsu"

pth = "/home/wanglab/Documents/prv_inputs/hypothalamus/otsu"

imgs = ['cj_ann_prv_jg32_hypothal_z650-810_01_lbl.tif',
     '20180306_jg_bl6f_prv_16_647_010na_7d5um_250msec_10povlp_ch00_C00_Z0450-0500_06_lbl.tif',
     'cj_ann_prv_jg29_hypothal_z700-800_02_lbl.tif',
     '20180305_jg_bl6f_prv_12_647_010na_7d5um_250msec_10povlp_ch00_C00_400-440_03_lbl.tif',
     'cj_ann_prv_jg24_hypothal_z400-550_04_lbl.tif',
     'cj_ann_prv_jg05_hypothal_z661-760_02_lbl.tif',
     '20180306_jg_bl6f_prv_16_647_010na_7d5um_250msec_10povlp_ch00_C00_Z0650-0700_00_lbl.tif',
     '20180306_jg_bl6f_prv_16_647_010na_7d5um_250msec_10povlp_ch00_C00_Z0650-0700_05_lbl.tif',
     '20180305_jg_bl6f_prv_11_647_010na_7d5um_250msec_10povlp_ch00_C00_300-345_01_lbl.tif',
     'cj_ann_prv_jg32_hypothal_z710-810_02_lbl.tif',
     '20180215_jg_bl6f_prv_10_647_010na_z7d5um_250msec_10povlap_ch00_z200-400_y4500-4850_x2050-2400_lbl.tif',
     '20180306_jg_bl6f_prv_16_647_010na_7d5um_250msec_10povlp_ch00_C00_Z0650-0700_01_lbl.tif',
     'zd_ann_prv_jg32_hypothal_z710-810_02_lbl.tif']

for img in imgs:

    #read label image        
    lbl = tifffile.imread(os.path.join(pth, img))
    
    #find roi pths associated w that dataset
    roipths = [os.path.join(roifld, xx) for xx in os.listdir(roifld) if img[:-8] in xx and img[:13] == xx[:13]]

    if not roipths == 0:
        print(img)
        for roipth in roipths:     
            rois = [xx for xx in read_roi_zip(roipth, include_roi_name=True) if ".zip.roi" not in xx[0]]
            print("\n length of rois: {}\n".format(len(rois)))
            #format so each rois is [z,y,x, [contour]]; remeber IMAGEJ has one-based numerics for z plane. NOTE roi is ZYX, contour[XY???], why swap_cols
            rois = [(map(int, xx[0].replace(".roi","").split("-")), xx[1]) for xx in rois]
            rois = [(xx[0][0]-1, xx[0][1], xx[0][2], swap_cols(xx[1], 0,1)) for xx in rois]        
            for roi in rois:
                zi, yi, xi, yxcontour = roi
                blank = np.zeros_like(lbl[0])
                segment = cv2.fillPoly(blank, [np.int32(yxcontour)], color=255)
                y0, x0 = np.nonzero(segment)        
                lbl[zi, y0, x0] = 0
    
        tifffile.imsave("/home/wanglab/Documents/prv_inputs/hypothalamus/new_lbls/{}".format(os.path.basename(img)), lbl)
    