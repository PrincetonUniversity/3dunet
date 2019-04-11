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

roifld = "/home/wanglab/Documents/prv_inputs/rois_to_rm"

lblfld = "/home/wanglab/Documents/prv_inputs/otsu/"

pth = "/home/wanglab/Documents/prv_inputs/otsu"

imgs = ['JGANNOTATION_20180305_jg_bl6f_prv_12_647_010na_7d5um_250msec_10povlp_ch00_C00_400-440_01_lbl.tif',
     'JGANNOTATION_20180305_jg_bl6f_prv_12_647_010na_7d5um_250msec_10povlp_ch00_C00_400-440_02_lbl.tif',
     'JGANNOTATION_20180306_jg_bl6f_prv_16_647_010na_7d5um_250msec_10povlp_ch00_C00_Z0450-0500_01_lbl.tif',
     'cj_ann_prv_jg05_neocortex_z250-449_02_lbl.tif',
     'cj_ann_prv_jg05_neocortex_z250-449_02_lbl.tif',
     'cj_ann_prv_jg05_neocortex_z250-449_02_lbl.tif',
     'cj_ann_prv_jg05_neocortex_z250-449_02_lbl.tif',
     'cj_ann_prv_jg05_neocortex_z250-449_02_lbl.tif',
     'cj_ann_prv_jg05_neocortex_z250-449_02_lbl.tif',
     'cj_ann_prv_jg05_neocortex_z250-449_02_lbl.tif',
     'cj_ann_prv_jg05_neocortex_z310-449_01_lbl.tif',
     'cj_ann_prv_jg24_neocortex_z300-400_01_lbl.tif',
     'cj_ann_prv_jg24_neocortex_z300-400_02_lbl.tif',
     'cj_ann_prv_jg29_neocortex_z700-800_02_lbl.tif',
     "cj_ann_prv_jg32_neocortex_z650-810_01_lbl.tif",
     'zd_ann_prv_jg05_neocortex_z310-449_01_lbl.tif',
     'zd_ann_prv_jg24_neocortex_z300-400_01_lbl.tif',
     'zd_ann_prv_jg29_neocortex_z300-500_01_lbl.tif',
     "zd_ann_prv_jg32_neocortex_z650-810_01_lbl.tif"
     ]

for img in imgs:
        
    lbl = tifffile.imread(os.path.join(pth, img))
    
    roipths = [os.path.join(roifld, xx) for xx in os.listdir(roifld) if os.path.basename(img)[6:-4] in xx]
    
    if not roipths == 0:
        for roipth in roipths:     
            rois = [xx for xx in read_roi_zip(roipth, include_roi_name=True) if ".zip.roi" not in xx[0]]
            #format so each rois is [z,y,x, [contour]]; remeber IMAGEJ has one-based numerics for z plane. NOTE roi is ZYX, contour[XY???], why swap_cols
            rois = [(map(int, xx[0].replace(".roi","").split("-")), xx[1]) for xx in rois]
            rois = [(xx[0][0]-1, xx[0][1], xx[0][2], swap_cols(xx[1], 0,1)) for xx in rois]        
            for roi in rois:
                zi, yi, xi, yxcontour = roi
                blank = np.zeros_like(lbl[0])
                segment = cv2.fillPoly(blank, [np.int32(yxcontour)], color=255)
                y0, x0 = np.nonzero(segment)        
                lbl[zi, y0, x0] = 0
    
        tifffile.imsave("/home/wanglab/Documents/prv_inputs/new_lbls/{}".format(os.path.basename(img)), lbl)
    