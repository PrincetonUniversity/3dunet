#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:16:05 2018

@author: wanglab
"""

import os, random
from subprocess import check_output

#function to run
def sp_call(call):
    print(check_output(call, shell=True))
    return

#run/running thorugh cnn
scratch_dir = '/jukebox/scratch/zmd/'
done = [xx for xx in os.listdir(os.path.join(scratch_dir, '3dunet_data/transfered'))+os.listdir(os.path.join(scratch_dir, 'for_tom')) if 'experiments' not in xx]
print('\n {} are done\n'.format(len(done))) 

#all in antero tracing folder
tracing_fld = '/jukebox/wang/pisano/tracing_output/antero_4x'
brains = os.listdir(tracing_fld)

#find ones not yet processed
left = [xx for xx in brains if xx not in done]
print('\n {} are left\n'.format(len(left)))

#get a random sample
random = [left[random.randrange(len(left))] for i in range(10)]

print('\n 10 brain samples: \n*************************************************************************\n {}'.format(random))

repo = '/jukebox/wang/zahra/python/3dunet'
slurm_script = os.path.join(repo, 'cnn_preprocess.sh')

pths = [os.path.join(tracing_fld, xx) for xx in random]
print(pths)

for pth in pths:
    
    #submit preprocessing jobs
    call = 'sbatch cnn_preprocess.sh {}'.format(pth)
    print(call)
    sp_call(call)
