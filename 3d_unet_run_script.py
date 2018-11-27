#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 08:32:24 2018

@author: tpisano
"""

#https://github.com/PrincetonUniversity/3dunet/blob/master/run_chnk_fwd.sh
#https://github.com/PrincetonUniversity/3dunet/blob/master/pytorchutils/run_chnk_fwd.py

#repull repo to tiger2 (gpu)
#https://github.com/PrincetonUniversity/3dunet
#git clone https://username:password@github.com/PrincetonUniversity/3dunet.git

#to activate
#module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda3/5.2.0
#. activate 3dunet


#transfer from PNI scratch to tigress using globus https://app.globus.org/file-manager
import os
inn = '/jukebox/scratch/zmd/'
out = '/scratch/gpfs/zmd/'
logs = os.path.join(out, 'logs')
if not os.path.exists(logs): os.mkdir(logs)
repo = '/tigress/zmd/3dunet'
slurm_script = 'run_chnk_fwd.sh'
fld = os.path.join(out, 'slurm_scripts')
if not os.path.exists(fld): os.mkdir(fld)
print(fld)

#function to run
from subprocess import check_output
def sp_call(call):
    print(check_output(call, shell=True))
    return


#loop
paths = [xx for xx in os.listdir(out) if 'completed' not in xx and 'split_data' not in xx]
paths_to_call = []
for pth in paths:
    
    #read file
    with open(os.path.join(repo, slurm_script), 'r') as fl:
        lines = fl.readlines()
        fl.close()
    
    #modify
    new_lines = lines[:]
    for i,line in enumerate(lines):
        
        #new dir
        if 'cd pytorchutils/' in line:
            new_lines[i] = line.replace('pytorchutils/', '/tigress/zmd/3dunet/pytorchutils/')
        
        #path to folder
        if 'python run_chnk_fwd.py 20181115_zd_train' in line:
            new_lines[i] = line.replace('20180327_jg42_bl6_lob6a_05', pth)
            
        #logs
        if 'logs/chnk_' in line:
            new_lines[i] = line.replace('logs', logs)
            
    #save out
    with open(os.path.join(fld, 'run_'+pth+'.sh'), 'w+') as fl:
        [fl.write(new_line) for new_line in new_lines]
        fl.close()
        
    #collect
    paths_to_call.append(os.path.join(fld, 'run_'+pth+'.sh'))
    
#call
for pth in paths_to_call:
    call = 'sbatch --array=0-100 {}'.format(pth)
    print(call)
    sp_call(call)
    