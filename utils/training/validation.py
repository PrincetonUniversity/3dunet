# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:44:04 2018

@author: zahramansoor
"""

import re, os
import matplotlib.pyplot as plt
import h5py, numpy as np
from scipy.stats import linregress

def save_stats_h5(fname):
    '''Function to extract test loss and training loss values from h5 files saved in training.
    '''

    with h5py.File(fname) as f:
        print('keys of file:\n {}'.format(list(f.keys())))
        print('base lr value: {}'.format(f['base_lr'].value))
        test = list(f['test'].keys())
        print('contents of test dict: \n {}'.format(test))
        train = list(f['train'].keys())
        print('contents of train dict: \n {}'.format(train))
        test_loss_arr = f['test'][test[2]].value
        train_loss_arr = f['train'][train[2]].value
        
    return test_loss_arr, train_loss_arr



def plot_val_curve(loss, start_iter = 0):
    '''Function to plot validation data loss value from .out file from training on tiger2
    Inputs:
        loss = array of loss values
        start_iter = iteration from which to start plotting from
    '''
    #set x
    iters = np.arange(start_iter, len(loss))
    
    #linear regression
    fit = np.polyfit(iters, loss[start_iter:], 1)
    fit_fn = np.poly1d(fit)
    linreg_stats = linregress(iters, loss[start_iter:])
    
    #plot
    plt.rcParams.update({'font.size': 8})
    plt.figure()
    plt.plot(loss[start_iter:], 'r')
    plt.xlabel('# of iterations in thousands')
    plt.ylabel('loss value')
    plt.title('3D U-net validation curve for H129')          
#        plt.savefig(os.path.join(pth, 'val_zoom'), dpi = 300)
#    plt.close() 
    plt.figure()
    plt.plot(iters, loss[start_iter:], 'yo', iters, fit_fn(iters), '--k')
    
    return linreg_stats