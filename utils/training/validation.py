# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:44:04 2018

@author: zahramansoor
"""

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



def plot_val_curve(loss, start_iter = 0, m = 50):
    '''Function to plot validation data loss value from .out file from training on tiger2
    Inputs:
        loss = array of loss values
        start_iter = iteration from which to start plotting from
        m = multiple at which log was saved (in parameter dictionary)
    '''
    #set x and y
    iters = np.arange(0, len(loss))
    if len(loss) > 1000: 
        loss = np.take(loss, np.arange(0, len(loss)-1, m)) 
        iters = np.take(iters, np.arange(0, len(iters)-1, m))
    
    #linear regression
    fit = np.polyfit(iters[start_iter:], loss[start_iter:], 1)
    fit_fn = np.poly1d(fit)
    linreg_stats = linregress(iters[start_iter:], loss[start_iter:])
    
    #plot
    plt.rcParams.update({'font.size': 8})
    plt.figure()
    plt.plot(loss[start_iter:], 'ro')
    plt.xlabel('# of iterations in thousands')
    plt.ylabel('loss value')
    plt.title('3D U-net validation curve for H129')          
#        plt.savefig(os.path.join(pth, 'val_zoom'), dpi = 300)
#    plt.close() 
    plt.figure()
    plt.plot(iters[start_iter:], loss[start_iter:], 'yo', iters[start_iter:], fit_fn(iters[start_iter:]), '--k')
    
    return linreg_stats