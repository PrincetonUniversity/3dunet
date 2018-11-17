### Using nicholasturner1's and torms3's PyTorchUtils, DataProvider3, DataTools, and Augmentor for the Wang lab cerebellar tracing project and Princeton BRAIN COGs histology core.

## Installation:

** Suggest you make an environment (in python 3+) for installing the following dependencies **

`conda create -n 3dunet python=3.5`

`pip install numpy scipy h5py matplotlib scikit-image cython torch torchvision (make sure it is torch 0.4+)`

`pip install tifffile tensorboardX`

If installing (locally) on a linux machine, make sure you have all the boost libraries (important for working with torms3's DataTools):

`sudo apt-get install libboost-all-dev` (this can take time)

`git clone https://github.com/torms3/DataTools.git`

Go to the dataprovider3, DataTools, and augmentor directories and run (for each directory):
`python setup.py install`

## Run:
- main scripts are located in the pytorchutils directory
- modify parameters in the respective run files
1. `run_exp.py` --> training
    - adjust patch/window size for training
    - adjust read and writing functions for data structures (default is HDF5)
    - adjust data and experiment directories
2. `run_fwd.py` --> inference
    - adjust patch/window size for inference (try to be consistent with training)
    - adjust read and writing functions for data structures (default is HDF5)
3. `run_chnk_fwd.py` --> large-scale inference
    - adjust patch/window size for inference (try to be consistent with training)
    - adjust read and writing functions for data structures (default is TIFF)
  
Contact: zmd@princeton.edu
