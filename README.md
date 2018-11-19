### Using nicholasturner1's and torms3's PyTorchUtils, DataProvider3, DataTools, and Augmentor for the Wang lab cerebellar tracing project and the Princeton BRAIN COGs histology core.

## Installation:

**NOTE**: preprocessing done using tpisano's lightsheet package (https://github.com/PrincetonUniversity/lightsheet).

**For running inference, make an environment (in python 3+) and install following dependencies**

- `conda create -n 3dunet python=3.5`
- `pip install numpy scipy h5py matplotlib scikit-image cython torch torchvision` (make sure it is torch 0.4+)
- `pip install tifffile tensorboardX`

If installing (locally) on a linux machine, make sure you have all the boost libraries (important for working with torms3's DataTools):
`sudo apt-get install libboost-all-dev` (this can take time)

Clone the necessary C++ extension scripts for working with DataProvider3:
`git clone https://github.com/torms3/DataTools.git`

Go to the dataprovider3, DataTools, and augmentor directories and run (for each directory):
`python setup.py install`

## Run:
- main scripts are located in the pytorchutils directory
- modify parameters in the respective run files
    - parameter dictionary
    - read and writing functions for data structures
    - data and experiment directories
1. `run_exp.py` --> training
2. `run_fwd.py` --> inference
3. `run_chnk_fwd.py` --> large-scale inference

Contact: tpisano@princeton.edu, zmd@princeton.edu

## Demo:
- demo script to run training and large-scale inference

1. navigate to the `demo.py` script in the pytorchutils directory
2. modify data directories in the parameter directory
3. initialise a tiff file of shape greater than 20, 192, 192 in z, y, x in a folder called 'input_patches'
4. on the command line in the pytorchutils directory, type: `python demo.py test models/RSUNet.py samplers/soma.py augmentors/flip_rotate.py 10 test --batch_sz 1 --nobn --noeval --tag demo`
5. find the output in the 'cnn_patches' subfolder in the specified data directory in parameter dictionary
