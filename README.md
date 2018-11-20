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

1. if working with a slurm-based scheduler:
	1. run `sbatch run_demo.sh` within the main repo directory
4. else, type within the main repo directory:
	1. `python setup_demo_script.py`
	2. navigate to the pytorchutils directory
	2. `python demo.py demo models/RSUNet.py samplers/demo_sampler.py augmentors/flip_rotate.py 10 --batch_sz 1 		   		--nobn --noeval --tag demo` 
5. output will be in a 'demo/cnn_output' subfolder (as a TIFF)
