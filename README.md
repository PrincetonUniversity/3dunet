### Three-dimensional CNN with a U-Net architecture (Gornet et al., 2019; K. Lee, Zung, Li, Jain, & Sebastian Seung, 2017) with added packages developed by Kisuk Lee (Massachusetts Institute of Technology), Nick Turner (Princeton University), James Gornet (Columbia University), and Kannan Umadevi Venkatarju (Cold Spring Harbor Laboratories) for the Wang lab cerebellar tracing project and the Princeton BRAIN CoGS histology core.

#### Dependencies
| [DataProvider3](https://github.com/torms3/DataProvider3)  | 
| [PyTorchUtils](https://github.com/nicholasturner1/PyTorchUtils)  | 
| [Augmentor](https://github.com/torms3/Augmentor)  | 
| [DataTools](https://github.com/torms3/DataTools)  | 

## Installation:

**NOTE**: preprocessing done using tpisano's lightsheet package (https://github.com/PrincetonUniversity/lightsheet).

**For running inference, make an environment (in python 3+) and install following dependencies**

- `conda create -n 3dunet python=3.5`
- `pip install numpy scipy h5py matplotlib pandas scikit-image cython torch torchvision` (make sure it is torch 0.4+)
- `pip install tifffile tensorboardX`

If installing (locally) on a linux machine, make sure you have all the boost libraries (important for working with torms3's DataTools):
`sudo apt-get install libboost-all-dev` (this can take time)

Clone the necessary C++ extension scripts for working with DataProvider3:
`git clone https://github.com/torms3/DataTools.git`

Go to the dataprovider3, DataTools, and augmentor directories and run (for each directory):
`python setup.py install`

## Run:
- main GPU-based scripts are located in the pytorchutils directory (python 3+)
1. `run_exp.py` --> training
    - lines 64-98: modify data directory, train and validation sets, and named experiment   	  directory (in which the experiment directory of logs and model weights is stored) 
2. `run_fwd.py` --> inference
    - lines 57 & 65: modify experiment and data directory 
3. `run_chnk_fwd.py` --> large-scale inference
    - lines 82 & 90: modify experiment and data directory 
    - if working with a slurm-based scheduler:
	1. modify `run_chnk_fwd.sh` in the main repo
	2. use `python pytorchutils/run_chnk_fwd.py -h` for more info on command line 		arguments
4. modify parameters (stride, window, # of iterations, etc.) in the main parameter dictionaries
- `cell_detect.py` --> CPU-based pre-processing and post-processing based on tpisano's lightsheet package (python 2+)
    - if working with a slurm-based scheduler, 
	1. `cnn_preprocess.sh` --> chunks full sized data from specified directory  
	2. `cnn_postprocess.sh` --> reconstructs and uses connected components to find cell measures
    - output is a '3dunet_output' directory containing a '[brain_name]_cell_measures.csv'

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
