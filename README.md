Using nicholasturner1's and torms3's PyTorchUtils, DataProvider3, DataTools, Augmentor for the Wang lab cerebellar tracing project and Princeton BRAIN COGs histology core.

Installation:

Suggest you make an environment (in python 3+) for installing the following dependencies

conda create -n 3dunet python=3.5
pip install numpy scipy h5py matplotlib scikit-image cython torch torchvision (make sure it is torch 0.4+)
pip install tifffile tensorboardX

If installing (locally) on a linux machine, make sure you have all the boost libraries (important for working with torms3's DataTools):
sudo apt-get install libboost-all-dev (this can take time)

In the main repo, git clone https://github.com/torms3/DataTools.git

Go to the dataprovider3, DataTools, and augmentor directories and run (for each directory):
python setup.py install

The main run scripts are located in the pytorchutils directory.

Modify parameters in the respective run files.

'run_exp.py' --> training

'run_fwd.py' --> inference

'run_chnk_fwd.py' --> large-scale inference

Contact: zmd@princeton.edu
