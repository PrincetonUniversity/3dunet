#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:2
#SBATCH --contiguous
#SBATCH --mem=14000 #14gbs
#SBATCH -t 8500                 # time (minutes)
#SBATCH -o logs/cnn_train_20181115_%j.out
#SBATCH -e logs/cnn_train_20181115_%j.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda/5.2.0
. activate 3dunet_py3
python run_exp.py 20181115_zd_train models/RSUNet.py samplers/soma.py augmentors/flip_rotate.py --batch_sz 16 --gpus 0,1
