#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:4
#SBATCH --contiguous
#SBATCH --mem=14000 #14gbs
#SBATCH -t 8500                 # time (minutes)
#SBATCH -o cnn_train_20181115_part2_%j.out
#SBATCH -e cnn_train_20181115_part2_%j.err

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda/5.2.0
. activate 3dunet_py3
python run_exp.py 20181115_zd_train models/RSUNet.py samplers/soma.py augmentors/flip_rotate.py --batch_sz 20 --chkpt_num 281900 --gpus 0,1,2,3
