#!/bin/bash
#SBATCH -p all                # partition (queue)
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:1
#SBATCH --contiguous
#SBATCH --mem=3000 #30gbs
#SBATCH -t 5                # time (minutes)
#SBATCH -o /scratch/gpfs/zmd/logs/chnk_%a_%j.out
#SBATCH -e /scratch/gpfs/zmd/logs/chnk_%a_%j.err

echo "Array Index: $SLURM_ARRAY_TASK_ID"

cd pytorchutils/

module load cudatoolkit/10.0 cudnn/cuda-10.0/7.3.1 anaconda/5.2.0
. activate 3dunet_py3
python run_chnk_fwd.py 20181115_zd_train models/RSUNet.py 300000 20170212_tp_bl6_crii_rlat_04 --gpus 0 --noeval --tag noeval ${SLURM_ARRAY_TASK_ID}
