#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 1                      # number of cores
#SBATCH -t 10                # time (minutes)
#SBATCH -o /scratch/gpfs/zmd/logs/cnn_step21_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/gpfs/zmd/logs/cnn_step21_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 5000 #5 gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anaconda3/5.3.1
. activate 3dunet

echo "Experiment name:" "$@"

python cell_detect.py 21 0 "$@" 
