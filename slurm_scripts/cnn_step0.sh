#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 8                      # number of cores
#SBATCH -t 20                # time (minutes)
#SBATCH -o logs/testcnn_step0.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/testcnn_step0.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 25000 #25 gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/5.1.0
. activate lightsheet

echo "Experiment name:" "$@"

python cell_detect.py "$@" 0

