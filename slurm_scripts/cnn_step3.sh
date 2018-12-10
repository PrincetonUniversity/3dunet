#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 10                     # number of cores
#SBATCH -t 200                # time (minutes)
#SBATCH -o /scratch/zmd/logs/cnn_step3_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e /scratch/zmd/logs/cnn_step3_%j.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 100000 #100 gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/5.1.0
. activate lightsheet

echo "Experiment name:" "$@"

python cell_detect.py 3 0 "$@" 
