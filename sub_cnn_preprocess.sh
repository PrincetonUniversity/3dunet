#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 1                      # number of cores
#SBATCH -t 10                # time (minutes)
#SBATCH -o logs/cnn_preprocess.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/cnn_preprocess.err        # STDERR #add _%a to see each array job

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list
cat /proc/meminfo

module load anacondapy/5.1.0
. activate lightsheet

#generate memmap array of full size cell channel data
OUT0=$(sbatch slurm_scripts/cnn_step0.sh "$@") 
echo $OUT0

#generate chunks for cnn input
OUT1=$(sbatch --dependency=afterany:${OUT0##* } slurm_scripts/cnn_step1.sh "$@") 
echo $OUT1