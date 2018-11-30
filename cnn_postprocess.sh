#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 1                      # number of cores
#SBATCH -t 10                # time (minutes)
#SBATCH -o logs/cnn_postprocess_%j.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/cnn_postprocess_%j.err        # STDERR #add _%a to see each array job

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list
cat /proc/meminfo

module load anacondapy/5.1.0
. activate lightsheet

#generate memmap array of reconstructed cnn output
OUT0=$(sbatch slurm_scripts/cnn_step2.sh "$@") 
echo $OUT0

#generate cell centers and measurements
OUT1=$(sbatch --dependency=afterany:${OUT0##* } slurm_scripts/cnn_step3.sh "$@") 
echo $OUT1

#functionality
#go to 3dunet main directory and type sbatch sub_cnn_postprocess.sh [path to lightsheet package output directory]
