#!/bin/env bash
#
#SBATCH -p all                # partition (queue)
#SBATCH -c 8                      # number of cores
#SBATCH -t 200                # time (minutes)
#SBATCH -o logs/cnn_step0.out        # STDOUT #add _%a to see each array job
#SBATCH -e logs/cnn_step0.err        # STDERR #add _%a to see each array job
#SBATCH --contiguous #used to try and get cpu mem to be contigous
#SBATCH --mem 25000 #25 gbs

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

cat /proc/$$/status | grep Cpus_allowed_list

module load anacondapy/5.1.0
. activate lightsheet

python cell_detect.py /jukebox/wang/pisano/tracing_output/antero_4x/20180327_jg42_bl6_lob6a_05 0

