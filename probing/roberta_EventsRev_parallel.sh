#!/bin/bash                         
#SBATCH -t 03:00:00
#SBATCH -n 1                      
#SBATCH --gres=gpu:1             
#SBATCH --constraint=any-gpu     
#SBATCH -p evlab     
#SBATCH --array=1-24       
python roberta_EventsRev_parallel.py $SLURM_ARRAY_TASK_ID
