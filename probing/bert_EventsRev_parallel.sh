#!/bin/bash                         
#SBATCH -t 04:00:00
#SBATCH -n 1                      
#SBATCH --gres=gpu:titan-x:1
#SBATCH -p evlab     
#SBATCH --array=1-24       
python bert_EventsRev_parallel.py $SLURM_ARRAY_TASK_ID
