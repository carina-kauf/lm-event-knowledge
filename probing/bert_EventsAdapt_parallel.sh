#!/bin/bash                         
#SBATCH -t 06:00:00
#SBATCH -n 1                      
#SBATCH --gres=gpu:titan-x:1
#SBATCH -p evlab     
#SBATCH --array=1-24       
python bert_EventsAdapt_parallel.py $SLURM_ARRAY_TASK_ID
