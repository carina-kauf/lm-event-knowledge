#!/bin/bash                         
#SBATCH -t 16:00:00
#SBATCH -n 1
#SBATCH --mem=20G                      
#SBATCH --gres=gpu:titan-x:1
#SBATCH -p evlab     
#SBATCH --array=1-49
module load openmind/anaconda/3-2019.10       
python gpt2_EventsRev_parallel.py $SLURM_ARRAY_TASK_ID 
