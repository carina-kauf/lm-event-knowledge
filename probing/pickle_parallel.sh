#!/bin/bash                         
#SBATCH -t 00:30:00
#SBATCH -n 1                      
#SBATCH --gres=gpu:titan-x:1
#SBATCH -p evlab     
#SBATCH --array=1-49
module load openmind/anaconda/3-2019.10
python pickle_parallel.py $SLURM_ARRAY_TASK_ID 'normal' 'AAN-AI' 'EventsAdapt' 'gpt2-xl'



