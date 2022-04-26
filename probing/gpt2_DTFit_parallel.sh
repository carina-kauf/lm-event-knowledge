#!/bin/bash                         
#SBATCH -t 40:00:00
#SBATCH -n 1                      
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=20G
#SBATCH -p evlab     
#SBATCH --array=1-49       
module load openmind/anaconda/3-2019.10
python gpt2_DTFit_parallel.py $SLURM_ARRAY_TASK_ID
