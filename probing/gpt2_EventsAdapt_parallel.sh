#!/bin/bash                         
#SBATCH -t 70:00:00
#SBATCH -n 1                      
#SBATCH --mem=20G
#SBATCH --gres=gpu:1 --constraint=any-gpu
#SBATCH -p evlab     
#SBATCH --array=1-49       
module load openmind/anaconda/3-2019.10
python gpt2_EventsAdapt_parallel.py $SLURM_ARRAY_TASK_ID
