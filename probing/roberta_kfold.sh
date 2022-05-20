#!/bin/bash                         
#SBATCH -t 05:00:00
#SBATCH -n 1                      
#SBATCH --gres=gpu:titan-x:1
#SBATCH -p evlab     
#SBATCH --array=1-25
module load openmind/anaconda/3-2019.10       
python roberta_kfold.py $SLURM_ARRAY_TASK_ID 'normal' 'normal' '/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_DTFit_SentenceSet.csv'
