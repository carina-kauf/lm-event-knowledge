#!/bin/bash                         
#SBATCH -t 70:00:00
#SBATCH -n 1                      
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=20G
#SBATCH -p evlab     
#SBATCH --array=1-49       
module load openmind/anaconda/3-2019.10
python gpt2_DTFit_parallel.py $SLURM_ARRAY_TASK_ID '/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_EventsAdapt_SentenceSet.csv'

