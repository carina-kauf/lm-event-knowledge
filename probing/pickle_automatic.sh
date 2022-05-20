#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -n 1                      
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=20G
#SBATCH -p evlab     
module load openmind/anaconda/3-2019.10
python pickle_automatic.py 48 'normal' 'normal' 'EventsAdapt' 'gpt2-xl'
