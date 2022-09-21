#!/bin/bash                         
#SBATCH -t 05:00:00
#SBATCH -n 1                      
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=20G

module load openmind/anaconda/3-2019.10       
python load_embeddings.py --model_name gpt-j --dataset_name DTFit #EventsAdapt EventsRev
