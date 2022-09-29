#!/bin/bash                         
#SBATCH -t 40:00:00
#SBATCH -n 1                      
#SBATCH --mem=80G
#SBATCH -p evlab

echo 'Sourcing environment'
source /om2/user/ckauf/anaconda/etc/profile.d/conda.sh
conda activate events3.8

python load_embeddings.py --model_name gpt-j --dataset_name EventsAdapt
