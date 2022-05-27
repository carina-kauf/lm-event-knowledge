#!/bin/bash                         
#SBATCH -t 03:00:00
#SBATCH -n 1                      
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=20G
#SBATCH -p evlab     
module load openmind/anaconda/3-2019.10       
python load_embeddings.py 'bert-large-cased' 'EventsRev'
