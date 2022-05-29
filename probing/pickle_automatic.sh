#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH --mem=20G                      
#SBATCH -p evlab     
module load openmind/anaconda/3-2019.10
python pickle_newsplit_check.py 24 'active-active' 'AAN-AAR' 'EventsAdapt' 'gpt2-xl'
