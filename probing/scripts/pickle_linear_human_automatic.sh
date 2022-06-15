#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -p evlab 
module load openmind/anaconda/3-2019.10
python pickle_human_predict_plausibility.py 24 'active-active' 'normal-AAR' 'EventsAdapt' 'bert-large-cased'

