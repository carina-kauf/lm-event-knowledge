#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH -p evlab 

source /om2/user/ckauf/anaconda/etc/profile.d/conda.sh
conda activate events3.8

echo "Environment ready!"

outfile="regression_model2human_${1}_log.txt"
python regression_model2human_multiclass.py --model_name ${1} > $outfile

# for model in gpt-j; do sbatch run_model2human.sh $model; done
# for model in gpt-j gpt2-xl roberta-large bert-large-cased; do sbatch run_model2human.sh $model; done
