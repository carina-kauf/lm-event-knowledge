#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mem 20G
#SBATCH -n 1
#SBATCH -p evlab 

source /om2/user/ckauf/anaconda/etc/profile.d/conda.sh
conda activate events3.8
pip install ordinal
pip install sklearn
pip install tqdm

python regression_model2human_multiclass.py 'gpt-j' 'DTFit' 'normal' 'normal' > gpt-j_3classes_log.txt
python regression_model2human_multiclass.py 'gpt-j' 'EventsRev' 'normal' 'normal'  >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'normal' 'normal' >> gpt-j_3classes_log.txt

python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'active-active' 'AAN-AAN' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt''active-active' 'AAN-AAR' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'active-active' 'AAN-AI' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'active-active' 'AI-AAN' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'active-active' 'AI-AAR' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'active-active' 'AI-AI' >> gpt-j_3classes_log.txt

python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'active-active' 'normal' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'active-passive' 'normal' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'passive-active' 'normal' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'passive-passive' 'normal' >> gpt-j_3classes_log.txt

python regression_model2human_multiclass.py 'gpt-j' 'EventsAdapt' 'active-active' 'normal-AAR' >> gpt-j_3classes_log.txt
