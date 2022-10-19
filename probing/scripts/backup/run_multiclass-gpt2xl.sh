#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH --mem 30G
#SBATCH -n 1
#SBATCH -p evlab 

source /om2/user/ckauf/anaconda/etc/profile.d/conda.sh
conda activate events3.8
pip install ordinal
pip install sklearn
pip install tqdm

python regression_model2human_multiclass.py 'gpt2-xl' 'DTFit' 'normal' 'normal' > gpt2-xl_3classes_log.txt
python regression_model2human_multiclass.py 'gpt2-xl' 'EventsRev' 'normal' 'normal'  >> gpt2-xl_3classes_log.txt
python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'normal' 'normal' >> gpt2-xl_3classes_log.txt

python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'active-active' 'AAN-AAN' >> gpt2-xl_3classes_log.txt
python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt''active-active' 'AAN-AAR' >> gpt2-xl_3classes_log.txt
python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'active-active' 'AAN-AI' >> gpt2-xl_3classes_log.txt
python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'active-active' 'AI-AAN' >> gpt2-xl_3classes_log.txt
python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'active-active' 'AI-AAR' >> gpt2-xl_3classes_log.txt
python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'active-active' 'AI-AI' >> gpt2-xl_3classes_log.txt

python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'active-active' 'normal' >> gpt2-xl_3classes_log.txt
python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'active-passive' 'normal' >> gpt2-xl_3classes_log.txt
python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'passive-active' 'normal' >> gpt2-xl_3classes_log.txt
python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'passive-passive' 'normal' >> gpt2-xl_3classes_log.txt

python regression_model2human_multiclass.py 'gpt2-xl' 'EventsAdapt' 'active-active' 'normal-AAR' >> gpt2-xl_3classes_log.txt
