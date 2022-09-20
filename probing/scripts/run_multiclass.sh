#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -p evlab 
#module load openmind/anaconda/3-2019.10

python regression_model2human_multiclass.py.py 'gpt-j' 'DTFit' 'normal' 'normal' > gpt-j_3classes_log.txt
python regression_model2human_multiclass.py.py 'gpt-j' 'EventsRev' 'normal' 'normal'  >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'normal' 'normal' >> gpt-j_3classes_log.txt

python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'active-active' 'AAN-AAN' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt''active-active' 'AAN-AAR' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'active-active' 'AAN-AI' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'active-active' 'AI-AAN' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'active-active' 'AI-AAR' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'active-active' 'AI-AI' >> gpt-j_3classes_log.txt

python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'active-active' 'normal' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'active-passive' 'normal' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'passive-active' 'normal' >> gpt-j_3classes_log.txt
python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'passive-passive' 'normal' >> gpt-j_3classes_log.txt

python regression_model2human_multiclass.py.py 'gpt-j' 'EventsAdapt' 'active-active' 'normal-AAR' >> gpt-j_3classes_log.txt
