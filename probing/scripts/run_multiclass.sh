#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -p evlab 
#module load openmind/anaconda/3-2019.10

python pickle_linear_human_multiclass.py 24 'normal' 'normal' 'DTFit' 'gpt2-xl' > gpt2-xl_3classes_log.txt
python pickle_linear_human_multiclass.py 24 'normal' 'normal' 'EventsRev' 'gpt2-xl' >> gpt2-xl_3classes_log.txt
python pickle_linear_human_multiclass.py 24 'normal' 'normal' 'EventsAdapt' 'gpt2-xl' >> gpt2-xl_3classes_log.txt

python pickle_linear_human_multiclass.py 24 'active-active' 'AAN-AAN' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt
python pickle_linear_human_multiclass.py 24 'active-active' 'AAN-AAR' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt
python pickle_linear_human_multiclass.py 24 'active-active' 'AAN-AI' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt
python pickle_linear_human_multiclass.py 24 'active-active' 'AI-AAN' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt
python pickle_linear_human_multiclass.py 24 'active-active' 'AI-AAR' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt
python pickle_linear_human_multiclass.py 24 'active-active' 'AI-AI' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt

python pickle_linear_human_multiclass.py 24 'active-active' 'normal' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt
python pickle_linear_human_multiclass.py 24 'active-passive' 'normal' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt
python pickle_linear_human_multiclass.py 24 'passive-active' 'normal' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt
python pickle_linear_human_multiclass.py 24 'passive-passive' 'normal' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt

python pickle_linear_human_multiclass.py 24 'active-active' 'normal-AAR' 'EventsAdapt' 'gpt2-xl'>> gpt2-xl_3classes_log.txt
