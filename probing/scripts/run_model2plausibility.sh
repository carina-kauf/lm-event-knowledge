#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --mem 20G
#SBATCH -n 1
#SBATCH -p evlab 

source /om2/user/ckauf/anaconda/etc/profile.d/conda.sh
conda activate events3.8

python regression_model2plausibility.py 'gpt-j' 'DTFit' 'normal' 'normal' > gpt-j_model2plausibility_log.txt
python regression_model2plausibility.py 'gpt-j' 'EventsRev' 'normal' 'normal'  >> gpt-j_model2plausibility_log.txt
python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'normal' 'normal' >> gpt-j_model2plausibility_log.txt

python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'active-active' 'AAN-AAN' >> gpt-j_model2plausibility_log.txt
python regression_model2plausibility.py 'gpt-j' 'EventsAdapt''active-active' 'AAN-AAR' >> gpt-j_model2plausibility_log.txt
python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'active-active' 'AAN-AI' >> gpt-j_model2plausibility_log.txt
python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'active-active' 'AI-AAN' >> gpt-j_model2plausibility_log.txt
python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'active-active' 'AI-AAR' >> gpt-j_model2plausibility_log.txt
python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'active-active' 'AI-AI' >> gpt-j_model2plausibility_log.txt

python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'active-active' 'normal' >> gpt-j_model2plausibility_log.txt
python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'active-passive' 'normal' >> gpt-j_model2plausibility_log.txt
python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'passive-active' 'normal' >> gpt-j_model2plausibility_log.txt
python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'passive-passive' 'normal' >> gpt-j_model2plausibility_log.txt

python regression_model2plausibility.py 'gpt-j' 'EventsAdapt' 'active-active' 'normal-AAR' >> gpt-j_model2plausibility_log.txt
