"""
Move files from model score creation directory (Event_Knowledge_Model_Comparison) to model_score directory
"""

import glob, os
import shutil
import re

os.makedirs('model_scores', exist_ok=True)

name2exp = {
    'ev1': 'EventsRev',
    'new-EventsAdapt': 'EventsAdapt',
    'dtfit': 'DTFit'
}

variations = {
    'ev1': re.compile('ev1[\.\_]'),
    'dtfit': re.compile('dtfit\.|DTFit\_'),
    'new-EventsAdapt': re.compile('(new-|newsentences_)EventsAdapt')
}

for exp in ['ev1', 'new-EventsAdapt', 'dtfit']:
    score_dir = f'model_scores/{name2exp[exp]}'
    os.makedirs(score_dir, exist_ok=True)
    pat = variations[exp]

    files = [f for f in glob.glob(f'Event_Knowledge_Model_Comparison/results/**/*.txt', recursive=True) if re.search(pat, f)]
    print(len(files))
    for file in files:
        shutil.copy(file, score_dir)
