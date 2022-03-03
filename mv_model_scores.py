"""
Move files from model score creation directory (Event_Knowledge_Model_Comparison) to model_score directory
"""

import glob, os
import shutil

os.makedirs('model_scores', exist_ok=True)

name2exp = {
    'ev1': 'EventsRev',
    'new-EventsAdapt': 'EventsAdapt',
    'dtfit': 'DTFit'
}

for exp in ['ev1', 'new-EventsAdapt', 'dtfit']:
    score_dir = f'model_scores/{name2exp[exp]}'
    os.makedirs(score_dir, exist_ok=True)
    files = glob.glob(f'Event_Knowledge_Model_Comparison/results/**/{exp}.*', recursive=True)
    print(len(files))
    for file in files:
        shutil.copy(file, score_dir)
