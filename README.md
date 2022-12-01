# Event knowledge in large language models: the gap between the impossible and the unlikely

By Carina Kauf*, Anna A. Ivanova*, Giulia Rambelli, Emmanuele Chersoni, Jingyuan S. She, Zawad Chowdhury, Evelina Fedorenko, Alessandro Lenci

*(the two lead authors contributed equally to this work)*

## Directory structure

Local directories:
* **analyses**: main analysis scripts and results
* **model_scores**: sentence scores for all datasets and all models (ported from **Event_Knowledge_Model_Comparison**)
* **probing**: code and results for classifier probing results in LLMs
* **sentence_info**: basic sentence features, such as length and word/phrase frequency

Submodules:
* **Event_Knowledge_Model_Comparison**: code to extract model scores 
* **beh-ratings-events**: human ratings

## Dataset name aliases
Dataset 1 - EventsAdapt (based on Fedorenko et al, 2020)

Dataset 2 - DTFit (based on Vassallo et al, 2018)

Dataset 3 - EventsRev (based on Ivanova et al, 2021)

### TODO
* **Score files**: Right now we're moving the score files between Event_Knowledge_Model_Comparison/results and model_scores via a script (because of the way in which we read the model scores in the analysis scripts) > streamline. [note: results last moved on 2022-10-21]
* **tinyLSTM scores**: are sums of token surprisal. Length-normalized scores can be obtained via dividing the score by the number of tokens (white-space separated in the penultimate column in the tinyLSTM score files)
