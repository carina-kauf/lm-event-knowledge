# check

### `check_token_same_embeddings.py`
* sanity-check: making sure that sentences of the same token length have the same embeddings
* output: should return True

### `pickle_linear_human_check_voice.py`
* y_test, y_pred, and the corresponding sentences to check the outputs of all four voice conditions
* output directory: probing/results/check_sentences_voice_ridge_regression

### `pickle_linear_human_check_trialtype.py`
* y_test, y_pred, and the corresponding sentences to check the outputs of all four sentence conditions
* output directory: probing/results/check_sentences_trialtype

### 'ceiling_AI-AAN_unclassified_sentences'
* slurm output to check why the human ceilings for AAN-AI is unanimously 0.7307692307692307. Turns out that it's actually the case.

### 'ceiling_AAN-AI_unclassified_sentences'
* no problems with this one, just checked as a contrast to AI-AAN
