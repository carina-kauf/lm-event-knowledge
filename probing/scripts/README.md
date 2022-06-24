# scripts

The analysis has four parts:

## 1. loading the embeddings and saving them as pickle files

This is a new approach. We had previously extracted embeddings directly in the scripts, that is, to tokenize and get the hidden layers for every sentence. This approach took significant amount of time even with parallel coding, which was approximately ~60-80 minutes per layer. We then decided to first save the hidden layers of all sentences as pickle files, and load them in each script. The scripts to load these embeddings are below:

### `load_embeddings.py`
* load embeddings for specified model and dataset
* run on commandline: `python load_embeddings.py [model_name] [dataset_name]`
* output files: probing/sentence_embeddings/{dataset_name}_{model_name}.pickle

> :warning: the probing/sentence_embeddings directory with all the pickle files are too large to push, so you'd need to run the above loading scipts on your local machine, and make sure to `mkdir sentence_embeddings` so the pickle fileswill be saved as probing/sentence_embeddings/{dataset_name}_{model_name}.pickle. Doing so will allow you to load them with the current code in `pickle_newsplit_automatic.py` and `pickle_linear_human_mse.py`.

## 2. Predicting plausibilities from model embeddings

This file reads the pre-loaded pickle files, and outputs accuracy matrices of the classification accuracies for all 24/48 layers and all 10 folds. It takes five commandline arguments: layer_num, voice_type, sentence_type, dataset_name, and model_name. Details below:

### `pickle_newsplit_automatic.py`
* most up-to-date [embeddings --> plausibility]. Metric: accuracy with logit regression
* run on commandline: `python pickle_newsplit_automatic.py [layer_num] [voice_type] [sentence_type] [dataset_name] [model_name]`
* output files: 14 conditions per model = 42 csv files in total, named as probing/parallel_layers/new_layers/{model_name}_{dataset_name}_{voice_type}_{sentence_type}.csv
* a combined file of the above 42 conditions (with human ceiling columns): probing/parallel_layers/model_plausibility_with_ceilings.csv
* newest train-test-split: AI-AI and AAN-AAN: train on 9/10, test on 1/10; AI-AAN/AAN-AI/AI-AAR/AAN-AAR: train on 9/10 of former, and test on all of later; normal-AAR: train on AI and AAN, test on all AAR

> * issue 1: non-all-zero first layer accuracies of AAR sentences (05/29/22, email)
> * solved: trialtypes were splitted with indices, not ItemNum. Now both VoiceType and TrialType are splitted using ItemNum, and AAR conditions all have 0 for the first layer.

## 3. Creating ceilings for model predictions with human ratings

Classification performances across conditions are varied. Therefore, it's good to establish a ceiling (gold standard) for each condition, which would be the classification accuracies of human ratings --> plausibility. 

### `human_predict_plausibility.py`
* ceilings for model predictions [normed human ratings --> plausibility].  Metric: accuracy with logit regression
* run on commandline: `python human_predict_plausibility.py 24 [voice_type] [sentence_type] [dataset_name] bert-large-cased`
* output files: 14 conditions = 14 csv files in total, named as probing/parallel_layers/model_human_ceiling/bert-large-cased_{dataset_name}_{voice_type}_{sentence_type}.csv
* a combined file of the above 14 conditions: probing/parallel_layers/human_plausibility_ceilings.csv
* also outputs: `check.csv` in probing/. Sanity check of the dataframe: whether plausible/implausible is converted to 1/0, and whether AAR contains all 1's.
* also outputs (in slurm.out): printed train, test dataframes and unclassified sentences across folds

> `layer_num` and `model_name` do not matter here, thus pre-set to `24` and 'bert-large-cased'.They were kept only because it's easier to combine for plotting with the current plotting code. Remember to confirm it's only 14 files when plotting. 

## 4. Predicting human scores with model embeddings

This is a work-in-progress. We want to see how well the embeddings can predict normed human scores. We originally used Ridge regression and recorded the mean squared error or r^2 as a metric for classification accuracy. However, the plot turned out weird (issue 1) so we want to switch to another regression that bounds the output to (0,1). This is currently unresolved (issue 2). 

### `pickle_linear_human_mse.py`
* predicting human ratings [embeddings --> normed human ratings]. Metric: mean squared error with Ridge regression
* run on commandline: `python pickle_linear_human_mse.py [layer_num] [voice_type] [sentence_type] [dataset_name] [model_name]`
* output files: 14 conditions per model = 42 csv files in total, named as probing/parallel_layers/human_predictions/{model_name}_{dataset_name}_{voice_type}_{sentence_type}.csv

> 
> * issue 1: weird-shaped mse plot (06/02/22, email). A huge bump in bert's plot, and very high mse values for gpt2
> 
> * solved: plotted y_pred against y_test (06/07, slack), and found that much of y_pred exceeds the (0,1) y_test bound, thus started looking for bounded regression method. 
> 
> * issue 2: `sm.GLM(y_train, x_train, family=sm.families.Binomial())` returned `statsmodels.tools.sm_exceptions.PerfectSeparationError: Perfect separation detected, results not available` and `sm.Logit(y_train, x_train)` returned `numpy.linalg.LinAlgError: Singular matrix`
> 
> * :construction: issue 2 unresolved: waiting to find a bounded regression method that works



