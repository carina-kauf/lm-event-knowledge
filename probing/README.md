# probing

### `pickle_newsplit_automatic.py`
* most up-to-date [embeddings --> plausibility]. Metric: accuracy with logit regression
* run on commandline: `python pickle_newsplit_automatic.py [layer_num] [voice_type] [sentence_type] [dataset_name] [model_name]`
* output files: 14 conditions per model = 42 csv files in total, named as probing/parallel_layers/new_layers/{model_name}_{dataset_name}_{voice_type}_{sentence_type}.csv
* newest train-test-split: AI-AI and AAN-AAN: train on 9/10, test on 1/10; AI-AAN/AAN-AI/AI-AAR/AAN-AAR: train on 9/10 of former, and test on all of later; normal-AAR: train on AI and AAN, test on all AAR

> * issue 1: non-all-zero first layer accuracies of AAR sentences (05/29/22, email)
> * solved: trialtypes were splitted with indices, not ItemNum. Now both VoiceType and TrialType are splitted using ItemNum, and AAR conditions all have 0 for the first layer.

### `human_predict_plausibility.py`
* ceilings for model predictions [normed human ratings --> plausibility].  Metric: accuracy with logit regression
* run on commandline: `python human_predict_plausibility.py 24 [voice_type] [sentence_type] [dataset_name] bert-large-cased`
* output files: 14 conditions = 14 csv files in total, named as probing/parallel_layers/model_human_ceiling/bert-large-cased_{dataset_name}_{voice_type}_{sentence_type}.csv

> :warning: `layer_num` and `model_name` do not matter here, thus pre-set to `24` and 'bert-large-cased'.They were kept only because it's easier to combine for plotting with the current plotting code. Remember to confirm it's only 14 files when plotting. 

### `pickle_linear_human_mse.py`
* predicting human ratings [embeddings --> normed human ratings]. Metric: mean squared error with Ridge regression
* run on commandline: `python pickle_linear_human_mse.py [layer_num] [voice_type] [sentence_type] [dataset_name] [model_name]`
* output files: 14 conditions per model = 42 csv files in total, named as probing/parallel_layers/human_predictions/{model_name}_{dataset_name}_{voice_type}_{sentence_type}.csv

> * :construction: issue 2 unresolved: waiting to find a bounded regression method that works
> 
> * issue 1: weird-shaped mse plot (06/02/22, email). A huge bump in bert's plot, and very high mse values for gpt2
> 
> * solved: plotted y_pred against y_test (06/07, slack), and found that much of y_pred exceeds the (0,1) y_test bound, thus started looking for bounded regression method. 
> 
> * issue 2: `sm.GLM(y_train, x_train, family=sm.families.Binomial())` returned `statsmodels.tools.sm_exceptions.PerfectSeparationError: Perfect separation detected, results not available` and `sm.Logit(y_train, x_train)` returned `numpy.linalg.LinAlgError: Singular matrix`

### `pickle_linear_human_mse.py`
* predicting human ratings [embeddings --> normed human ratings]. Metric: R-squared with Ridge regression
* same as `pickle_linear_human_mse.py`. Only difference is the r^2 metric




