# probing analysis

* **pickle_newsplit_automatic**: most up-to-date [embeddings --> plausibility] file. 
* commandline arguments: python pickle_newsplit_automatic.py [layer_num] [voice_type] [sentence_type] [dataset_name] [model_name]
* output files: 14 conditions per model = 42 csv files in total, named as probing/parallel_layers/new_layers/{model_name}_{dataset_name}_{voice_type}_{sentence_type}.csv
* newest train-test-split: AI-AI and AAN-AAN: train on 9/10, test on 1/10; AI-AAN/AAN-AI/AI-AAR/AAN-AAR: train on 9/10 of former, and test on all of later; normal-AAR: train on AI and AAN, test on all AAR
* fixed previous error of: non-all-zero first layer accuracies of AAR sentences (05/29/22, email). Reason: trialtypes were splitted with indices, not ItemNum. Now both VoiceType and TrialType are splitted using ItemNum, and AAR conditions all have 0 for the first layer.

