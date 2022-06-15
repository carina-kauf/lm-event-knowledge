# load_embeddings

### `load_embeddings_bert.py`
* load bert embeddings
* run on commandline: `python load_embeddings_bert.py [model_name] [dataset_name]`
* output files: probing/sentence_embeddings/{dataset_name}_{model_name}.pickle

### `load_embeddings_roberta.py`
* load roberta embeddings
* run on commandline: `python load_embeddings_roberta.py [model_name] [dataset_name]`
* output files: probing/sentence_embeddings/{dataset_name}_{model_name}.pickle

### `load_embeddings_gpt2.py`
* load gpt2 embeddings
* run on commandline: `python load_embeddings_gpt2.py [model_name] [dataset_name]`
* output files: probing/sentence_embeddings/{dataset_name}_{model_name}.pickle

### `load_embeddings.py`
* An attempt to standarlize the process to load different models with dictionaries. Has some unfixed errors, but moved on to other priorities. The current pickle files were generated with the above three files.
