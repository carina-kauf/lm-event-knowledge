import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import pandas as pd
import sys
import pickle

model_name = 'gpt2-xl'
dataset_name = str(sys.argv[1])
dataset = pd.read_csv(f'/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_{dataset_name}_SentenceSet.csv')

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name,
                                    output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )
model.eval()

def get_vector(sentence):
  tokenized_text = tokenizer.tokenize(tokenizer.eos_token + sentence + tokenizer.eos_token)
  tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
  with torch.no_grad():
      outputs = model(tensor_input)
      # `hidden_states` has shape [24 x 1 x 22 x 1024]
      hidden_states = outputs[2]
  return hidden_states

embedding_dict = {}
for i in range(len(dataset['Sentence'])):
  embedding_dict[dataset['Sentence'][i]] = get_vector(dataset['Sentence'][i])

with open(f'/om2/user/jshe/lm-event-knowledge/probing/sentence_embeddings/{dataset_name}_{model_name}.pickle', 'wb') as handle:
  pickle.dump(embedding_dict, handle)
