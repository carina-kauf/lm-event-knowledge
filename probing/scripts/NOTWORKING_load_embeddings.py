#load_embedding 
import numpy as np
import torch
import pandas as pd
import sys
import pickle

from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, BertModel, RobertaModel, GPT2LMHeadModel

dic_tokenizers = {'bert-large-cased': BertTokenizer.from_pretrained('bert-large-cased'), 'roberta-large': RobertaTokenizer.from_pretrained('roberta-large'), 'gpt2-xl': GPT2Tokenizer.from_pretrained('gpt2-xl')}
dic_models = {'bert-large-cased': BertModel.from_pretrained('bert-large-cased', output_hidden_states=True), 'roberta-large': RobertaModel.from_pretrained('roberta-large', output_hidden_states=True), 'gpt2-xl': GPT2LMHeadModel.from_pretrained('gpt-xl', output_hidden_states=True)}

model_name = str(sys.argv[1])
dataset_name = str(sys.argv[2])

dataset = pd.read_csv(f'/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_{dataset_name}_SentenceSet.csv')

tokenizer = dic_tokenizers[model_name]
model = dic_models[model_name]
model.eval()

def get_vector(sentence):
  tokenized_text = tokenizer.tokenize(tokenizer.cls_token + sentence + tokenizer.sep_token)
  tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
  with torch.no_grad():
      outputs = model(tensor_input)
      # `hidden_states` has shape [24 x 1 x 22 x 1024]
      hidden_states = outputs[2]
  return hidden_states

#check_sentence = get_vector(dataset['Sentence'][0])
#print(check_sentence)

embedding_dict = {}
for i in range(len(dataset['Sentence'])):
  embedding_dict[dataset['Sentence'][i]] = get_vector(dataset['Sentence'][i])

with open(f'/om2/user/jshe/lm-event-knowledge/probing/sentence_embeddings/{dataset_name}_{model_name}_2.pickle', 'wb') as handle:
  pickle.dump(embedding_dict, handle)

