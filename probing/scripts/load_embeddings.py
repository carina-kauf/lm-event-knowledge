#load_embedding 
import numpy as np
import torch
import pandas as pd
import sys
import pickle
import os
import os.path
import argparse

from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, BertModel, RobertaModel, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer

def get_vector(sentence, args):
    if 'gpt' in args.model_name:
        tokenized_text = tokenizer.tokenize(tokenizer.eos_token + sentence + tokenizer.eos_token)
    else:
        tokenized_text = tokenizer.tokenize(tokenizer.cls_token + sentence + tokenizer.sep_token)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
    with torch.no_grad():
        outputs = args.model(tensor_input)
        # `hidden_states` has shape [24 x 1 x 22 x 1024]
        hidden_states = outputs[2]
    return hidden_states

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--dataset_name', required=True, help="Should be one of {'DTFit', 'EventsRev', 'EventsAdapt'}")
    args = parser.parse_args()
    
    dic_tokenizers = {'bert-large-cased': BertTokenizer.from_pretrained('bert-large-cased'),
                  'roberta-large': RobertaTokenizer.from_pretrained('roberta-large'),
                  'gpt2-xl': GPT2Tokenizer.from_pretrained('gpt2-xl'),
                  'gpt-j': AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')}

    dic_models = {'bert-large-cased': BertModel.from_pretrained('bert-large-cased', output_hidden_states=True),
                  'roberta-large': RobertaModel.from_pretrained('roberta-large', output_hidden_states=True),
                  'gpt2-xl': GPT2LMHeadModel.from_pretrained('gpt2-xl', output_hidden_states=True),
                  'gpt-j': AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B', output_hidden_states=True)}

    dataset = pd.read_csv(os.path.abspath(f'../../analyses_clean/clean_data/clean_{args.dataset_name}_SentenceSet.csv'))
    
    tokenizer = dic_tokenizers[args.model_name]
    model = dic_models[args.model_name]
    model.eval()
    
    embedding_dict = {}
    for i in range(len(dataset['Sentence'])):
        embedding_dict[dataset['Sentence'][i]] = get_vector(dataset['Sentence'][i], args)
    
    savedir = os.path.abspath('../sentence_embeddings')
    os.makedirs(savedir, exist_ok=True)

    with open(f'{savedir}/{args.dataset_name}_{args.model_name}.pickle', 'wb') as handle:
        pickle.dump(embedding_dict, handle)
        
if __name__ == "__main__":
    main()
