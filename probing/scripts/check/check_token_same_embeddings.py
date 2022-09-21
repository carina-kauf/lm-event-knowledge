from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')

config = GPT2Config.from_pretrained('gpt2-xl', output_hidden_states=True)

model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

import torch

import numpy as np

sents = ["The criminal arrested the cop.",
"The cop arrested the criminal."]

sentence_activations = []

for sent in sents:

  tokenized_text = tokenizer.tokenize(tokenizer.eos_token + sent + tokenizer.eos_token)

  tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])

  print(tensor_input)

  with torch.no_grad():

    outputs = model(tensor_input, output_hidden_states=True)

    print(outputs.keys())

# `hidden_states` has shape [7 x 1 x 7 x 768]

    hidden_states = outputs[2]



    sentence_activations.append(hidden_states)
torch.equal(sentence_activations[0][0][0][0],sentence_activations[1][0][0][0])

