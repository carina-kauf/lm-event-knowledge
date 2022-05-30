import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sys
import pickle

layer_num = int(sys.argv[1])
voice_type = str(sys.argv[2])
sentence_type = str(sys.argv[3])
dataset_name = str(sys.argv[4])
model_name = str(sys.argv[5])

with open(f'/om2/user/jshe/lm-event-knowledge/probing/sentence_embeddings/{dataset_name}_{model_name}.pickle', 'rb') as f:
    hidden_states = pickle.load(f)

#print('check if it matches pickle output', list(hidden_states.values())[0][0])

def get_vector(sentence, layer):
  if model_name == 'gpt2-xl':
      sentence_embedding = hidden_states[sentence][layer][0][-1]
  else:
      sentence_embedding = hidden_states[sentence][layer][0][0]
  return sentence_embedding.numpy()

def break_array_tolist(array):
  a = []
  for item in array:
    for it in item:
      a.append(it)
  return a

def split_dataset_sentence(fold, dataset, voice_type, sentence_type):
  AI_sentences = dataset[dataset['TrialType'] == 'AI']
  AI_sentences = AI_sentences[AI_sentences['Voice'] == 'active']
  AI_sentences = AI_sentences.reset_index(drop = True)

  pool_ai = np.array_split(AI_sentences.index.values, fold_num)
  train_index_ai = break_array_tolist(pool_ai[:fold] + pool_ai[fold+1:])
  train_ai = pd.concat(AI_sentences[AI_sentences.index == j] for j in train_index_ai)
  train_ai = train_ai.reset_index(drop = True)

  test_index_ai = [x for x in AI_sentences.index.values if x not in train_index_ai]
  test_ai = pd.concat((AI_sentences[AI_sentences.index == j] for j in test_index_ai))

  AAN_sentences = dataset[dataset['TrialType'] == 'AAN']
  AAN_sentences = AAN_sentences[AAN_sentences['Voice'] == 'active']
  AAN_sentences = AAN_sentences.reset_index(drop = True)

  pool_aan = np.array_split(AAN_sentences.index.values, fold_num)
  train_index_aan = break_array_tolist(pool_aan[:fold] + pool_aan[fold+1:])
  train_aan = pd.concat(AAN_sentences[AAN_sentences.index == j] for j in train_index_aan)
  train_aan = train_aan.reset_index(drop = True)

  test_index_aan = [x for x in AAN_sentences.index.values if x not in train_index_aan]
  test_aan = pd.concat((AAN_sentences[AAN_sentences.index == j] for j in test_index_aan))
  
  AAR_sentences = dataset[dataset['TrialType'] == 'AAR']
  AAR_sentences = AAR_sentences[AAR_sentences['Voice'] == 'active']
  AAR_sentences = AAR_sentences.reset_index(drop = True)

  if sentence_type == 'AI-AI':
    train = train_ai
    train = train.reset_index(drop = True)
    test = test_ai
    test = test.reset_index(drop = True)
  elif sentence_type == 'AI-AAN':
    train = train_ai
    train = train.reset_index(drop = True)
    test = AAN_sentences
    test = test.reset_index(drop = True)
  elif sentence_type == 'AAN-AAN':
    train = train_aan
    train = train.reset_index(drop = True)
    test = test_aan
    test = test_aan.reset_index(drop = True)  
  elif sentence_type == 'AAN-AI':
    train = train_aan
    train = train.reset_index(drop = True)
    test = AI_sentences
    test = test.reset_index(drop = True)
  elif sentence_type == 'AI-AAR':
    train = train_ai
    train = train.reset_index(drop = True)
    test = AAR_sentences
    test = test.reset_index(drop = True)
  elif sentence_type == 'AAN-AAR':
    train = train_aan
    train = train.reset_index(drop = True)
    test = AAR_sentences
    test = test.reset_index(drop = True)
  elif sentence_type == 'normal-AAR':
    train = pd.concat([train_ai, train_aan])
    train = train.reset_index(drop = True)
    test = AAR_sentences
    test = test.reset_index(drop = True)    
  else:
    pass
  
  return train, test

def split_dataset(fold, dataset, voice_type, sentence_type):
  #active/passive conditions
  if sentence_type == 'normal':
    if dataset_name == 'EventsAdapt':
      dataset = dataset[dataset.TrialType != "AAR"]
      dataset = dataset.reset_index()
    else: 
      pass

    unique_pairs_num = dataset['ItemNum'].nunique()
    unique_index = dataset['ItemNum'].unique()

    shuffled_unique_index = np.random.permutation(unique_index)
    pool = np.array_split(shuffled_unique_index, fold_num)

    test_index = pool[fold]
    train_index = [i for i in unique_index if i not in test_index]
    test = pd.concat((df_DT[df_DT['ItemNum'] == j] for j in test_index))
    train = pd.concat((df_DT[df_DT['ItemNum'] == k] for k in train_index))

    test = test.reset_index()
    train = train.reset_index()

    if voice_type == 'active-active':
      train = train[train['Voice'] == 'active']
      test = test[test['Voice'] == 'active']
      test = test.reset_index(drop = True)
      train = train.reset_index(drop = True)
    elif voice_type == 'passive-passive':
      train = train[train['Voice'] == 'passive']
      test = test[test['Voice'] == 'passive']     
      test = test.reset_index(drop = True)
      train = train.reset_index(drop = True)
    elif voice_type == 'active-passive':
      train = train[train['Voice'] == 'active']
      test = test[test['Voice'] == 'passive']
      test = test.reset_index(drop = True)
      train = train.reset_index(drop = True)
    elif voice_type == 'passive-active':
      train = train[train['Voice'] == 'passive']
      test = test[test['Voice'] == 'active']
      test = test.reset_index(drop = True)
      train = train.reset_index(drop = True)
    else:
      pass
  #ai/aan/aar splitting conditions
  else:
    train, test = split_dataset_sentence(reg_trial, df_DT, voice_type, sentence_type)
  
  return train, test

dataset = f'/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_{dataset_name}_SentenceSet.csv'
output_path = f'/om2/user/jshe/lm-event-knowledge/probing/parallel_layers/new_layers/{model_name}_{dataset_name}_{voice_type}_{sentence_type}.csv'
df_DT = pd.read_csv(dataset)

fold_num = 10
np.random.seed(42)
if dataset_name == 'EventsAdapt':
  df_DT.loc[df_DT['TrialType'] == 'AAR', 'Plausibility'] = 'Plausible'

######for testing only
#df_DT = df_DT.head(10)

out = []
for layer in range(layer_num):
  Accuracy = []
  for reg_trial in range(fold_num):
    #changed
    train, test = split_dataset(reg_trial, dataset, voice_type, sentence_type)

    x_train = []
    for i in range(len(train)):
      x_train.append(get_vector(train['Sentence'][i], layer))
    x_train = np.array(x_train)

    x_test = []
    for j in range(len(test)):
      x_test.append(get_vector(test['Sentence'][j], layer))
    x_test = np.array(x_test)

    y_train = np.array(train["Plausibility"])
    y_test = np.array(test["Plausibility"])

    # Fitting regression
    logreg = LogisticRegression(max_iter=500, solver='liblinear')
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    Accuracy.append(metrics.accuracy_score(y_test, y_pred))
  print('layer', layer_num, statistics.mean(Accuracy))
  print(Accuracy)
  out.append(Accuracy)
  
df_out = pd.DataFrame(out)
df_out.to_csv(output_path, header = False)
