import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sys
import pickle

layernum = int(sys.argv[1])
layernum = layernum - 1
voice_type = str(sys.argv[2])
sentence_type = str(sys.argv[3])
dataset_name = str(sys.argv[4])
model_name = str(sys.argv[5])

with open(f'/om2/user/jshe/lm-event-knowledge/probing/sentence_embeddings/{dataset_name}_{model_name}.pickle', 'rb') as f:
    hidden_states = pickle.load(f)

print(list(hidden_states.values())[0][0])

def get_vector(sentence, layer):
    sentence_embedding = hidden_states[sentence][layer][0][-1]
    return sentence_embedding.numpy()

dataset = f'/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_{dataset_name}_SentenceSet.csv'
output_path = f'/om2/user/jshe/lm-event-knowledge/probing/parallel_layers/{model_name}_{dataset_name}_{voice_type}_{sentence_type}_{layernum}.csv'
df_DT = pd.read_csv(dataset)

# Read in files
if dataset_name == 'EventsAdapt':
  df_DT = df_DT[df_DT.TrialType != "AAR"]
  df_DT = df_DT.reset_index()
else: 
  pass

# Splitting
fold_num = 10
np.random.seed(42)
unique_pairs_num = df_DT['ItemNum'].nunique()
unique_index = df_DT['ItemNum'].unique()

shuffled_unique_index = np.random.permutation(unique_index)
pool = np.array_split(shuffled_unique_index, fold_num)

Accuracy = []
for reg_trial in range(fold_num):
  test_index = pool[reg_trial]
  print(test_index)
  train_index = [i for i in unique_index if i not in test_index]
  print(train_index)
  test = pd.concat((df_DT[df_DT['ItemNum'] == j] for j in test_index))
  print(test)
  train = pd.concat((df_DT[df_DT['ItemNum'] == k] for k in train_index))
  print(train)
  test = test.reset_index()
  train = train.reset_index()

  print("checkpoint1", train, test)
  if voice_type == 'active-active':
    train = train[train['Voice'] == 'active']
    test = test[test['Voice'] == 'active']
  elif voice_type == 'passive-passive':
    train = train[train['Voice'] == 'passive']
    test = test[test['Voice'] == 'passive']     
  elif voice_type == 'active-passive':
    train = train[train['Voice'] == 'active']
    test = test[test['Voice'] == 'passive']
  elif voice_type == 'passive-active':
    train = train[train['Voice'] == 'passive']
    test = test[test['Voice'] == 'active']
  else:
    pass
  print("checkpoint2", train, test)
  if sentence_type == 'AI-AI':
    train = train[train['TrialType'] == 'AI']
    test = test[test['TrialType'] == 'AI']
  elif sentence_type == 'AI-AAN':
    train = train[train['TrialType'] == 'AI']
    test = test[test['TrialType'] == 'AAN']
  elif sentence_type == 'AAN-AAN':
    train = train[train['TrialType'] == 'AAN']
    test = test[test['TrialType'] == 'AAN']
  elif sentence_type == 'AAN-AI':
    train = train[train['TrialType'] == 'AAN']
    test = test[test['TrialType'] == 'AI']
  else:
    pass
  print("checkpoint3", train, test)

  x_train = []
  for i in range(len(train)):
    x_train.append(get_vector(train['Sentence'][i], layernum))
  x_train = np.array(x_train)

  x_test = []
  for j in range(len(test)):
    x_test.append(get_vector(test['Sentence'][j], layernum))
  x_test = np.array(x_test)

  y_train = np.array(train["Plausibility"])
  y_test = np.array(test["Plausibility"])

  # Fitting regression
  logreg = LogisticRegression(max_iter=500, solver='liblinear')
  logreg.fit(x_train, y_train)
  y_pred = logreg.predict(x_test)
  Accuracy.append(metrics.accuracy_score(y_test, y_pred))
  print(statistics.mean(Accuracy))
  print(Accuracy)
df_out = pd.DataFrame(Accuracy)
df_out = df_out.transpose()
df_out.to_csv(output_path, header = False)

