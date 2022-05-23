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

print(list(hidden_states.values())[0][0])
# check_sentence = pd.read_csv("f'/om2/user/jshe/lm-event-knowledge/probing/sentence_embeddings/{dataset_name}_{model_name}_check_sentence.csv")
# corresponding_sentence = np.array(list(hidden_states.values())[0][0])
# assert check_sentence == corresponding_sentence

def get_vector(sentence, layer):
    if model_name == 'gpt2-xl':
        sentence_embedding = hidden_states[sentence][layer][0][-1]
    else:
        sentence_embedding = hidden_states[sentence][layer][0][0]
    return sentence_embedding.numpy()

dataset = f'/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_{dataset_name}_SentenceSet.csv'
output_path = f'/om2/user/jshe/lm-event-knowledge/probing/parallel_layers/{model_name}_{dataset_name}_{voice_type}_{sentence_type}.csv'
df_DT = pd.read_csv(dataset)
######for testing only
#df_DT = df_DT.head(10)

# Read in files
if dataset_name == 'EventsAdapt' and sentence_type == 'normal':
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

out = []
for layer in range(layer_num):
  Accuracy = []
  for reg_trial in range(fold_num):
    test_index = pool[reg_trial]
    train_index = [i for i in unique_index if i not in test_index]
    test = pd.concat((df_DT[df_DT['ItemNum'] == j] for j in test_index))
    train = pd.concat((df_DT[df_DT['ItemNum'] == k] for k in train_index))
    test = test.reset_index()
    train = train.reset_index()

    print("before-voice-split", train, test)
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
    print("need-only-active", train, test)
    if sentence_type == 'AI-AI':
      train = train[train['TrialType'] == 'AI']
      test = test[test['TrialType'] == 'AI']
      test = test.reset_index(drop = True)
      train = train.reset_index(drop = True)
    elif sentence_type == 'AI-AAN':
      train = train[train['TrialType'] == 'AI']
      test = test[test['TrialType'] == 'AAN']
      test = test.reset_index(drop = True)
      train = train.reset_index(drop = True)
    elif sentence_type == 'AAN-AAN':
      train = train[train['TrialType'] == 'AAN']
      test = test[test['TrialType'] == 'AAN']
      test = test.reset_index(drop = True)
      train = train.reset_index(drop = True)
    elif sentence_type == 'AAN-AI':
      train = train[train['TrialType'] == 'AAN']
      test = test[test['TrialType'] == 'AI']
      test = test.reset_index(drop = True)
      train = train.reset_index(drop = True)
    else:
      pass
    print("after-sentence-split", train, test)

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
