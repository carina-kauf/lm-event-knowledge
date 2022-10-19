import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
import os

from sklearn import metrics
import sys
import pickle

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def get_vector(sentence, layer):
    if 'gpt' in model_name:
        sentence_embedding = hidden_states[sentence][layer][0][-1]
    else:
        sentence_embedding = hidden_states[sentence][layer][0][0]
    return sentence_embedding.numpy()


def break_array_tolist(array):
    a = []
    for item in array:
        a.append(item)
    return a


def split_dataset_sentence(fold, dataset, voice_type, sentence_type):
    """
    Split the dataset by sentence type, returns train and test dataframes.

    Current train-test splits: AI-AI and AAN-AAN: train on 9/10, test on 1/10; AI-AAN/AAN-A  I/AI-AAR/AAN-AAR: train on 9/10 of former, and test on all of later; normal-AAR: train   on AI and AAN, test on all AAR.

    Parameters
    ---------
    fold: int
    default = 10
    dataset: pd.Dataframe
    default = df
    voice_type: str
    'normal', 'active-active', 'passive-passive', 'active-passive', 'passive-active'
    sentence_type: str
    'normal', 'AI-AI', 'AAN-AAN', 'AI-AAN', 'AAN-AI'
    """
    AI_sentences = dataset[dataset['TrialType'] == 'AI']
    AI_sentences = AI_sentences[AI_sentences['Voice'] == 'active']
    AI_sentences = AI_sentences.reset_index(drop = True)

    unique_index_ai = np.random.permutation(AI_sentences['ItemNum'].unique())
    pool_ai = np.array_split(unique_index_ai, fold_num)
    test_index_ai = break_array_tolist(pool_ai[fold])
    train_index_ai = [i for i in unique_index_ai if i not in test_index_ai]
    train_ai = pd.concat((AI_sentences[AI_sentences['ItemNum'] == j] for j in train_index_ai))
    test_ai = pd.concat((AI_sentences[AI_sentences['ItemNum'] == k] for k in test_index_ai))
    train_ai = train_ai.reset_index(drop = True)
    test_ai = test_ai.reset_index(drop = True)

    AAN_sentences = dataset[dataset['TrialType'] == 'AAN']
    AAN_sentences = AAN_sentences[AAN_sentences['Voice'] == 'active']
    AAN_sentences = AAN_sentences.reset_index(drop = True)

    unique_index_aan = np.random.permutation(AAN_sentences['ItemNum'].unique())
    pool_aan = np.array_split(unique_index_aan, fold_num)
    test_index_aan = pool_aan[fold]
    train_index_aan = [i for i in unique_index_aan if i not in test_index_aan]
    train_aan = pd.concat((AAN_sentences[AAN_sentences['ItemNum'] == j] for j in train_index_aan))
    test_aan = pd.concat((AAN_sentences[AAN_sentences['ItemNum'] == k] for k in test_index_aan))
    train_aan = train_aan.reset_index(drop = True)
    test_aan = test_aan.reset_index(drop = True)

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
    """
    Split the dataset by voice type, returns train and test dataframes.

    This split runs when sentence_type == 'normal', otherwise, it automatically triggers sp  lit_dataset_sentence() to only splitting by sentence types

    Parameters
    ---------
    fold: int
        default = 10
    dataset: pd.Dataframe
        default = df
    voice_type: str
        'normal', 'active-active', 'passive-passive', 'active-passive', 'passive-active'
    sentence_type: str
        'normal', 'AI-AI', 'AAN-AAN', 'AI-AAN', 'AAN-AI'
    """

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
    test = pd.concat((dataset[dataset['ItemNum'] == j] for j in test_index))
    train = pd.concat((dataset[dataset['ItemNum'] == k] for k in train_index))

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
        train, test = split_dataset_sentence(reg_trial, df, voice_type, sentence_type)

    return train, test

def split_dataset(fold, dataset, voice_type, sentence_type):
  """
  Split the dataset by voice type, returns train and test dataframes.
  
  This split runs when sentence_type == 'normal', otherwise, it automatically triggers split_dataset_sentence() to only splitting by sentence types

  Parameters
  ---------
  fold: int
        default = 10
  dataset: pd.Dataframe
        default = df
  voice_type: str
        'normal', 'active-active', 'passive-passive', 'active-passive', 'passive-active'
  sentence_type: str
        'normal', 'AI-AI', 'AAN-AAN', 'AI-AAN', 'AAN-AI'
  """

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
    test = pd.concat((dataset[dataset['ItemNum'] == j] for j in test_index))
    train = pd.concat((dataset[dataset['ItemNum'] == k] for k in train_index))

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
    train, test = split_dataset_sentence(reg_trial, df, voice_type, sentence_type)
  
  return train, test

dataset = os.path.abspath(f'../../analyses_clean/clean_data/clean_{dataset_name}_df.csv')
output_path = os.path.abspath(f'../results/model2human/{model_name}_{dataset_name}_{voice_type}_{sentence_type}.csv')
df = pd.read_csv(dataset, low_memory=False)

fold_num = 10
np.random.seed(42)

df = df[df['Metric'] == 'human']

# from collections import Counter
#print(Counter(df["Score"].apply(lambda x: round(x, 0))))
#ss

df = df.reset_index(drop = True)
######for testing only
#df = df.head(10)

out = []
for layer in tqdm(range(layer_num)):
  Accuracy = []
  for reg_trial in range(fold_num):
    train, test = split_dataset(reg_trial, df, voice_type, sentence_type)

    x_train = []
    for i in range(len(train)):
      x_train.append(get_vector(train['Sentence'][i], layer))
    x_train = np.array(x_train)

    x_test = []
    for j in range(len(test)):
      x_test.append(get_vector(test['Sentence'][j], layer))
    x_test = np.array(x_test)

    mapdic = {1:0,2:0,3:1,4:1,5:1,6:2,7:2}

    y_train = np.array(train["Score"].apply(lambda x: mapdic[round(x, 0)]))
    y_test = np.array(test["Score"].apply(lambda x: mapdic[round(x, 0)]))
    #y_train = np.array(train["Score"].apply(lambda x: round(x, 0)))
    #y_test = np.array(test["Score"].apply(lambda x: round(x, 0)))

    # Fitting classification

    svc_bal = SVC(kernel='linear', class_weight="balanced", probability=True)
    svc_imb = SVC(kernel='linear', class_weight=None, probability=True)
    clf = LogisticRegression(solver="saga",multi_class="multinomial",penalty="l1",max_iter=30,random_state=42)

    clf = svc_bal 
    clf.fit(x_train, y_train)


    #print(clf.__repr__())

    y_pred = clf.predict(x_test)
    print(classification_report(y_test, clf.predict(x_test)))
    Accuracy.append(accuracy_score(y_test, y_pred)) #clf.score(y_test, y_pred))

  print('layer', layer, statistics.mean(Accuracy), flush=True)
  print(Accuracy)
  out.append(Accuracy)
  
df_out = pd.DataFrame(out)
df_out.to_csv(output_path, header = False)