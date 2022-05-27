import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
import sys

layernum = int(sys.argv[1])
layernum = layernum - 1
voice_type = str(sys.argv[2])
sentence_type = str(sys.argv[3])
dataset_input = str(sys.argv[4])

def output(model_name, voice, sentence, dataset_name):
  tokenizer = BertTokenizer.from_pretrained(model_name)
  model = BertModel.from_pretrained(model_name,
                                    output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )
  model.eval()

  def get_vector(text, layer):
      tokenized_text = tokenizer.tokenize("[CLS] " + text + " [SEP]")
      tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)])
      with torch.no_grad():
          outputs = model(tensor_input)
          # `hidden_states` has shape [24 x 1 x 22 x 1024]
          hidden_states = outputs[2]
      sentence_embedding = hidden_states[layer][0][0]
      return sentence_embedding.numpy()

  # Read in files
  df_DT = pd.read_csv(dataset_name)
  if dataset_name == '/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_EventsRev_SentenceSet.csv':
    output_path = '/om2/user/jshe/lm-event-knowledge/probing/parallel_layers/EventsRev_'+ str(layernum) + fold_num + voice + sentence + '.csv'

  if dataset_name == '/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_DTFit_SentenceSet.csv':
    output_path = '/om2/user/jshe/lm-event-knowledge/probing/parallel_layers/DTFit_'+ str(layernum) + '.csv'
  # Exclude AAR rows only for EventsAdapt
  if dataset_name == '/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_EventsAdapt_SentenceSet.csv':
    df_DT = df_DT[df_DT.TrialType != "AAR"]
    df_DT = df_DT.reset_index()
    output_path = '/om2/user/jshe/lm-event-knowledge/probing/parallel_layers/EventsAdapt_'+ str(layernum) + '.csv'
  
  fold_num = 10
  np.random.seed(42)
  unique_pairs_num = df_DT['ItemNum'].nunique()
  unique_index = df_DT['ItemNum'].unique()

  shuffled_unique_index = np.random.permutation(unique_index)
  pool = np.array_split(shuffled_unique_index, fold_num)

  Accuracy = []
  for reg_trial in range(fold_num):
    test_index = pool[reg_trial]
    train_index = [i for i in unique_index if i not in test_index]
    test = pd.concat((df_DT[df_DT['ItemNum'] == j] for j in test_index))
    train = pd.concat((df_DT[df_DT['ItemNum'] == k] for k in train_index))
    test = test.reset_index()
    train = train.reset_index()

    print("1", train, test)
    if voice == 'active-active':
      train = train[train['Voice'] == 'active']
      test = test[test['Voice'] == 'active']
    elif voice == 'passive-passive':
      train = train[train['Voice'] == 'passive']
      test = test[test['Voice'] == 'passive']     
    elif voice == 'active-passive':
      train = train[train['Voice'] == 'active']
      test = test[test['Voice'] == 'passive']
    elif voice == 'passive-active':
      train = train[train['Voice'] == 'passive']
      test = test[test['Voice'] == 'active']
    else:
      pass
    print("2", train, test)
    if sentence == 'AI-AI':
      train = train[train['TrialType'] == 'AI']
      test = test[test['TrialType'] == 'AI']
    elif sentence == 'AI-AAN':
      train = train[train['TrialType'] == 'AI']
      test = test[test['TrialType'] == 'AAN']
    elif sentence == 'AAN-AAN':
      train = train[train['TrialType'] == 'AAN']
      test = test[test['TrialType'] == 'AAN']
    elif sentence == 'AAN-AI':
      train = train[train['TrialType'] == 'AAN']
      test = test[test['TrialType'] == 'AI']
    else:
      pass
    print("2", train, test)

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
output('bert-large-cased', voice_type, sentence_type, dataset_input)
