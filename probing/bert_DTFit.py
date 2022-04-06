import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns

def output(model_name, dataset_name):
  tokenizer = BertTokenizer.from_pretrained(model_name)
  model = BertModel.from_pretrained(model_name,
                                    output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )
  model.eval()


  def get_vector(text, layer):
      marked_text = "[CLS] " + text + " [SEP]"
      tokenized_text = tokenizer.tokenize(marked_text)
      indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
      segments_ids = [1] * len(tokenized_text)
      tokens_tensor = torch.tensor([indexed_tokens])
      segments_tensors = torch.tensor([segments_ids])

      with torch.no_grad():
          outputs = model(tokens_tensor, segments_tensors)
          # `hidden_states` has shape [24 x 1 x 22 x 1024]
          hidden_states = outputs[2]
      # `token_vecs` is a tensor with shape [22 x 1024], the first index in hidden_states determines which layer it's on
      token_vecs = hidden_states[layer][0]
      # get the first layer of token_vecs
      sentence_embedding = token_vecs[0]
      return sentence_embedding.numpy()

  # Read in files
  df_DT = pd.read_csv(dataset_name)

  # Exclude AAR rows only for EventsAdapt
  if dataset_name == "clean_EventsAdapt_SentenceSet.csv":
    df_DT = df_DT[df_DT.TrialType != "AAR"]
    df_DT = df_DT.reset_index()


  def split(dataset, train_ratio):
      unique_pairs_num = dataset['ItemNum'].nunique()
      unique_index = df_DT['ItemNum'].unique()
      random_list_index = np.random.choice(unique_index, round(unique_pairs_num*train_ratio), replace = False)   # list of random indices to form training set

      train = pd.concat((df_DT[df_DT['ItemNum'] == i] for i in random_list_index))

      test_pool = [i for i in unique_index if i not in random_list_index]   #list of ItemNum indices to form testing set
      test = pd.concat((df_DT[df_DT['ItemNum'] == i] for i in test_pool))

      train = train.reset_index()
      test = test.reset_index()
      return train, test


  # Populating layer entries
  out = []
  for layernum in range(24):
    Accuracy = []
    for reg_trial in range(10):
      # Splitting the dataset into training and testing
      train, test = split(df_DT, 0.85)
      x_train = []
      for i in range(len(train)):
          x_train.append(get_vector(train["Sentence"][i], layernum))
      x_train = np.array(x_train)

      x_test = []
      for j in range(len(test)):
          x_test.append(get_vector(test["Sentence"][j], layernum))
      x_test = np.array(x_test)

      y_train = np.array(train["Plausibility"])
      y_test = np.array(test["Plausibility"])

      # Fitting regression
      logreg = LogisticRegression(max_iter=500, solver='liblinear')
      logreg.fit(x_train, y_train)
      y_pred = logreg.predict(x_test)
      Accuracy.append(metrics.accuracy_score(y_test, y_pred))

    print("layer", layernum, statistics.mean(Accuracy))
    print(Accuracy)
    out.append(Accuracy)


  # Plotting
  df_out = pd.DataFrame(out)
  df_out = df_out.transpose()
  sns.set_theme(style="darkgrid")
  ax = sns.pointplot(data = df_out)
  ax.set(xlabel="Layers", ylabel = "Accuracy")
  ax.axhline(0.5, linestyle="dashed")
  ax.set_title(dataset_name)
  plt.show()
  print(df_out)
  df_out.to_csv('/om2/user/jshe/DTFit_layers.csv')
output('bert-large-cased', '/om2/user/jshe/lm-event-knowledge/analyses_clean/clean_data/clean_DTFit_SentenceSet.csv')