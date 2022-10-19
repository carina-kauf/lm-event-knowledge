import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sys
import pickle
import os
import os.path
from tqdm import tqdm
import argparse

from regression_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--condition', type=str)
    args = parser.parse_args()
    
    model_name = args.model_name
    dataset_name = args.dataset_name
    voice_type = args.condition.split("+")[0]
    sentence_type = args.condition.split("+")[1]
        
    #load embeddings
    hidden_states = load_embeddings(dataset_name, model_name)
    # get number of layers
    sent_key = list(hidden_states.keys())[0]
    layer_num = np.shape(hidden_states[sent_key])[0]

    #read dataset
    dataset = os.path.abspath(f'../../analyses_clean/clean_data/clean_{dataset_name}_df.csv')
    df = pd.read_csv(dataset, low_memory=False)

    #revert random implausibility label for AAR sentences
    if dataset_name == 'EventsAdapt':
        df.loc[df['TrialType'] == 'AAR', 'Plausibility'] = 'Plausible'

    #iterate over subset conditions & perform regressions
    fold_num = 10
    np.random.seed(42)

    print(f"\n{model_name} | {dataset_name} | {voice_type} | {sentence_type}", flush=True)

    output_path = os.path.abspath(f'../results/model2plausibility/{model_name}_{dataset_name}_{voice_type}_{sentence_type}.csv')

    out = []
    for ind, layer in tqdm(enumerate(range(layer_num))):
        Accuracy = []
        for reg_trial in range(fold_num):
            #changed
            train, test = split_dataset(reg_trial, df, voice_type, sentence_type, dataset_name, fold_num)

            x_train = []
            for i in range(len(train)):
                x_train.append(get_vector(hidden_states, train['Sentence'][i], layer, model_name))
            x_train = np.array(x_train)

            x_test = []
            for j in range(len(test)):
                x_test.append(get_vector(hidden_states, test['Sentence'][j], layer, model_name))
            x_test = np.array(x_test)

            y_train = np.array(train["Plausibility"])
            y_test = np.array(test["Plausibility"])

            # Fitting regression
            logreg = LogisticRegression(max_iter=500, solver='liblinear')
            logreg.fit(x_train, y_train)
            y_pred = logreg.predict(x_test)
            Accuracy.append(metrics.accuracy_score(y_test, y_pred))
        print('layer', ind, statistics.mean(Accuracy), flush=True)
        print(Accuracy, flush=True)
        out.append(Accuracy)

        df_out = pd.DataFrame(out)
        df_out.to_csv(output_path, header = False)
            
if __name__ == "__main__":
    print(os.environ["SLURM_JOB_ID"], flush=True)
    main()