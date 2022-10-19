import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os
from sklearn import metrics
import sys
import pickle
from tqdm import tqdm
import argparse

import warnings
warnings.filterwarnings("ignore")

from regression_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    
    model_name = args.model_name
    
    for dataset_name in ["EventsRev", "DTFit", "EventsAdapt"]:
        
        #load embeddings
        hidden_states = load_embeddings(dataset_name, model_name)
        # get number of layers
        sent_key = list(hidden_states.keys())[0]
        layer_num = np.shape(hidden_states[sent_key])[0]
        
        #read dataset
        dataset = os.path.abspath(f'../../analyses_clean/clean_data/clean_{dataset_name}_df.csv')
        df = pd.read_csv(dataset, low_memory=False)
            
        df = df[df['Metric'] == 'human']
        df = df.reset_index(drop = True)
        
        #iterate over subset conditions & perform regressions
        fold_num = 10
        np.random.seed(42)
        
        for (voice_type, sentence_type) in voiceType_sentenceType_Zip:
            
            if dataset_name != "EventsAdapt" and (voice_type, sentence_type) != ('normal','normal'):
                continue #nothing to compute for the other datasets

            print(f"\n{model_name} | {dataset_name} | {voice_type} | {sentence_type}", flush=True)

            output_path = os.path.abspath(f'../results/model2human/{model_name}_{dataset_name}_{voice_type}_{sentence_type}.csv')
            
            if os.path.isfile(output_path):
                print("Already computed! Skipping!")
                continue

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
                    
                    y_pred = clf.predict(x_test)
                    print(classification_report(y_test, clf.predict(x_test)))
                    Accuracy.append(accuracy_score(y_test, y_pred)) #clf.score(y_test, y_pred))
                    
                print('layer', ind, statistics.mean(Accuracy), flush=True)
                print(Accuracy, flush=True)
                out.append(Accuracy)

            df_out = pd.DataFrame(out)
            df_out.to_csv(output_path, header = False)
            
if __name__ == "__main__":
    print(os.environ["SLURM_JOB_ID"], flush=True)
    main()