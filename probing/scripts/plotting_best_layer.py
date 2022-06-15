import pandas as pd
import numpy as np

df = pd.read_csv("all_combined_kfold.csv")

unique_index = np.unique(df['Plot'], return_index=True)
indexes = unique_index[1]

unique_models = []
for index in sorted(indexes):
  a = df['Plot'][index]
  unique_models.append(a)

out = []
for plot in unique_models:
  store = []
  for j in range(len(np.unique(df.loc[(df['Plot'] == plot)]['Layer']))):
    subset = df.loc[(df['Plot'] == plot) & (df['Layer'] == j), ['Accuracy']]
    mean_subset = np.array(subset['Accuracy'].mean())
    store.append(mean_subset)
  out.append(store)

best_mean_performances = []
for i in range(len(out)):
  best_mean_performances.append(np.nanmax(out[i]))

def from_iterable(iterables):
    for it in iterables:
        for element in it:
            yield element

add_column = []
i = 0
for plot in np.unique(df['Plot']):#33
  store = []
  for j in range(len(np.unique(df.loc[(df['Plot'] == plot)]['Layer']))):#24/48
    for k in range(10):
      store.append(best_mean_performances[i])
  add_column.append(store)
  i = i+1

new_column = list(from_iterable(add_column))

df['BestLayer'] = new_column
#df.to_csv('check.csv')

## Plotting all models general

df_normal = df.loc[(df['TrialType'] == 'normal') & (df['VoiceType'] == 'normal')]

import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=df, kind="bar",
    x="Model", y="BestLayer", hue="Dataset",
     ci="sd", palette="dark", alpha=.6, height=6)
g.despine(left=True)
g.set_axis_labels("", "Accuracy")
g.legend.set_title("Datasets")
g.set(title='All models - General')

## Plotting EventsAdapt by VoiceType

df_events = df.loc[(df['Dataset'] == 'EventsAdapt') & (df['TrialType'] == 'normal')]

import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=df_events, kind="bar",
    x="VoiceType", y="BestLayer", hue="Model",
     ci="sd", palette="dark", alpha=.6, height=6, aspect = 1.2)
g.despine(left=True)
g.set_axis_labels("", "Accuracy")
g.legend.set_title("Datasets")
g.set(title='EventsAdapt - VoiceType')

## Plotting EventsAdapt by SentenceType

df_events = df.loc[(df['Dataset'] == 'EventsAdapt') & (df['VoiceType'] == 'active-active')]

import pandas as pd
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=df_events, kind="bar",
    x="TrialType", y="BestLayer", hue="Model",
     ci="sd", palette="dark", alpha=.6, height=6)
g.despine(left=True)
g.set_axis_labels("", "Accuracy")
g.legend.set_title("Datasets")
g.set(title='EventsAdapt - SentenceType')
