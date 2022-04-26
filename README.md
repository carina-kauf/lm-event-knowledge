# lm-event-knowledge

### TODO
* **Score files**: Right now we're moving the score files between Event_Knowledge_Model_Comparison/results and model_scores via a script (because of the way in which we read the model scores in the analysis scripts) > streamline. [note: results last moved on 2022-04-25]
* **tinyLSTM scores**: are sums of token surprisal. Length-normalized scores can be obtained via dividing the score by the number of tokens (white-space separated in the penultimate column in the tinyLSTM score files)
