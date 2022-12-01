# Analyses

Key files: 

* **preprocess_scores**: takes model scores, human ratings, and sentence info for each dataset and merges them together. Relies on *dataloader_utils*. Produces files in `clean_data/`
* **main_results**: plots and stats for the main analyses. Generates on *stats_utils*
* **main_results_extended_EventsAdapt**: analyses specific to Dataset 1
* **probing_model2plausibility**: probing analyses reported in the main text
