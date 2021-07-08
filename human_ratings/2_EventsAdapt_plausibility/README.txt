This document outlines the organization of the 2_EventsAdapt_plausibility folder, as organized by Zawad Chowdhury on 8 June, 2021.

This project aimed to get human ratings of the plausibility of sentences in the EventsAdapt dataset. We first cleaned the dataset and generated sentences within the classes AI, AAN and AAR. Each sentence or item had 4 sentences generated, by switching the voice and switching the subject/object. Then we split the sentences into MTurk input csvs using turkolizer. We ran a huge MTurk survey, and obtained 598 responses after filtering. 

The code for the first step (cleaning and generating sentences) is kept in sentence_generation:
- adapt-realmaterials.xls was the provided EventsAdapt dataset
- changes_to_sentences.txt documents manual changes made from the original dataset.
- events_adapt_sentence_making.py was the code for sentence generation
- EventsAdapt_unique_sentence_data.csv was a helper csv.

The output of this code is newsentences_EventsAdapt.csv, which is in the main folder.

The code for turkolizer is kept in the main folder:
- turkolizer.py is an edited version of the turkolizer package, adding scripting support and the ability to append to existing turk.csv files
- generate_turkolizer_lists.py generates the turk.csv files for our MTurk survey.
-html_maker_rating.py is used to make the HTML file for MTurk.
- newsentences_EventsAdapt.csv includes all the sentences used.

In the process of making turk.csv files, we partition the sentences into different txt files which can be input into turkolizer. These are stored in turkolizer_files.

The output turk.csv files (for all three batches of our survey) and the MTurk html is stored in mturk_inputs.

The raw data from the MTurk survey is stored in results_raw.

The data is analyzed in analyses:
- analyze_subjects.R is for analyzing the data by each worker / HIT. This is used to generate a spreadsheet of the data by worker.
- data_summ_by_worker_ALL.csv is the above spreadsheet.

If performing the survey in multiple stages, analyze_subjects.R has some code for incorporating old data or looking only at new data (to approve/reject HITs). Examples of spreadsheets used for this are data_summ_by_worker_AIonly_oldchecks.csv and data_summ_by_worker_AIonly_new.csv respectively.

The data (in data_summ_by_worker_ALL.csv) is filtered based on the following criterion:
- Must say English is first language (Answer.English == Yes)
- Must say country is US. (Answer.country == USA)
- Must answer attention checks properly (filler.left == 7 and filler.right == 1)
- The average ratings for plausible sentences should be at least 1 more than the average ratings for implausible sentences (diff >= 1)
- Must answer most of the questions (na.pct < 0.1)
- Must be proficient in English (judged manually by entries in Answer.proficiency1, Answer.proficiency2). This is stored in profcheck.

We did this manually (in the spreadsheet data_summ_by_worker_ALL.csv, using Excel's Data>Filter), but this could be easily implemented into the R code. All entries which meet these criteria get a "yes" in the good_data column, and otherwise get a "no".

Once filtering is done, analyze_data.R can work with the filtered data. Here are the existing analysis outputs:

- longform_data.csv, outputing all ratings provided.
- EventsAdapt_data_summ.csv, a summary of the data by item.


