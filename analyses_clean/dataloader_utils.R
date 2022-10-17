################################
## ---- helper functions -----
################################

#helper from https://stackoverflow.com/questions/35775696/trying-to-use-dplyr-to-group-by-and-apply-scale
scale_this <- function(x) as.vector(scale(x))

min_max <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

get_normalization_fn <- function(normalization_type) {
  if (grepl("min-max", normalization_type)){
    normalization <- function(x) {return(min_max(x)) }
  } else if (grepl("zscore", normalization_type)){
    normalization <- function(x) {return(scale_this(x)) }
  } else if (grepl("none", normalization_type)) {
    normalization <- function(x) {return(x)}
  } else {
    stop(paste('Unknown normalization type: ', normalization_type))
  }
  return(normalization)
}

#uppercase first letter of string 
firstup <- function(x) {
  substr(x, 1, 1) <- toupper(substr(x, 1, 1))
  return(x)
}

clean_metric_name <- function(filename) {
  metric = substr(filename,1,nchar(filename)-4)
  
  #strip EventsAdapt dataset prefix
  metric = str_replace(metric, "new[_-][Ee]ventsAdapt[_\\.]", "")
  metric = str_replace(metric, "newsentences[_-]EventsAdapt[_\\.]", "")
  #strip EventsRev dataset prefix
  metric = str_replace(metric, "ev1[_\\.]", "")
  #strip DTFit dataset prefix
  metric = str_replace(metric, "dtfit[_\\.]", "")
  metric = str_replace(metric, "DTFit_vassallo[_\\.]", "")
  
  ##Assimilate model names between datasets & determine names for plotting
  # ppmi
  metric = str_replace(metric, "deps.scores_baseline1", "syntax.PPMI")
  metric = str_replace(metric, "scores_baseline1", "syntax.PPMI")
  # SDM model names
  metric = str_replace(metric, "v2.sdm-scores", "SDM")
  metric = str_replace(metric, "deps_SDM", "SDM")
  # thematic fit
  metric = str_replace(metric, "deps.update-model.TF-prod.n200", "thematicFit.prod")
  metric = str_replace(metric, "update-model.TF-prod.n200", "thematicFit.prod")
  # GPT
  metric = str_replace(metric, "gpt2-medium", "GPT2-medium")
  metric = str_replace(metric, "gpt2-xl", "GPT2-xl")
  metric = str_replace(metric, "gpt-neo", "GPT-neo")
  metric = str_replace(metric, "gpt-j", "GPT-J")
  # tinyLSTM
  metric = str_replace(metric, "surprisal_scores_tinylstm", "tinyLSTM.surprisal")
  metric = str_replace(metric, "vassallo_tinyLSTM", "tinyLSTM")
  metric = str_replace(metric, "tinylstm.surprisal_scores", "tinyLSTM.surprisal")
  # (Ro)BERT(a)
  metric = str_replace(metric, "bert-large-cased", "BERT-large")
  metric = str_replace(metric, "roberta-large", "RoBERTa-large")
  # All
  metric = str_replace(metric, "sentence-l2r-PLL.sentence_surp", "l2r")
  metric = str_replace(metric, "sentence-PLL.sentence_surp", "PLL")
  metric = str_replace(metric, "sentence_surp", "l2r")
  metric = str_replace(metric, ".average_byNrTokens", ".average")
  metric = str_replace(metric, ".average_byNrWords", ".averageByNrWords")
  #metric = str_replace(metric, "l2r.sentence_surp", "l2r")
  #metric = str_replace(metric, "sentence-PLL", "PLL")
  #metric = str_replace(metric, "sentence-l2r-PLL", "l2r")
  #metric = str_replace(metric, ".verb-PLL", ".pverb")
  #metric = str_replace(metric, ".last-word-PLL", ".plast")
  
  return(metric)
}

get_score_colnum <- function(metric) {
  if (grepl("BERT", metric) || grepl("GPT", metric) || grepl("LSTM", metric) || grepl("thematicFit", metric)) {
    score_colnum = 3
  } else if (grepl("PPMI", metric)) {
    score_colnum = 4
  } else if (grepl("SDM", metric)) {
    score_colnum = 6
    paste("SDM", score_colnum)
  } else {
    stop(paste("unknown metric: ", metric))
  }
  return(score_colnum)
}


################################
## ---- read in model data -----
################################

read_data <- function(directory, filename, normalization_type) {
  d = read.delim(paste(directory, filename, sep='/'), 
                 header=FALSE, sep='\t')
  
  #METRIC NAME
  metric = clean_metric_name(filename)
  
  #ROWS
  print(paste('Number of sentences in ', filename, ': ', nrow(d)))
  
  # set #of trials per item
  if (grepl("Adapt", filename, fixed = TRUE) == TRUE) { #if EventsAdapt
    target_trialnr = 4
  } else { 
    target_trialnr = 2
  } 
  
  #Add sentence number from 0 to len(dataset) -1
  tgt_len = nrow(d)-1
  sentnums = c(0:tgt_len)
  
  #COLUMNS
  sent_colnum = 2
  score_colnum = get_score_colnum(metric)
  
  # create cleaned-up dataframe
  d = d[,c(sent_colnum, score_colnum)] 
  colnames(d) = c("Sentence", "Score")
  d$Score = as.numeric(d$Score)
  d = d %>%
    mutate(SentenceNum = sentnums) %>%
    mutate(ItemNum = SentenceNum %/% target_trialnr) %>%
    mutate(Metric = metric) %>%
    filter(!Score=="None")  %>%
    mutate(NormScore = normalization(Score)) %>%
    #strip space before final period 
    mutate(Sentence = str_replace(Sentence, " [.]", ".")) %>%
    #Add final period where missing
    mutate(Sentence = trimws(Sentence, which="both")) %>%
    mutate(Sentence = ifelse(endsWith(Sentence, "."),Sentence,paste(Sentence, ".", sep=""))) %>%
    #uppercase first word in sentence to align with other model sentence sets
    mutate(Sentence = firstup(Sentence))
  
  # assign plausibility labels
  if (experiment=="DTFit") {
    d = d %>%
      mutate(Plausibility = ifelse(SentenceNum%%2==0, 'Implausible', 'Plausible')) 
  } else {
    d = d %>% 
      mutate(Plausibility = ifelse(SentenceNum%%2==0, 'Plausible', 'Implausible')) 
  }
    
  return(d)
}
