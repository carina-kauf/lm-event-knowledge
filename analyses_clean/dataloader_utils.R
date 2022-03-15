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
  metric = str_replace(mmtric, "new[_-][Ee]ventsAdapt[_\\.]", "")
  metric = str_replace(metric, "newsentences[_-]EventsAdapt[_\\.]", "")
  #strip EventsRev dataset prefix
  metric = str_replace(metric, "ev1[_\\.]", "")
  #strip DTFit dataset prefix
  metric = str_replace(metric, "dtfit[_\\.]", "")
  metric = str_replace(metric, "DTFit_vassallo[_\\.]", "")
  
  ##Assimilate model names between datasets & determine names for plotting
  # ppmi
  metric = str_replace(Metric, "deps.scores_baseline1", "syntax.PPMI")
  # SDM model names
  metric = str_replace(Metric, "v2.sdm_scores", "SDM")
  # thematic fit
  metric = str_replace(Metric, "deps.update-model.TF-prod.n200", "thematicFit.prod")
  # GPT2
  metric = str_replace(Metric, "gpt2-medium", "GPT2-medium")
  metric = str_replace(Metric, "gpt2-xl", "GPT2-xl")
  metric = str_replace(Metric, "sentence-prob", "l2r")
  # tinyLSTM
  metric = str_replace(Metric, "surprisal_scores_tinylstm", "tinyLSTM.surprisal")
  # Bidirectional
  metric = str_replace(Metric, "sentence-PLL", "PLL")
  metric = str_replace(Metric, "sentence-l2r-PLL", "l2r")
  metric = str_replace(Metric, ".verb-PLL", ".pverb")
  metric = str_replace(Metric, ".last-word-PLL", ".plast")
  # BERT
  metric = str_replace(Metric, "bert-large-cased", "BERT-large")
  metric = str_replace(Metric, "roberta-large", "RoBERTa-large")
  
  return(metric)
}


get_score_colnum <- function(experiment, metric) {
  if (grepl("BERT", metric) || grepl("GPT2", metric) || grepl("thematicFit", metric)) {
    score_colnum = 3
  } else if (grepl("LSTM", metric)) {
    score_colnum = 5
  } else if (grepl("PPMI", metric)) {
    score_colnum = 4
  } else if (grepl("SDM", metric)) {
    score_colnum = 6
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
  d = d[sent_colnum, score_colnum] 
  colnames(d) = c("Sentence", "Score")
  d = d %>%
    mutate(SentenceNum = sentnums) %>%
    mutate(ItemNum = SentenceNum %/% target_trialnr) %>%
    mutate(Plausibility = ifelse(SentenceNum%%2==0, 'Plausible', 'Implausible')) %>%
    mutate(Metric = metric) %>%
    mutate(NormScore = normalization(Score)) %>%
    #strip space before final period 
    mutate(Sentence = str_replace(Sentence, " [.]", ".")) %>%
    #Add final period where missing
    mutate(Sentence = ifelse(endsWith(Sentence, "."),Sentence,paste(Sentence, ".", sep="")))
    
  return(d)
}


################################
## ---- read in model data for DTFit -----
################################



# custom function to read a datatable
read_data_DTFit <- function(directory, filename) {
  d = read.delim(paste(directory, filename, sep='/'), 
                 header=FALSE, sep='\t')
  
  #ROWS
  print(paste('Number of sentences in ', filename, ': ', nrow(d)))
  
  #COLUMNS
  #check for target number of *columns* in file
  if (ncol(d)==3 | ncol(d)==4 | ncol(d)==5 | ncol(d)==7) {
    
    target_trialnr = 2
    
    #streamline input format
    if (ncol(d)==3) {
      d = d  %>%
        rename(SentenceNum=V1, Sentence=V2, Score=V3)
    }
    else if (ncol(d)==4) {
      d = d  %>%
        select(V1,V2,V4) %>% #select relevant columns (do not choose Typicality column to streamline how it's assigned)
        rename(SentenceNum=V1, Sentence=V2, Score=V4)
    }
    
    else if (ncol(d)==5) { #If surprisal_scores_tinylstm
      d = d  %>%
        select(V1,V2,V5) %>% 
        rename(SentenceNum=V1, Sentence=V2, Score=V5)
    }
    
    else if (ncol(d)==7) {
      d = d  %>%
        select(V1,V3,V7) %>% #select relevant columns (others include grammatical tags)
        rename(SentenceNum=V1, Sentence=V3, Score=V7)
    }
    
    d = d %>%
      mutate(Typicality = ifelse(SentenceNum%%2==0, 'Atypical', 'Typical')) %>%
      mutate(ItemNum = SentenceNum %/% target_trialnr) %>%
      
      # 2. NAME MODEL_METRIC
      #add metric column
      mutate(Metric = substr(filename,1,nchar(filename)-4)) %>%
      
      #strip DTFit dataset prefix
      mutate(Metric = str_replace(Metric, "dtfit[_\\.]", "")) %>%
      mutate(Metric = str_replace(Metric, "DTFit_vassallo[_\\.]", "")) %>%
      
      ##Assimilate model names between datasets & determine names for plotting
      # ppmi
      mutate(Metric = str_replace(Metric, "deps.scores_baseline1", "syntax.PPMI")) %>%
      # SDM model names
      mutate(Metric = str_replace(Metric, "v2.sdm_scores", "SDM")) %>%
      # thematic fit
      mutate(Metric = str_replace(Metric, "deps.update-model.TF-prod.n200", "thematicFit.prod")) %>%
      # GPT2
      mutate(Metric = str_replace(Metric, "gpt2-medium", "GPT2-medium")) %>%
      mutate(Metric = str_replace(Metric, "gpt2-xl", "GPT2-xl")) %>%
      mutate(Metric = str_replace(Metric, "sentence-prob", "l2r")) %>%
      # tinyLSTM
      mutate(Metric = str_replace(Metric, "surprisal_scores_tinylstm", "tinyLSTM.surprisal")) %>%
      # Bidirectional
      mutate(Metric = str_replace(Metric, "sentence-PLL", "PLL")) %>%
      mutate(Metric = str_replace(Metric, "sentence-l2r-PLL", "l2r")) %>%
      mutate(Metric = str_replace(Metric, ".verb-PLL", ".pverb")) %>%
      mutate(Metric = str_replace(Metric, ".last-word-PLL", ".plast")) %>%
      # BERT
      mutate(Metric = str_replace(Metric, "bert-large-cased", "BERT-large")) %>%
      mutate(Metric = str_replace(Metric, "roberta-large", "RoBERTa-large"))
    
    # 3. PREPROCESS SCORES
    #Normalize scores for all models
    d = d %>%
      mutate(NormScore = normalization(Score))
    
    # 4. PROCESS SENTENCES
    d = d  %>%
      #strip space before final period for alignment with TrialTypes etc below
      mutate(Sentence = str_replace(Sentence, " [.]", ".")) %>%
      #Add final period where missing
      mutate(Sentence = ifelse(endsWith(Sentence, "."),Sentence,paste(Sentence, ".", sep=""))) %>%
      #uppercase first word in sentence to align with other model sentence sets
      mutate(Sentence = firstup(Sentence))
    
    return(d)
  } 
  else {
    print(paste('unexpected number of columns in file: ', 'number of columns: ', filename, ncol(d)))
    return(NULL)
  }
}
