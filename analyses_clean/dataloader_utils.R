################################
## ---- normalization -----
################################

#helper function #https://stackoverflow.com/questions/35775696/trying-to-use-dplyr-to-group-by-and-apply-scale
scale_this <- function(x) as.vector(scale(x))

min_max <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

################################
## ---- read in model data -----
################################
read_data <- function(directory, filename) {
  d = read.delim(paste(directory, filename, sep='/'), 
                 header=FALSE, sep='\t')
  
  #ROWS
  print(paste('Number of sentences in ', filename, ': ', nrow(d)))
  
  #COLUMNS
  #check for target number of *columns* in file
  if (ncol(d)%in%c(3,4,6)) {
    
    # 0. PREPARATION (set #of trials per item)
    if (grepl("Adapt", filename, fixed = TRUE) == TRUE) { #if EventsAdapt
      target_trialnr = 4
    } else if (grepl("ev1", filename, fixed = TRUE) == TRUE){ #if EventsRev
      target_trialnr = 2
    } else {
      print(paste("unknown experiment for file: ", filename))
      target_trialnr = 0
    }
    
    #streamline input format
    if (ncol(d)==3){
      d = d  %>%
        rename(SentenceNum=V1, Sentence=V2, Score=V3) %>%
        mutate(ItemNum = SentenceNum %/% target_trialnr)
    }
    else if (ncol(d)==4){
      d = d  %>%
        select(V1,V2,V4) %>% #select relevant columns (do not choose Plausibility column to streamline how it's assigned
        #(e.g. all lowercase etc))
        rename(SentenceNum=V1, Sentence=V2, Score=V4) %>%
        mutate(ItemNum = SentenceNum %/% target_trialnr)
    }
    
    else if ((ncol(d)==6) & (grepl("smooth", filename) == FALSE)) { #If surprisal_scores_tinylstm
      #Add sentence number from 0 to len(dataset) -1
      tgt_len = nrow(d)-1
      sentnums = c(0:tgt_len)
      d = d  %>%
        select(V1,V5) %>% #select relevant columns (others include UNKification for EventsAdapt dataset)
        rename(Sentence=V1, Score=V5) %>%
        mutate(SentenceNum = sentnums) %>% #This adds a column SentenceNum from 0 to len(dataframe)-1
        mutate(ItemNum = SentenceNum %/% target_trialnr)
    }
    
    else if ((ncol(d)==6) & (grepl("smooth", filename) == TRUE)) {
      d = d  %>%
        select(V1,V3,V6) %>% #select relevant columns (others include grammatical tags)
        rename(SentenceNum=V1, Sentence=V3, Score=V6) %>%
        mutate(ItemNum = SentenceNum %/% target_trialnr)
    }
    
    d = d %>%
      mutate(Plausibility = ifelse(SentenceNum%%2==0, 'Plausible', 'Implausible')) %>%
      
      # 2. NAME MODEL_METRIC
      #add metric column
      mutate(Metric = substr(filename,1,nchar(filename)-4)) %>%
      
      #strip EventsAdapt dataset prefix
      mutate(Metric = str_replace(Metric, "new[_-][Ee]ventsAdapt[_\\.]", "")) %>%
      mutate(Metric = str_replace(Metric, "newsentences[_-]EventsAdapt[_\\.]", "")) %>%
      #strip EventsRev dataset prefix
      mutate(Metric = str_replace(Metric, "ev1[_\\.]", "")) %>%
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
      #strip space before final period 
      mutate(Sentence = str_replace(Sentence, " [.]", ".")) %>%
      #Add final period where missing
      mutate(Sentence = ifelse(endsWith(Sentence, "."),Sentence,paste(Sentence, ".", sep="")))
    
    return(d)
  } 
  else {
    print(paste('unexpected number of columns in file: ', 'number of columns: ', filename, ncol(d)))
    return(NULL)
  }
}


################################
## ---- read in model data for DTFit -----
################################
#uppercase first letter of string (needed for fast_vector_sum)
firstup <- function(x) {
  substr(x, 1, 1) <- toupper(substr(x, 1, 1))
  x
}


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
