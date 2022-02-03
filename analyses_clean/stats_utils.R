################################
## ---- binom_pval --------
################################
calculate_binom_pval <- function(numCorrect, numTotal) {
  return(sapply(c(1:length(numCorrect)), 
                function(i){ binom.test(numCorrect[i], numTotal[i])$p.value}))
}

################################
## ---- significance level label --------
################################
plabel <- function(value) {
  plabel = ifelse(value<0.001, "***", 
                  ifelse(value<0.01, "**",
                         ifelse(value<0.05, "*", "n.s.")))
  return(plabel)
}

################################
## ---- chi-square --------
################################
# get the X-square and p-value of the X-square 2 sample test. Assume the total num is the same
calculate_chisq <- function(a, b, total) {
  result = prop.test(c(a, total), c(b, total))
  return(c(as.numeric(result$statistic), result$p.value))
}

calculate_chisq_vectorized_chi <- function(numCorrect, numTotal, num_correct_human) {
  return(sapply(c(1:length(numCorrect)), 
                function(i){ calculate_chisq(numCorrect[i], num_correct_human[i], numTotal[i])[1]}))
}

calculate_chisq_vectorized_p <- function(numCorrect, numTotal, num_correct_human) {
  return(sapply(c(1:length(numCorrect)), 
                function(i){ calculate_chisq(numCorrect[i], num_correct_human[i], numTotal[i])[2]}))
}

################################
## ---- correlations_df --------
################################

## ---- split for trialtype --------
get_correlation_df_tt <- function(analysis, humanMetric, trialTypes, input_dat, model_list) {
  df_correlation = data.frame()
  
  for (t in seq_along(trialTypes)){
    print(paste(trialTypes[t]))
    dat.model2human.tt = input_dat %>%
      filter(TrialType==trialTypes[t])
    
    if (analysis=='model2human') {
      subset1 <- dat.model2human.tt$ModelScore
      subset2 <- dat.model2human.tt$HumanScore
    } else if (analysis=='plausibility') {
      subset1 <- dat.model2human.tt$Plausible
      subset2 <- dat.model2human.tt$Implausible   
    } else if (analysis=='voice') {
      subset1 <- dat.model2human.tt$active
      subset2 <- dat.model2human.tt$passive
    } else if (analysis=='synonym') {
      subset1 <- dat.model2human.tt$Version1
      subset2 <- dat.model2human.tt$Version2
    } else {
      message("Analysis type not defined, don't know how to subset.")
    }
  
    val1.human = subset1[dat.model2human.tt$Metric==humanMetric]
    val2.human = subset2[dat.model2human.tt$Metric==humanMetric]  
    
    for(i in seq_along(model_list)){
      
      val1.model = subset1[dat.model2human.tt$Metric==model_list[i]]
      val2.model = subset2[dat.model2human.tt$Metric==model_list[i]]
  
      pval2zero = cor.test(val1.model, val2.model, method="pearson")$p.value
        corvals = c(
          cor(val1.human, val2.human, method="pearson"),
          cor(val1.model, val2.model, method="pearson"),
          cor(val1.human, val1.model, method="pearson"),
          cor(val1.human, val2.model, method="pearson"),
          cor(val2.human, val1.model, method="pearson"),
          cor(val2.human, val2.model, method="pearson"))
        test2humans = cocor.dep.groups.nonoverlap(corvals[1], corvals[2], corvals[3],
                                                  corvals[4], corvals[5], corvals[6],
                                                  n=length(val1.model),
                                                  test='raghunathan1996')
        pval2humans= test2humans@raghunathan1996$p.value
    
        # add vector to a dataframe
        df <- data.frame(model_list[i], trialTypes[t],corvals[2], pval2zero, pval2humans)
        df_correlation <- rbind(df_correlation,df)
      }
   }
  colnames(df_correlation) = c("Metric", "TrialType", "Correlation", "pVal2zero", "pVal2humans")
  
  # adjust for multiple comparisons
  df_correlation = df_correlation %>%
    mutate(pVal2zeroAdjusted = p.adjust(pVal2zero, method="fdr", n=length(pVal2zero)),
           pVal2humansAdjusted = p.adjust(pVal2humans, method="fdr", n=length(pVal2humans))) %>%
    mutate(pVal2zeroLabel = ifelse(pVal2zeroAdjusted<0.001, "***",
                                   ifelse(pVal2zeroAdjusted<0.01, "**",
                                          ifelse(pVal2zeroAdjusted<0.05, "*", ""))),
           pVal2humansLabel = ifelse(pVal2humansAdjusted<0.001, "***",
                                     ifelse(pVal2humansAdjusted<0.01, "**",
                                            ifelse(pVal2humansAdjusted<0.05, "*", ""))))
  return(df_correlation)
}


## ---- unsplit for trialtype --------
get_correlation_df <- function(analysis, humanMetric, input_dat, model_list) {
df_correlation = data.frame()
  if (analysis=='model2human') {
    subset1 <- input_dat$ModelScore
    subset2 <- input_dat$HumanScore
  } else if (analysis=='plausibility') {
    subset1 <- input_dat$Plausible
    subset2 <- input_dat$Implausible   
  } else if (analysis=='voice') {
    subset1 <- input_dat$active
    subset2 <- input_dat$passive
  } else if (analysis=='synonym') {
    subset1 <- input_dat$Version1
    subset2 <- input_dat$Version2
  } else {
    message("Analysis type not defined, don't know how to subset.")
  }
  
  val1.human = subset1[input_dat$Metric==humanMetric]
  val2.human = subset2[input_dat$Metric==humanMetric]  
  
  for(i in seq_along(model_list)){
    
    val1.model = subset1[input_dat$Metric==model_list[i]]
    val2.model = subset2[input_dat$Metric==model_list[i]]
    
    pval2zero = cor.test(val1.model, val2.model, method="pearson")$p.value
    corvals = c(
      cor(val1.human, val2.human, method="pearson"),
      cor(val1.model, val2.model, method="pearson"),
      cor(val1.human, val1.model, method="pearson"),
      cor(val1.human, val2.model, method="pearson"),
      cor(val2.human, val1.model, method="pearson"),
      cor(val2.human, val2.model, method="pearson"))
    test2humans = cocor.dep.groups.nonoverlap(corvals[1], corvals[2], corvals[3],
                                              corvals[4], corvals[5], corvals[6],
                                              n=length(val1.model),
                                              test='raghunathan1996')
    pval2humans= test2humans@raghunathan1996$p.value
    
    # add vector to a dataframe
    df <- data.frame(model_list[i],corvals[2], pval2zero, pval2humans)
    df_correlation <- rbind(df_correlation,df)
  }
  colnames(df_correlation) = c("Metric", "Correlation", "pVal2zero", "pVal2humans")

  # adjust for multiple comparisons
  df_correlation = df_correlation %>%
    mutate(pVal2zeroAdjusted = p.adjust(pVal2zero, method="fdr", n=length(pVal2zero)),
           pVal2humansAdjusted = p.adjust(pVal2humans, method="fdr", n=length(pVal2humans))) %>%
    mutate(pVal2zeroLabel = ifelse(pVal2zeroAdjusted<0.001, "***",
                                   ifelse(pVal2zeroAdjusted<0.01, "**",
                                          ifelse(pVal2zeroAdjusted<0.05, "*", ""))),
           pVal2humansLabel = ifelse(pVal2humansAdjusted<0.001, "***",
                                     ifelse(pVal2humansAdjusted<0.01, "**",
                                            ifelse(pVal2humansAdjusted<0.05, "*", ""))))
  return(df_correlation)
}