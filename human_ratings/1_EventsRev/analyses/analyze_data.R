# Created on 2020-05-26 by Anna Ivanova
# Based on the code by Rachel Ryskin

rm(list=ls())
library(tidyverse)
library(stringr)
library(stringi)

# READ DATA
filenames=c('../results_raw/Batch_4055141_batch_results.csv',
  '../results_raw/Batch_4055390_batch_results.csv',
  '../results_raw/Batch_4055655_batch_results.csv',
  '../results_raw/Batch_4057413_batch_results.csv')

data <- lapply(filenames, read.csv)
data = do.call("rbind", data)
            
num.trials = 42  # maximum number of trials per participant

# only keep WorkerId and cols that Start with Answer or Input
data = data %>% select(starts_with('Input'),starts_with('Answer'),starts_with('WorkerId')) 

# gather (specify the list of columns you need)
data = data %>% gather(key='variable',value="value",
                       -WorkerId,-Input.list,-Answer.country,
                       -Answer.English,-Answer.answer)

# separate
data = data %>% separate(variable, into=c('Type','TrialNum'),sep='__',convert=TRUE) 

# spread
data = data %>% spread(key = Type, value = value)

# exclude bad workers (note: currently done manually)
data = data %>%
  filter(!(WorkerId %in% c('AT8S19U5993HR', 'A2R1A479K07ME5')))                   # bad responses

## Summarize ratings data 
data$Answer.Rating <- as.numeric(data$Answer.Rating)

data = data %>% 
  separate(Input.code,into=c('TrialType','Plausibility','Item','xx1','xx2'),sep='_')
data$TrialType = NULL
data$xx1 = NULL
data$xx2 = NULL

## SAVE A LONGFORM VERSION OF YOUR DATA
write_csv(data,"longform_data.csv")


# ANALYSES

## Look at data by participant (TODO: add avg rating for plaus and implaus)
data = data %>% 
  group_by(WorkerId) %>%
  mutate(
    na.pct = mean(is.na(Answer.Rating)),
    n = length(Answer.Rating)) %>%
  ungroup()

data_summ = data %>% 
  group_by(WorkerId) %>%
  summarize(
    na.pct = mean(is.na(Answer.Rating)),
    n = length(Answer.Rating))

## save a summary of individual subjects' performance
write_csv(data_summ,"data_summ_by_worker.csv")

z_score = function(xs) {
  (xs - mean(xs)) / sd(xs)
}

#filter for US, English, na, and duplicate, then get scores
data$Item = as.numeric(data$Item)
data.good = data %>%
  filter(Answer.English == "yes" &
         Answer.country == "USA" &
         n <= num.trials) %>%
  filter(!is.na(Answer.Rating)) %>%
  filter(Item<=40)      # remove attention check items

data.good.summary = data.good %>%
  group_by(Plausibility, Item) %>%
  summarize(
    m = mean(Answer.Rating),
    stdev= sd(Answer.Rating),
#    se = stdev/sqrt(n()),
#    upper= m+se*1.96,
#    lower=m-se*1.96
  )
  
## save a summary of individual subjects' performance
data.good.summary = data.good.summary[,c(2,1,3,4)]
write_csv(data.good.summary[order(data.good.summary$Item),],
          "EventsRev_data_summ.csv")

# graphs of ratings by condition 
p1 = ggplot(data=data.good.summary)+
  stat_summary(mapping = aes(x = Plausibility, y = m), 
               geom = 'col', fun.y = 'mean', color = 'black')+
  geom_point(mapping = aes(x = Plausibility, y = m), 
             shape=21, size=2, alpha=0.5, stroke=1.5,
             position=position_jitter(width=0.15, height=0),
             show.legend = FALSE)+
  stat_summary(mapping = aes(x = Plausibility, y = m), 
               geom = 'errorbar', fun.data = 'mean_se', 
               color = 'black', size = 1.5, width=0.3)+
  ylab('Plausibility')+
  theme_classic()

ggsave('EventsRev plaus plot.png', p1, width=10, height=10, units="cm")

print(p1)
