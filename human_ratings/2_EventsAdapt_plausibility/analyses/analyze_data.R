# Created on 2020-05-26 by Anna Ivanova
# Based on the code by Rachel Ryskin
# edited on 2021-02-28 by Zawad Chowdhury

rm(list=ls())
library(tidyverse)
library(stringr)
library(stringi)

# READ DATA
filenames=c('../results_raw/Batch_4332828_batch_results_raw.csv',
            '../results_raw/Batch_4368386_batch_results_raw.csv')

data <- lapply(filenames, read.csv)
data = do.call("rbind", data)
            
num.trials = 54  # maximum number of trials per participant

# only keep WorkerId and cols that Start with Answer or Input
data = data %>% select(starts_with('Input'),starts_with('Answer'),
                       starts_with('WorkerId'), starts_with('AssignmentId')) %>%
            select(-Input.list, -Answer.answer, -Answer.proficiency1,
                   -Answer.proficiency2)


# exclude bad workers
# data = data %>%
#   filter(!(WorkerId %in% c('A35LWWZHYTBJES', 'A15A618QS7DD79', 'A1IC1DQ0QQBOOZ',
#                            'A3V2XCDF45VN9X', 'A179LPB3NPSEF8', 'A13ASIJ31D76UN',
#                            'A2717S28QHY09K')))                   # bad responses
workers = read.csv("data_summ_by_worker_AIonly_summ.csv")
workers = workers %>% filter(is.na(Use))
data = data %>% filter(!(WorkerId %in% workers$WorkerId))

# gather (specify the list of columns you need)
data = data %>% gather(key='variable',value="value",
                       -WorkerId,-Answer.country, -Answer.English, -AssignmentId)

# separate
data = data %>% separate(variable, into=c('Type','TrialNum'),sep='__',convert=TRUE) 

# spread
data = data %>% spread(key = Type, value = value)

## Summarize ratings data 
data$Answer.Rating <- as.numeric(data$Answer.Rating)

## replace plausible-0 with plausible0
data$Input.code <- gsub('plausible-0', 'plausible0', data$Input.code)
data$Input.code <- gsub('plausible-1', 'plausible1', data$Input.code)


data = data %>% 
  separate(Input.code,into=c('TrialType','cond','Item','xx1','xx2'),sep='_') %>%
  separate(cond, into=c('Voice', 'Plausibility', 'xx3'), sep='-')

# info we don't need
data$xx3 = NULL
data$xx1 = NULL
data$xx2 = NULL

## SAVE A LONGFORM VERSION OF YOUR DATA
write_csv(data,"longform_data.csv")


# ANALYSES

## Look at data by question

summ_by_item = data %>%
  group_by(Voice, Plausibility, Item) %>%
  summarize(
    n = length(Answer.Rating),
    mean = mean(Answer.Rating, na.rm = TRUE)
  )

write.csv(summ_by_item, "data_by_item.csv")

## Look at data by participant (TODO: fix avg rating for plaus and implaus)
data = data %>%
  group_by(WorkerId) %>%
  mutate(
    na.pct = mean(is.na(Answer.Rating)),
    n = length(Answer.Rating)) %>%
  ungroup()
# 
# data_summ = data %>% 
#   group_by(WorkerId) %>%
#   summarize(
#     na.pct = mean(is.na(Answer.Rating)),
#     n = length(Answer.Rating))
# 
# data_byplausibility = data %>% 
#   group_by(WorkerId, Plausibility) %>% 
#   summarise_at(c("Answer.Rating"), funs(mean(., na.rm=TRUE)))

## save a summary of individual subjects' performance
# write_csv(data_summ,"data_summ_by_worker.csv")

z_score = function(xs) {
  (xs - mean(xs)) / sd(xs)
}

#filter for US, English, na, and duplicate, then get scores
###TODO: add filter for filler.left and filler.right??
data$Item = as.numeric(data$Item)
data.good = data %>%
  filter(Answer.English == "yes" &
         Answer.country == "USA" &
         Answer.profcheck1 == "Yes" &
         Answer.profcheck2 == "Yes" &
         TrialType != "filler" &
         n <= num.trials) %>%
  filter(!is.na(Answer.Rating)) %>%
  select(-Answer.English, -Answer.country, -Answer.profcheck1, -Answer.profcheck2,
         -na.pct, -n)

write_csv(data.good,"good_data.csv")


data.good.summary = data.good %>%
  group_by(Item, Plausibility, Voice) %>%
  summarize(
    m = mean(Answer.Rating),
    stdev= sd(Answer.Rating),
#    se = stdev/sqrt(n()),
#    upper= m+se*1.96,
#    lower=m-se*1.96
  )
  
## save a summary of individual subjects' performance
# data.good.summary = data.good.summary[,c(2,1,3,4, 5)]
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
