## This codes reproduces the results of 2-GWI experiments in section 5
## of the ACRA paper

library(ggplot2)
library(tidyr)
library(dplyr)


data = read.csv("results/2GWI-Ling.csv") ## use "results/2GWI-UCI.csv"
                                          ## to read UCI results
                                          ## use "results/2GWI-Ling.csv"
                                          ## to read Ling results

#Get accuracies
acc = function(yt, yr){
    sum(yt == yr)/length(yr)
}

data %>% group_by(idx,m) %>%
    summarize(seqAcraAccuracy = acc(y,ycSeqMCACRA),
              nbcAccuracy = acc(y, yNbC),
              nbAccuracy = acc(y, yNb)) %>% group_by(m) %>%
    summarize(seqAcra = mean(seqAcraAccuracy),
              nbc = mean(nbcAccuracy),
              nb = mean(nbAccuracy))



#Get FPR
fpr = function(yt, yr){
    sum(yr == 1 & yt == 0)/sum(yt == 0)
}


data %>% group_by(idx,m) %>%
    summarize(seqAcraFPR = fpr(y,ycSeqMCACRA),
        nbcFPR = fpr(y, yNbC),
        nbFPR = fpr(y, yNb)) %>% group_by(m) %>%
    summarize(seqAcra = mean(seqAcraFPR),
        nbc = mean(nbcFPR),
        nb = mean(nbFPR))

#Get FNR
fnr = function(yt, yr){
    sum(yr == 0 & yt == 1)/sum(yt == 1)
}

data %>% group_by(idx,m) %>%
  summarize(seqAcraFNR = fnr(y,ycSeqMCACRA),
            nbcFNR = fnr(y, yNbC),
            nbFNR = fnr(y, yNb)) %>% group_by(m) %>%
  summarize(seqAcra = mean(seqAcraFNR),
            nbc = mean(nbcFNR),
            nb = mean(nbFNR))



