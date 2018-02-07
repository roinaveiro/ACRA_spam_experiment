### This code reproduces the plots of the first experiment of section 5 of
### the ACRA paper

library(ggplot2)
library(tidyr)
library(dplyr)


data = read.csv("results/AcraVSAcraMCBig.csv")


#Get accuracies
acc = function(yt, yr){
    sum(yt == yr)/length(yr)
}



data %>% group_by(idx,m) %>%
    summarize(acraAccuracy = acc(y,ycACRA),
              seqAcraAccuracy = acc(y,ycSeqMCACRA))%>% group_by(m) %>%
    summarize(acra = mean(acraAccuracy),
              seqAcra = mean(seqAcraAccuracy))




#Get FPR
fpr = function(yt, yr){
    sum(yr == 1 & yt == 0)/sum(yt == 0)
}


data %>% group_by(idx,m) %>%
  summarize(acraFPR = fpr(y,ycACRA),
            seqAcraFPR = fpr(y,ycSeqMCACRA)) %>% group_by(m) %>%
  summarize(acra = mean(acraFPR),
            seqAcra = mean(seqAcraFPR))

#Get FPR
fnr = function(yt, yr){
    sum(yr == 0 & yt == 1)/sum(yt == 1)
}



data %>% group_by(idx,m) %>%
  summarize(acraFNR = fnr(y,ycACRA),
            seqAcraFNR = fnr(y,ycSeqMCACRA)) %>% group_by(m) %>%
  summarize(acra = mean(acraFNR),
            seqAcra = mean(seqAcraFNR))


# Plot times
data %>% 
    mutate(sUAcraSeqAcra = ACRATime/SeqMCACRATime) ->
    data
library(plyr)
vlines <- ddply(data, .(m), summarize, mean = median(sUAcraSeqAcra))

p = ggplot(merge(data,vlines), aes(sUAcraSeqAcra)) + geom_histogram(bins = 50) +
  geom_vline(data = ddply(data, "m", summarize, mediana = median(sUAcraSeqAcra)), 
             aes(xintercept=mediana, color = "Median", linetype = "Median")) +
        geom_vline(data = ddply(data, "m", summarize, mean = mean(sUAcraSeqAcra)), 
                   aes(xintercept=mean,  color = "Mean", linetype = "Mean")) +
        scale_color_manual(name = "", values = c(Mean = "blue", Median = "red")) +
        scale_linetype_manual(name = "", values = c(Mean = "dotted", Median = "dashed")) +
  facet_grid(m~ .) +
  coord_cartesian(xlim = c(0,30)) +
        xlab("Speed Up") + ylab("")
        
p = p + theme_bw()        
p        
        
ggsave("figures/speedUp2", device = "eps")


