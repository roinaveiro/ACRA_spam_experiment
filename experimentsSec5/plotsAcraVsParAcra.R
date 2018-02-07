### This code reproduces the plots of the second experiment of section 5 of
### the ACRA paper

library(ggplot2)
library(tidyr)
library(dplyr)


data = read.csv("results/AcraVSAcraPar.csv")


# Plot times
data %>% 
    mutate(sUAcraSeqAcra = ACRATime/ParTime) ->
    data
library(plyr)
vlines <- ddply(data, .(m), summarize, mean = median(sUAcraSeqAcra))

p = ggplot(merge(data,vlines), aes(sUAcraSeqAcra)) + geom_histogram(bins = 50) +
  geom_vline(data = ddply(data, " m", summarize, mediana = median(sUAcraSeqAcra)), 
             aes(xintercept=mediana, color = "Median", linetype = "Median")) +
        geom_vline(data = ddply(data, "m", summarize, mean = mean(sUAcraSeqAcra)), 
                   aes(xintercept=mean,  color = "Mean", linetype = "Mean")) +
        scale_color_manual(name = "", values = c(Mean = "blue", Median = "red")) +
    scale_linetype_manual(name = "", values = c(Mean = "dotted", Median = "dashed")) +
 xlab("Speed Up") + ylab("") + facet_grid(m~ .) 

    
p = p + theme_bw()   
p
ggsave("figures/speedUp_par", device = "eps")

