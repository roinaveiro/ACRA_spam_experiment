library(ggplot2)
library(dplyr)
library(R2jags)
source("utils.R")

for(i in 1:10){
  train_path <- paste0('data/train', as.character(i), '.csv')
  test_path <-  paste0('data/test', as.character(i), '.csv')
  jags_path <-  paste0('data/jags', as.character(i), '.rds')
  out_path <-  paste0('data/test_att', as.character(i), '.csv')
  ###
  train = read.csv(train_path)
  test = read.csv(test_path)
  jags.out = readRDS(file = jags_path)
  test$probability <- NULL
  names = colnames(test)[-55]
  att <- attack_test_set(test, jags.out, names)
  write.csv(att, out_path, row.names=FALSE)
  print(i)
}

acc_clean <- c()
acc_att <- c()
for(i in 1:10){
  train_path <- paste0('data/train', as.character(i), '.csv')
  test_path <-  paste0('data/test', as.character(i), '.csv')
  jags_path <-  paste0('data/jags', as.character(i), '.rds')
  out_path <-  paste0('data/test_att', as.character(i), '.csv')
  
  test <- read.csv(test_path)
  test_att <- read.csv(out_path)
  jags_out <- readRDS(file = jags_path)
  test$probability <- NULL
  
  test_mat <- model.matrix(spam~.,test)
  prob_clean <- predict_lr(test_mat, jags_out, nsamp=10000)
  acc_clean <- c(acc_clean, accuracy(test$spam, prob_clean))
  
  
  test_att_mat <- model.matrix(~.,test_att)
  prob_att <- predict_lr(test_att_mat, jags_out, nsamp=10000)
  acc_att <- c(acc_att, accuracy(test$spam, prob_att))
  
}

i = 1
train_path <- paste0('data/train', as.character(i), '.csv')
test_path <-  paste0('data/test', as.character(i), '.csv')
jags_path <-  paste0('data/jags', as.character(i), '.rds')
out_path <-  paste0('data/test_att', as.character(i), '.csv')

test <- read.csv(test_path)
test_att <- read.csv(out_path)
jags_out <- readRDS(file = jags_path)
test$probability <- NULL



test_mat <- model.matrix(spam~.,test)
prob_clean <- predict_lr(test_mat, jags_out, nsamp=10000)
acc_clean <- c(acc_clean, accuracy(test$spam, prob_clean))


test_att_mat <- model.matrix(~.,test_att)
prob_att <- predict_lr(test_att_mat, jags_out, nsamp=10000)
acc_att <- c(acc_att, accuracy(test$spam, prob_att))

