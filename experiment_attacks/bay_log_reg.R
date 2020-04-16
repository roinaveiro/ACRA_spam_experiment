library(ggplot2)
library(dplyr)
library(R2jags)
source("utils.R")

data = read.csv('data/uciData.csv')
data = data[sample(nrow(data), nrow(data)), ]
data = data %>% mutate_if(is.numeric, as.factor)
tr_prop = 0.70
nsamp = 10000

for(i in 1:10){
  
  train_size <- floor(tr_prop * nrow(data))
  ## set the seed to make your partition reproducible
  ## set.seed(123)
  train_ind <- sample(seq_len(nrow(data)), size = train_size)
  train <- data[train_ind, ]
  test <- data[-train_ind, ]
  
  
  model_matrix = model.matrix( spam ~ ., train)
  jags.dat <- list(Y = as.numeric(train$spam)-1,
                   X = model_matrix,
                   n =  dim(train)[1], J = dim(train)[2] )
  
  test_matrix = model.matrix( spam ~ ., test)
  
  jags.out = train_lr(jags.dat, nsamp)
  pred = predict_lr(test_matrix, jags.out, nsamp)
  
  test[, 'probability'] = pred
  
  
  saveRDS(jags.out, file = paste0("data/jags", as.character(i), ".rds"))
  write.csv(train, paste0("data/train", as.character(i), ".csv"), row.names=FALSE)
  write.csv(test, paste0("data/test", as.character(i), ".csv"), row.names=FALSE)
}


#readRDS(file = "data/jags.rds")
#prr = as.integer(pred > 0.5)
#mean( prr == test$spam )
