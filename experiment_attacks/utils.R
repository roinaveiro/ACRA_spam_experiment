library(ggplot2)
library(dplyr)
library(R2jags)

####################################################################################################
####################################################################################################
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

getprob <- function(v) {
  return( mean(v==1) )
}

getprob <- function(v) {
  return( mean(v==1) )
}

invlogit <- function(x){
  return(1 / (1 + exp(-x)))
}

accuracy <- function(y, prob){
  y_pred <- as.numeric(prob > 0.5)
  return( mean(y_pred == y) )
}

####################################################################################################
####################################################################################################

logistic_model <-  function(){
  
  # Likelihood
  for(i in 1:n){
    Y[i] ~ dbern(q[i])
    logit(q[i]) <- inprod(X[i,], beta)
  }
  
  #Priors
  for(j in 1:J){
    beta[j] ~ dnorm(0,0.1)
  }
}

posterior_probability <- function(beta, x){
  probs <- invlogit(beta%*%x)
  samples = c()
  for(p in probs){
    samples = append(samples, rbinom(n=1, p=p, size=1))
  }
  return( getprob(samples) )
}

train_lr <- function(jags.dat, nsamp){
  jags_out <-  jags(data = jags.dat, model= logistic_model, inits= NULL,
                    parameters.to.save=c("beta"), n.iter=nsamp)
  return(jags_out)
}

predict_lr <- function(x_test, jags_out, nsamp){
  prediction <- c()
  mat <- jags_out$BUGSoutput$sims.matrix
  lmat <- dim(mat)[1]
  beta = jags_out$BUGSoutput$sims.matrix[lmat-1000:lmat,-56]
  for(i in 1:dim(x_test)[1]){
    x = x_test[i,]
    prediction = append(prediction, posterior_probability(beta, x))
  }
  return(prediction)
}

####################################################################################################
####################################################################################################

add1 <- function(x){
  id = diag(length(x))
  result = diag(length(x))
  
  for(i in 1:length(x)){
    result[i,] = id[i,] | x
  }
  return(result)
}

add2 <- function(x){
  mat1 = add1(x)
  mat2 = add1(mat1[1,])
  for(i in 2:dim(mat1)[1]){
    mat2 = rbind(mat2, add1(mat1[i,]) )
  }
  mat2 = rbind(mat2, x)
  mat2 = unique(mat2)
  return( mat2 )
}

####################################################################################################
####################################################################################################

advut <- function(yc, y, a){
  d = length(a)
  if((y==1) & (yc==1)){
    Y = rep(-5.0, d)
  }
  else{
    if((y==1) & (yc==0)){
      Y = rep(5.0, d)
    }
    else{
      Y = rep(0.0, d)
    }
  }
  B = a*rep(0.5, d)
  rho = rep(0.5, d)
  return (exp( rho * (Y - B) ))
}

attack <- function(x, y, jags.out, names){
  if(y==0){
    return(x)
  }
  else{
    new_test <- add2(x)
    distances <- rowSums(t( apply(new_test, 1, function(z) z-x) ))
    new_test <- as.data.frame(new_test)
    colnames(new_test) <- names
    att_test_matrix <- model.matrix(~.,new_test)
    probs <- predict_lr(att_test_matrix, jags.out, 10000)
    utilities <- probs * advut(1,1,distances) + (1.0 - probs)* advut(0,1,distances)
    result <-as.numeric(new_test[which.max(utilities),])
    return(result)
  }
}
  

attack_test_set <- function(test, jags.out, names){
  x <- as.numeric(test[1,-55])
  test_attacked <- attack(x, test$spam[1], jags.out, names)
  
  for(i in 2:dim(test)[1]){
    print(i/dim(test)[1])
    x <- as.numeric(test[i,-55])
    x_att <- attack(x, test$spam[i], jags.out, names)
    test_attacked <- rbind(test_attacked, x_att)
  }
  colnames(test_attacked) <- names
  return(test_attacked)
  
}







