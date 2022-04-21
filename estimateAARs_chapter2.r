
## Code to implement proposed approach from the paper
## "A Bayesian approach for estimating age-adjusted rates for low-prevalence diseases over space and time"
## by M Jay, J Oleson, M Charlton, and A Arab
## and Dissertation Chapter 2
## Note: to preserve confidentiality, the data are not included

## Load packages

library(INLA)
library(dplyr)
library(actuar)

## Load in data

nCounty # number of counties
nAge # number of age groups
nYear # number of years
liver # dataframe with county_id, year_id, recoded age group (age_group2), 
# population size, and number of deaths
adj # county adjacency matrix

# Sort dataframe by county, year, and age group
liver <- liver %>% arrange(county_id, year_id, age_group2)

# Create age group indicator variables
for(i in 1:(nAge-1)){ 
  liver[[paste0('age', i)]] <- ifelse(liver$age_group2 == i, 1, 0)
}

## Format data for INLA

data <- liver
n <- nrow(data)

y1 <- ifelse(data$deaths > 0, 1, 0)
y2 <- ifelse(data$deaths == 0, NA, data$deaths)

logpop <- log(data$population)

age1 <- data$age1
age2 <- data$age2
age3 <- data$age3
age4 <- data$age4
age5 <- data$age5
age6 <- data$age6
age7 <- data$age7
age8 <- data$age8
age9 <- data$age9
age10 <- data$age10
mu <- rep(1,n)

del1 <- data$year_id
gam1 <- data$county_id

del2 <- data$year_id
gam2 <- data$county_id
eps2 <- rep(1:(nCounty*nYear), each = nAge)

## Data for INLA

ldat1 <- list(Y = y1, age1 = age1, age2 = age2, age3 = age3, age4 = age4, age5 = age5,
              age6 = age6, age7 = age7, age8 = age8, age9 = age9, age10 = age10,
              mu = mu, logpop = logpop)

ldat2 <- list(Y = y2, age1 = age1, age2 = age2, age3 = age3, age4 = age4, age5 = age5,
              age6 = age6, age7 = age7, age8 = age8, age9 = age9, age10 = age10,
              mu = mu, logpop = logpop)

## Define half-cauchy prior on sigma 
## (Code to define half-cauchy prior from Bayesian inference with INLA by Virgilio Gomez-Rubio)
## https://becarioprecario.bitbucket.io/inla-gitbook/index.html

HC.prior  = "expression:
  sigma = exp(-theta/2);
  gamma = 10;
  log_dens = log(2) - log(pi) - log(gamma);
  log_dens = log_dens - log(1 + (sigma / gamma)^2);
  log_dens = log_dens - log(2) - theta / 2;
  return(log_dens);
"

## Bernoulli model fit

fit1 <- inla(Y ~ 0 + mu + age1 + age2 + age3 + age4 + age5 + age6 + age7 + age8 +
               age9 + age10 + 
               logpop + age1*logpop + age2*logpop + age3*logpop + age4*logpop + age5*logpop +
               age6*logpop + age7*logpop + age8*logpop + age9*logpop + age10*logpop +
               f(del1, model = "ar1", hyper = list(theta1 = list(prior = HC.prior), 
                                                   theta2 = list(prior = "betacorrelation", param = c(1,1)))) + 
               f(gam1, model = "besag", graph = adj, hyper = list(prec = list(prior = HC.prior))),
             data = ldat1, family = "binomial", 
             control.family = list(link = "cloglog"), 
             control.compute = list(dic = TRUE, waic = TRUE, config = TRUE))

## Truncated Poisson model fit

fit2 <- inla(Y ~ 0 + offset(logpop) + age1 + age2 + age3 + age4 + age5 + age6 + age7 + age8 + age9 + 
               age10 + mu +
               f(gam2, model = "besag", graph = adj, 
                 hyper = list(prec = list(prior = HC.prior))) + 
               f(del2, model = "ar1", hyper = list(theta1 = list(prior = HC.prior), 
                                                   theta2 = list(prior = "betacorrelation", param = c(1,1)))) + 
               f(eps2, model = "iid", hyper = list(prec = list(prior = HC.prior))),
             data = ldat2, family = c("zeroinflated.poisson0"), 
             control.family = list(list(hyper = list(prob = list(initial = -20, fixed = TRUE)))), 
             control.compute = list(dic = TRUE, waic = TRUE, config = TRUE))  

## DIC from the hurdle model

dic1 <- fit1$dic$dic
dic2 <- fit2$dic$dic
(dic <- dic1 + dic2)

## Compute age-adjusted rates and WAIC from posterior samples

## Function to compute pi from fit1 predictor

fun1 <- function(){
  1-exp(-exp(Predictor))
}

## Function to compute theta from fit2 predictor

fun2 <- function(){
  exp(Predictor)
}

## Draw 1000 samples of pi from the binomial fit

nSamp <- 1000
set.seed(211) 
samp1 <- inla.posterior.sample(nSamp, fit1, seed = 211)
pi_samp <- inla.posterior.sample.eval(fun1, samp1)

## Draw 1000 samples of theta from the truncated Poisson fit

set.seed(211) 
samp2 <- inla.posterior.sample(nSamp, fit2, seed = 211)
theta_samp <- inla.posterior.sample.eval(fun2, samp2)

## WAIC from the hurdle model

lik_mean <- rep(NA, nrow(theta_samp))
lik_var <- rep(NA, nrow(theta_samp))

for(i in 1:nrow(theta_samp)){
  
  if(y1[i] == 0){
    lik_mean[i] <- mean(exp(dbinom(0, size = 1, prob = pi_samp[i,], log = T)))
    lik_var[i] <- var(dbinom(0, size = 1, prob = pi_samp[i,], log = T))
  }
  else{
    lik_mean[i] <- mean(exp((dbinom(1, size = 1, prob = pi_samp[i,], log = T) + 
                               dztpois(y2[i], lambda = theta_samp[i,],  log = T))))
    lik_var[i] <- var(dbinom(1, size = 1, prob = pi_samp[i,], log = T) + 
                        dztpois(y2[i], lambda = theta_samp[i,],  log = T))
  }
}

lpd <- sum(log(lik_mean))
pwaic <- sum(lik_var)

waic <- -2*(lpd - pwaic)

waic

## Compute rates for all samples

rates <- (pi_samp /(1-exp(-theta_samp)))*theta_samp*100000/data$population

## Obtain mean rate for each county, year, and age group

rates <- rowMeans(rates)
rates <- matrix(rates, ncol = 11, byrow = T)

## Compute age-adjusted rates using a standard set of population weights

rates <- c(rates %*% pop_weights2)
rates <- cbind(unique(data[,1:2]), rates)
colnames(rates) <- c("county", "year", "rate")

aarates <- rates$rate
aarates_mat <- matrix(aarates, nrow = nCounty, ncol = nYear, byrow = T)