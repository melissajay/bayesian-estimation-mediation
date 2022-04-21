# Code for the decomposition analysis method proposed in Chapter 4
# This code illustrates the implementation of the proposed method on a simulated dataset

# Intervention: equalize mediator between ZIP codes with A = 0 and ZIP codes with A = 1
# Here, the mediator is continuous and note that the group variable, A, is simulated (it does not represent rural-urban status)

# Load in simulated dataset, neighborhood matrix, and population weights
load('simulatedDataset.RData')

# Load libraries
library(INLA)
library(dplyr)
library(data.table)

# Note that the columns and rows in neighborhoodMat are in the same order as the data
colnames(neighborhoodMat) <- rownames(neighborhoodMat) <- unique(simulatedData$zcta)

# Count number of ZIP codes and age groups
nZCTA <- length(unique(simulatedData$zcta))
nAge <- length(unique(simulatedData$age_group))

# Add a unique identifier for each ZIP code to the dataset
simulatedData$zcta_id <- rep(1:nZCTA, each = nAge)

# Add a unique identifier for each observation to the dataset
simulatedData$obs_id <- 1:(nZCTA*nAge)

# Create mediator dataset 
# Extract M, A, and C
# Ensure that there is one mediator value per ZIP code
med_dat <- filter(simulatedData, age_group == 1) %>% 
  select(M, A, C)

# Divide dataset by value of A
med_dat0 <- filter(med_dat, A == 0)
med_dat1 <- filter(med_dat, A == 1)

# Count number of ZIP codes in each group
n0 <- nrow(med_dat0)
n1 <- nrow(med_dat1)

# Create data frame with values to predict mediators
# Since the mediators will be predicted, we set their values to be NA
# Since our intervention involves matching the mediator distribution in areas with A = 1
# to be the same as areas with A = 0, we only need to obtain natural course
# predictions for all ZIP codes and mediator predictions under A = 0 for ZIP codes with A = 1
predict_mediator_values <- data.frame(M = NA,
                                      A = rep(0, n1),
                                      C = med_dat1$C)

# Augment the mediator dataset with the prediction dataset
med_dat_aug <- rbind(med_dat, predict_mediator_values)

# Define half-Cauchy(5) prior on standard deviation in INLA
HC.prior  = "expression:
  sigma = exp(-theta/2);
  gamma = 5;
  log_dens = log(2) - log(pi) - log(gamma);
  log_dens = log_dens - log(1 + (sigma / gamma)^2);
  log_dens = log_dens - log(2) - theta / 2;
  return(log_dens);
"

# Fit the mediator model
fit2 <- inla(M ~ A + C, 
             num.threads = 1, # set for reproducibility
             data = med_dat_aug, family = 'gaussian', 
             control.family = list(hyper = list(prec = list(prior = HC.prior))),
             control.compute = list(config = T)) # allows us to obtain samples of the linear predictors

# Set the number of posterior samples to draw
nSamp <- 1000

# Draw posterior samples of the natural course and counterfactual mediator values
set.seed(421)
mediatorSamps <- inla.posterior.sample(nSamp, fit2, seed = 421,
                                       selection = list(Predictor = 1:(nZCTA + n1)), 
                                       num.threads = 1)

# Create matrices to store the natural course and counterfactual mediator values
naturalMediators <- matrix(NA, nrow = nZCTA, ncol = nSamp)
counterfactualMediators <- matrix(NA, nrow = n1, ncol = nSamp)

# Store natural course linear predictors and hyperparameters
temp0 <- lapply(mediatorSamps, function(x) x$latent[1:nZCTA,])
hyper <- sapply(mediatorSamps, function(x) x$hyperpar[1])

# Obtain posterior predictions of natural course mediators
for(j in 1:nSamp){
  naturalMediators[,j] <- rnorm(nZCTA, mean = temp0[[j]], sd = 1 / sqrt(hyper[j]))
}

# Store counterfactual linear predictors
temp1 <- lapply(mediatorSamps, function(x) x$latent[(nZCTA + 1):(nZCTA + n1),])

# Obtain posterior predictions of counterfactual mediators
for(j in 1:nSamp){
  counterfactualMediators[,j] <- rnorm(n1, mean = temp1[[j]], sd = 1 / sqrt(hyper[j]))
}

rm(temp0, temp1)

# Fit the outcome model
fit1 <- inla(Y ~ 0 + factor(age_group) + A + M + C + 
               f(zcta_id, model = 'besag', graph = neighborhoodMat, hyper = list(prec = list(prior = HC.prior))) + 
               f(obs_id, model = 'iid', hyper = list(prec = list(prior = HC.prior))), 
             E = N, num.threads = 1,
             data = simulatedData, family = 'poisson',
             control.compute = list(config = T))

# Extract posterior samples of the model parameters and random effects
set.seed(422)
parameterSamps <- inla.posterior.sample(nSamp, fit1, seed = 422,
                                        selection = list('factor(age_group)1' = 1, 'factor(age_group)2' = 1, 'factor(age_group)3' = 1, 
                                                         'factor(age_group)4' = 1, 'factor(age_group)5' = 1, 'factor(age_group)6' = 1, 
                                                         A = 1, M = 1, C = 1,
                                                         obs_id = 1:(nZCTA*nAge), zcta_id = 1:nZCTA)) %>% 
  lapply(function(x) x$latent)

# Create empty matrices to store the natural course and counterfactual rates
nc_rate <- matrix(NA, nrow = nZCTA*nAge, ncol = nSamp)
cc_rate <- matrix(NA, nrow = n1*nAge, ncol = nSamp)

# Obtain posterior samples of natural course age group-specific rates
for(j in 1:nSamp){
  samps <- parameterSamps[[j]]
  rn <- rownames(samps)
  zcta_id <- rep(samps[which(grepl('zcta_id', rn)),1], each = nAge)
  obs_id <- samps[which(grepl('obs_id', rn)),1]
  alpha <- samps[which(grepl('age', rn)),1]
  beta1 <- samps[which(grepl('A:1', rn)),1]
  beta2 <- samps[which(grepl('M:1', rn)),1]
  beta3 <- samps[which(grepl('C', rn)),1]
  meds <- rep(naturalMediators[,j], each = nAge)
  lin_pred <-  rep(alpha, nZCTA) + beta1*simulatedData$A + beta2*meds + beta3*simulatedData$C + zcta_id + obs_id
  nc_rate[,j] <- 100000 * exp(lin_pred)
} 

# Obtain indices corresponding to each group and create reduced dataset
simulatedData_red <- filter(simulatedData, age_group == 1)
simulatedData_a1 <- filter(simulatedData, A == 1)
a1_idx1 <- which(simulatedData_red$A == 1)
a1_idx2 <- which(simulatedData$A == 1)
a0_idx1 <- which(simulatedData_red$A == 0)
a0_idx2 <- which(simulatedData$A == 0)

# Obtain predictions of the counterfactual age group-specific rates under the specified intervention
for(j in 1:nSamp){
  samps <- parameterSamps[[j]]
  rn <- rownames(samps)
  zcta_id <- rep(samps[which(grepl('zcta_id', rn))[a1_idx1],1], each = nAge)
  obs_id <- samps[which(grepl('obs_id', rn))[a1_idx2],1]
  alpha <- samps[which(grepl('age', rn)),1]
  beta1 <- samps[which(grepl('A:1', rn)),1]
  beta2 <- samps[which(grepl('M:1', rn)),1]
  beta3 <- samps[which(grepl('C', rn)),1]
  meds <- rep(counterfactualMediators[,j], each = nAge)
  lin_pred <- rep(alpha, n1) + beta1*1 + beta2*meds + 
    beta3*simulatedData_a1$C + zcta_id + obs_id
  cc_rate[,j] <- 100000 * exp(lin_pred)
} 

# Apply population weights to obtain AARs from natural course and counterfactual age group-specific rates
# Age groups are: <40, 40-49, 50-59, 60-69, 70-79, 80+
nc_rate <- apply(nc_rate, 2, function(x){x*rep(weights$weight, nZCTA)})
nc_rate <- data.frame(zcta = factor(simulatedData$zcta, levels = unique(simulatedData$zcta)), 
                      nc_rate) %>%
  group_by(zcta) %>%
  summarise_all(sum)

gc()

# Separate natural course rates for ZIP codes with A = 0 and A = 1
nc_aar_a0 <- colMeans(nc_rate[a0_idx1,-1])
nc_aar_a1 <- colMeans(nc_rate[a1_idx1,-1])

# Compute counterfactual AARs from counterfactual age group-specific rates
cc_aar_a1 <- data.frame(zcta = simulatedData_a1$zcta,
                         apply(cc_rate, 2, function(x){x*rep(weights$weight, n1)}))
cc_aar_a1 <- colMeans(aggregate(.~zcta, cc_aar_a1, sum)[,-1])

gc()

# Obtain posterior samples for the three AARDs of interest
nc_aar_diff <- nc_aar_a1 - nc_aar_a0
cc_aar_diff <- cc_aar_a1 - nc_aar_a0
red_aar_diff <- nc_aar_a1 - cc_aar_a1


# Create matrix of AARD_nat, AARD_count, and AARD_red estimates and lower/ upper bounds of 95% credible intervals
eff <- c(mean(nc_aar_diff), quantile(nc_aar_diff, probs = c(0.025, 0.975)),
         mean(cc_aar_diff), quantile(cc_aar_diff, probs = c(0.025, 0.975)),
         mean(red_aar_diff), quantile(red_aar_diff, probs = c(0.025, 0.975)))

names(eff) <- c('AARD_nat', 'AARD_nat - lower CI', 'AARD_nat - upper CI',
                'AARD_count', 'AARD_count - lower CI', 'AARD_count - upper CI',
                'AARD_red', 'AARD_red - lower CI', 'AARD_red - upper CI')         

round(eff, 3)
