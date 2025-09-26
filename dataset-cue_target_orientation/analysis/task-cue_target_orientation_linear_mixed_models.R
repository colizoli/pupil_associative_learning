# Linear Mixed Models (Bayesian)
# O.Colizoli 2025 Pupil dilation offers a time-window on prediction error
# Dataset #1: Cue-target orientation 2AFC task
##################################################################################################################################
# Comparing 2 models in the early (_t1) and late (_t2) time windows (independently):
# model1: feedback-locked pupil ~ surprise + entropy + information gain + pre-feedback baseline pupil + reaction time + (1|subject)
# model2: feedback-locked pupil ~ surprise + entropy*information gain + pre-feedback baseline pupil + reaction time + (1|subject)
##################################################################################################################################

# Load initial packages and libraries
# install.packages("tidyverse", type="source")
# install.packages("ggplot2", dependencies=TRUE)
# install.packages("brms")  
# install.packages("tidybayes")
# install.packages("lme4")
# install.packages("report")

library(tidyverse)
library(ggplot2) #ggplot
library(brms)
library(loo)
library(tidybayes)
library(report)

# run or load models?
run_models = TRUE

# set parameters for model testing
iters = 10000
warmups = 5000
nchains = 4

# set current working directory and load data 
setwd("~dataset-cue_target_orientation/derivatives/data_frames")
df <- read_csv("task-cue_target_orientation_subjects.csv")
df <- df[df$outlier_rt != 1, ] # drop the "flagged" trials
View(df)
head(df)
str(df) # what type of variables do we have

# create a directory for saving the mixed models
if (!dir.exists("linear_mixed_models")) {
  dir.create("linear_mixed_models")
}

# check whether the DVs are normally distributed
# early time window
(hist_t1 <- ggplot(df, aes(x = pupil_target_locked_t1)) +
    geom_histogram(colour = "#8B5A00", fill = "#CD8500") +
    theme_bw() +
    ylab("Count\n") +
    xlab("\npupil_target_locked_t1") +  # latin name for red knot
    theme(axis.text = element_text(size = 12),
          axis.title = element_text(size = 14, face = "plain")))    

# late time window
(hist_t2 <- ggplot(df, aes(x = pupil_target_locked_t2)) +
    geom_histogram(colour = "#8B5A00", fill = "#CD8500") +
    theme_bw() +
    ylab("Count\n") +
    xlab("\npupil_target_locked_t2") +  # latin name for red knot
    theme(axis.text = element_text(size = 12),
          axis.title = element_text(size = 14, face = "plain")))    

unique(df$subject) # print subjects

######################################
### Early time window - all trials ### 
######################################
if (run_models) {
# fixed slope with random intercept
model1_t1 <- brms::brm(pupil_target_locked_t1 ~ model_i + model_H + model_D + pupil_baseline_target_locked + reaction_time + (1|subject),
                    data = df, family = gaussian(), chains = nchains,
                    iter = iters, warmup = warmups)
saveRDS(model1_t1, "linear_mixed_models/model1_t1") # to save and prevent running again later

# fixed slope with random intercept
model2_t1 <- brms::brm(pupil_target_locked_t1 ~ model_i + model_H*model_D + pupil_baseline_target_locked + reaction_time + (1|subject),
                    data = df, family = gaussian(), chains = nchains,
                    iter = iters, warmup = warmups)
saveRDS(model1_t1, "linear_mixed_models/model1_t1") # to save and prevent running again later

# model comparison
# The LOO assesses the predictive ability of posterior distributions
# The value with an elpd of 0 should appear, that’s the model that shows the best fit to our data.
t1_model_comparison = loo(model1_t1, model2_t1,compare = TRUE)
t1_model_comparison
saveRDS(t1_model_comparison, "linear_mixed_models/t1_model_comparison") # to save and prevent running again later

}

if (!run_models){
  model1_t1 = readRDS("linear_mixed_models/model1_t1") 
  model2_t1 = readRDS("linear_mixed_models/model2_t1") 
  t1_model_comparison = readRDS("linear_mixed_models/t1_model_comparison") 
}

# get info winning model
summary(model1_t1)
plot(model1_t1)
pp_check(model1_t1)  # posterior predictive checks
r <- report(model1_t1, verbose = FALSE)
r
s = summary(as.data.frame(r))

#####################################
### Late time window - all trials ### 
#####################################
if (run_models) {
# fixed slope with random intercept
model1_t2 <- brms::brm(pupil_target_locked_t2 ~ model_i + model_H + model_D + pupil_baseline_target_locked + reaction_time + (1|subject),
                       data = df, family = gaussian(), chains = nchains,
                       iter = iters, warmup = warmups)
saveRDS(model1_t2, "linear_mixed_models/model1_t2") # to save and prevent running again later

# fixed slope with random intercept
model2_t2 <- brms::brm(pupil_target_locked_t2 ~ model_i + model_H*model_D + pupil_baseline_target_locked + reaction_time + (1|subject),
                       data = df, family = gaussian(), chains = nchains,
                       iter = iters, warmup = warmups)
saveRDS(model2_t2, "linear_mixed_models/model2_t2") # to save and prevent running again later

# model comparison
# The LOO assesses the predictive ability of posterior distributions
# The value with an elpd of 0 should appear, that’s the model that shows the best fit to our data.
t2_model_comparison <- loo(model1_t2, model2_t2, compare = TRUE)
t2_model_comparison
saveRDS(t2_model_comparison, "linear_mixed_models/t2_model_comparison") # to save and prevent running again later
}

if (!run_models){
  model1_t2 = readRDS("linear_mixed_models/model1_t2") 
  model2_t2 = readRDS("linear_mixed_models/model2_t2") 
  t2_model_comparison = readRDS("linear_mixed_models/t2_model_comparison") 
}

# get info winning model
summary(model1_t2)
plot(model1_t2)
pp_check(model1_t2)  # posterior predictive checks
r2 <- report(model1_t2, verbose = FALSE)
r2
s2 = summary(as.data.frame(r2))

################
### ACCURACY ### 
################

# split data frame based on participants' accuracy
df_correct <- df[df$correct == 1, ] # only correct trials
df_error <- df[df$correct == 0, ] # only error trials

View(df_correct)
View(df_error)

# check whether the DVs are normally distributed
# early time window
(hist_t1 <- ggplot(df_correct, aes(x = pupil_target_locked_t1)) +
    geom_histogram(colour = "#8B5A00", fill = "#CD8500") +
    theme_bw() +
    ylab("Count\n") +
    xlab("\npupil_target_locked_t1 correct") +  # latin name for red knot
    theme(axis.text = element_text(size = 12),
          axis.title = element_text(size = 14, face = "plain")))    

# late time window
(hist_t2 <- ggplot(df_correct, aes(x = pupil_target_locked_t2)) +
    geom_histogram(colour = "#8B5A00", fill = "#CD8500") +
    theme_bw() +
    ylab("Count\n") +
    xlab("\npupil_target_locked_t2 correct") +  # latin name for red knot
    theme(axis.text = element_text(size = 12),
          axis.title = element_text(size = 14, face = "plain")))    

(hist_t1 <- ggplot(df_error, aes(x = pupil_target_locked_t1)) +
    geom_histogram(colour = "#8B5A00", fill = "#CD8500") +
    theme_bw() +
    ylab("Count\n") +
    xlab("\npupil_target_locked_t1 error") +  # latin name for red knot
    theme(axis.text = element_text(size = 12),
          axis.title = element_text(size = 14, face = "plain")))    

# late time window
(hist_t2 <- ggplot(df_error, aes(x = pupil_target_locked_t2)) +
    geom_histogram(colour = "#8B5A00", fill = "#CD8500") +
    theme_bw() +
    ylab("Count\n") +
    xlab("\npupil_target_locked_t2 error") +  # latin name for red knot
    theme(axis.text = element_text(size = 12),
          axis.title = element_text(size = 14, face = "plain")))    


##########################################
### Early time window - CORRECT trials ### 
##########################################
if (run_models) {
  # fixed slope with random intercept
  model1_t1_correct <- brms::brm(pupil_target_locked_t1 ~ model_i + model_H + model_D + pupil_baseline_target_locked + reaction_time + (1|subject),
                                 data = df_correct, family = gaussian(), chains = nchains,
                                 iter = iters, warmup = warmups)
  saveRDS(model1_t1_correct, "linear_mixed_models/model1_t1_correct") # to save and prevent running again later
  
  # fixed slope with random intercept
  model2_t1_correct <- brms::brm(pupil_target_locked_t1 ~ model_i + model_H*model_D + pupil_baseline_target_locked + reaction_time + (1|subject),
                                 data = df_correct, family = gaussian(), chains = nchains,
                                 iter = iters, warmup = warmups)
  saveRDS(model2_t1_correct, "linear_mixed_models/model2_t1_correct") # to save and prevent running again later
  
  # model comparison
  # The LOO assesses the predictive ability of posterior distributions
  # The value with an elpd of 0 should appear, that’s the model that shows the best fit to our data.
  t1_model_comparison_correct = loo(model1_t1_correct, model2_t1_correct, compare = TRUE)
  t1_model_comparison_correct
  saveRDS(t1_model_comparison_correct, "linear_mixed_models/t1_model_comparison_correct") # to save and prevent running again later
  
}

if (!run_models){
  model1_t1_correct = readRDS("linear_mixed_models/model1_t1_correct") 
  model2_t1_correct = readRDS("linear_mixed_models/model2_t1_correct") 
  t1_model_comparison_correct = readRDS("linear_mixed_models/t1_model_comparison_correct") 
}

# get info winning model
summary(model1_t1_correct)
plot(model1_t1_correct)
pp_check(model1_t1_correct)  # posterior predictive checks
r_correct <- report(model1_t1_correct, verbose = FALSE)
r_correct
s_correct = summary(as.data.frame(r_correct))

#########################################
### Late time window - CORRECT trials ### 
#########################################
if (run_models) {
  
  # fixed slope with random intercept
  model1_t2_correct <- brms::brm(pupil_target_locked_t2 ~ model_i + model_H + model_D + pupil_baseline_target_locked + reaction_time + (1|subject),
                                 data = df_correct, family = gaussian(), chains = nchains,
                                 iter = iters, warmup = warmups)
  saveRDS(model1_t2_correct, "linear_mixed_models/model1_t2_correct") # to save and prevent running again later
  
  # fixed slope with random intercept
  model2_t2_correct <- brms::brm(pupil_target_locked_t2 ~ model_i + model_H*model_D + pupil_baseline_target_locked + reaction_time + (1|subject),
                                 data = df_correct, family = gaussian(), chains = nchains,
                                 iter = iters, warmup = warmups)
  saveRDS(model2_t2_correct, "linear_mixed_models/model2_t2_correct") # to save and prevent running again later
  
  # model comparison
  # The LOO assesses the predictive ability of posterior distributions
  # The value with an elpd of 0 should appear, that’s the model that shows the best fit to our data.
  t2_model_comparison_correct <- loo(model1_t2_correct, model2_t2_correct, compare = TRUE)
  t2_model_comparison_correct
  saveRDS(t2_model_comparison_correct, "linear_mixed_models/t2_model_comparison_correct") # to save and prevent running again later
}

if (!run_models){
  model1_t2_correct = readRDS("linear_mixed_models/model1_t2_correct") 
  model2_t2_correct = readRDS("linear_mixed_models/model2_t2_correct") 
  t2_model_comparison_correct = readRDS("linear_mixed_models/t2_model_comparison_correct") 
}

# get info winning model
summary(model1_t2_correct)
plot(model2_t1_correct)
pp_check(model1_t2_correct)  # posterior predictive checks
r2_correct <- report(model1_t2_correct, verbose = FALSE)
r2_correct
s2_correct = summary(as.data.frame(r2_correct))


########################################
### Early time window - ERROR trials ### 
########################################
if (run_models) {
  
  # fixed slope with random intercept
  model1_t1_error <- brms::brm(pupil_target_locked_t1 ~ model_i + model_H + model_D + pupil_baseline_target_locked + reaction_time + (1|subject),
                               data = df_error, family = gaussian(), chains = nchains,
                               iter = iters, warmup = warmups)
  saveRDS(model1_t1_error, "linear_mixed_models/model1_t1_error") # to save and prevent running again later
  
  # fixed slope with random intercept
  model2_t1_error <- brms::brm(pupil_target_locked_t1 ~ model_i + pupil_baseline_target_locked + reaction_time + model_H*model_D + (1|subject),
                               data = df_error, family = gaussian(), chains = nchains,
                               iter = iters, warmup = warmups)
  saveRDS(model2_t1_error, "linear_mixed_models/model2_t1_error") # to save and prevent running again later
  
  # model comparison
  # The LOO assesses the predictive ability of posterior distributions
  # The value with an elpd of 0 should appear, that’s the model that shows the best fit to our data.
  t1_model_comparison_error = loo(model1_t1_error, model2_t1_error, compare = TRUE)
  t1_model_comparison_error
  saveRDS(t1_model_comparison_error, "linear_mixed_models/t1_model_comparison_error") # to save and prevent running again later
  
}

if (!run_models){
  model1_t1_error = readRDS("linear_mixed_models/model1_t1_error") 
  model2_t1_error = readRDS("linear_mixed_models/model2_t1_error") 
  t1_model_comparison_error = readRDS("linear_mixed_models/t1_model_comparison_error") 
}

# get info winning model
summary(model1_t1_error)
plot(model1_t1_error)
pp_check(model1_t1_error)  # posterior predictive checks
r_error <- report(model1_t1_error, verbose = FALSE)
r_error
s_error = summary(as.data.frame(r_error))

#########################################
### Late time window - ERROR trials ### 
#########################################
if (run_models) {
  
  # fixed slope with random intercept
  model1_t2_error <- brms::brm(pupil_target_locked_t2 ~ model_i + model_H + model_D + pupil_baseline_target_locked + reaction_time + (1|subject),
                               data = df_error, family = gaussian(), chains = nchains,
                               iter = iters, warmup = warmups)
  saveRDS(model1_t2_error, "linear_mixed_models/model1_t2_error") # to save and prevent running again later
  
  # fixed slope with random intercept
  model2_t2_error <- brms::brm(pupil_target_locked_t2 ~ model_i + pupil_baseline_target_locked + reaction_time + model_H*model_D + (1|subject),
                               data = df_error, family = gaussian(), chains = nchains,
                               iter = iters, warmup = warmups)
  saveRDS(model2_t2_error, "linear_mixed_models/model2_t2_error") # to save and prevent running again later
  
  # model comparison
  # The LOO assesses the predictive ability of posterior distributions
  # The value with an elpd of 0 should appear, that’s the model that shows the best fit to our data.
  t2_model_comparison_error <- loo(model1_t2_error, model2_t2_error, compare = TRUE)
  t2_model_comparison_error
  saveRDS(t2_model_comparison_error, "linear_mixed_models/t2_model_comparison_error") # to save and prevent running again later
}

if (!run_models){
  model1_t2_error = readRDS("linear_mixed_models/model1_t2_error") 
  model2_t2_error = readRDS("linear_mixed_models/model2_t2_error") 
  t2_model_comparison_error = readRDS("linear_mixed_models/t2_model_comparison_error") 
}

# get info winning model
summary(model1_t2_error)
plot(model1_t2_error)
pp_check(model1_t2_error)  # posterior predictive checks
r2_error <- report(model1_t2_error, verbose = FALSE)
r2_error
s2_error = summary(as.data.frame(r2_error))
