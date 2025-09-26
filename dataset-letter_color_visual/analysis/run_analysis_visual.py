#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Pupil dilation offers a time-window in prediction error

Data set #2 Letter-color 2AFC task - RUN ANALYSIS HERE
Python code O.Colizoli 2024 (olympia.colizoli@donders.ru.nl)
Python 3.6

Notes
-----
>> conda install matplotlib # fixed the matplotlib crashing error in 3.6
# Need to have the EYELINK software installed on the terminal
================================================
"""

############################################################################
# PUPIL ANALYSES
############################################################################
# importing python packages
import os, sys, datetime, time, shutil
import numpy as np
import pandas as pd
import oddball_training_visual as training_higher
import preprocessing_functions_visual as pupil_preprocessing
import higher_level_functions_visual as higher
from IPython import embed as shell # for debugging

# -----------------------
# Levels (toggle True/False)
# ----------------------- 
training        = False  # process the logfiles and average the performance on the odd-ball training task
pre_process     = False  # pupil preprocessing is done on entire time series during the 2AFC decision task
trial_process   = False  # cut out events for each trial and calculate trial-wise baselines, baseline correct evoked responses (2AFC decision)
higher_level    = True   # all subjects' dataframe, pupil and behavior higher level analyses & figures (2AFC decision)
 
# -----------------------
# Paths
# ----------------------- 
# set path to home directory
home_dir        = os.path.dirname(os.getcwd()) # one level up from analysis folder
source_dir      = os.path.join(home_dir, 'raw')
data_dir        = os.path.join(home_dir, 'derivatives')
experiment_name = 'task-letter_color_visual_decision' # 2AFC Decision Task

# copy 'raw' to derivatives if it doesn't exist:
if not os.path.isdir(data_dir):
    shutil.copytree(source_dir, data_dir) 
else:
    print('Derivatives directory exists. Continuing...')

# -----------------------
# Participants
# -----------------------
ppns     = pd.read_csv(os.path.join(home_dir, 'analysis', 'participants_visual.csv'))
subjects = ['sub-{}'.format(s) for s in ppns['subject']]

# -----------------------
# Odd-ball Training Task, MEAN responses and group level statistics
# ----------------------- 
if training:  
    oddballTraining = training_higher.higherLevel(
        subjects          = subjects, 
        experiment_name   = 'task-letter_color_visual_training',
        project_directory = data_dir
        )
    oddballTraining.create_subjects_dataframe()       # drops missed trials, saves higher level data frame
    oddballTraining.average_conditions()              # group level data frames for all main effects + interaction
    oddballTraining.plot_behav()                      # plots behavior, group level, main effects + interaction
    oddballTraining.calculate_actual_frequencies()    # calculates the actual frequencies of pairs
    oddballTraining.information_theory_estimates()
    oddballTraining.plot_information_frequency()
    
# -----------------------
# Event-locked pupil parameters (shared)
# -----------------------
msgs                    = ['start recording', 'stop recording', 'phase 1', 'phase 7']; # this will change for each task (keep phase 1 for locking to breaks)
phases                  = ['phase 7'] # of interest for analysis
time_locked             = ['feed_locked'] # events to consider (note: these have to match phases variable above)
baseline_window         = 0.5 # seconds before event of interest
pupil_step_lim          = [[-baseline_window, 3.0]] # size of pupil trial kernels in seconds with respect to first event, first element should max = 0!
sample_rate             = 1000 # Hz
break_trials            = [60, 120, 180]  # which trial comes AFTER each break

# -----------------------
# 2AFC Decision Task, Pupil preprocessing, full time series
# -----------------------
if pre_process:
    # preprocessing-specific parameters
    tolkens = ['ESACC', 'EBLINK' ]      # check saccades and blinks based on EyeLink
    tw_blinks = 0.15                    # seconds before and after blink periods for interpolation
    mph       = 10      # detect peaks that are greater than minimum peak height
    mpd       = 1       # blinks separated by minimum number of samples
    threshold = 0       # detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors

    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj, experiment_name)

        pupilPreprocess = pupil_preprocessing.pupilPreprocess(
            subject             = subj,
            edf                 = edf,
            project_directory   = data_dir,
            eye                 = ppns['eye'][s],
            break_trials        = break_trials,
            msgs                = msgs, 
            tolkens             = tolkens,
            sample_rate         = sample_rate,
            tw_blinks           = tw_blinks,
            mph                 = mph,
            mpd                 = mpd,
            threshold           = threshold,
            )
        pupilPreprocess.convert_edfs()              # converts EDF to asc, msg and gaze files (run locally)
        pupilPreprocess.extract_pupil()             # read trials, and saves time locked pupil series as NPY array in processed folder
        pupilPreprocess.preprocess_pupil()          # blink interpolation, filtering, remove blinks/saccades, split blocks, percent signal change, plots output
    
        # need to reset object initialization to prevent interfence between preprocessing pipelines
        pupilPreprocess = pupil_preprocessing.pupilPreprocess(
            subject             = subj,
            edf                 = edf,
            project_directory   = data_dir,
            eye                 = ppns['eye'][s],
            break_trials        = break_trials,
            msgs                = msgs, 
            tolkens             = tolkens,
            sample_rate         = sample_rate,
            tw_blinks           = tw_blinks,
            mph                 = mph,
            mpd                 = mpd,
            threshold           = threshold,
            )
        pupilPreprocess.preprocess_pupil_raw_bp()       # filtering, percent signal change, percent signal change, plots output
    
        # need to reset object initialization to prevent interfence between preprocessing pipelines
        pupilPreprocess = pupil_preprocessing.pupilPreprocess(
            subject             = subj,
            edf                 = edf,
            project_directory   = data_dir,
            eye                 = ppns['eye'][s],
            break_trials        = break_trials,
            msgs                = msgs, 
            tolkens             = tolkens,
            sample_rate         = sample_rate,
            tw_blinks           = tw_blinks,
            mph                 = mph,
            mpd                 = mpd,
            threshold           = threshold,
            )
        pupilPreprocess.preprocess_pupil_interp_bp()    # blink interpolation, filtering, percent signal change, plots output

# -----------------------
# 2AFC Decision Task, Pupil trials & mean response per event type
# -----------------------      
if trial_process:  
    # process 1 subject at a time
    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj, experiment_name)
        trialLevel = pupil_preprocessing.trials(
            subject             = subj,
            edf                 = edf,
            project_directory   = data_dir,
            sample_rate         = sample_rate,
            phases              = phases,
            time_locked         = time_locked,
            pupil_step_lim      = pupil_step_lim, 
            baseline_window     = baseline_window
            )
        trialLevel.event_related_subjects(pupil_dv='pupil_psc')  # psc: percent signal change, per event of interest, 1 output for all trials+subjects
        trialLevel.event_related_baseline_correction()           # per event of interest, baseline corrrects evoked responses
        
        trialLevel.event_related_subjects_raw_bp(pupil_dv='pupil_psc')  # psc: percent signal change, per event of interest, 1 output for all trials+subjects
        trialLevel.event_related_baseline_correction_raw_bp()           # per event of interest, baseline corrrects evoked responses
        
        trialLevel.event_related_subjects_interp_bp(pupil_dv='pupil_psc')  # psc: percent signal change, per event of interest, 1 output for all trials+subjects
        trialLevel.event_related_baseline_correction_interp_bp()           # per event of interest, baseline corrrects evoked responses
        
        trialLevel.event_related_subjects_nuisance(pupil_dv='pupil_nuisance_psc')  # psc: percent signal change, per event of interest, 1 output for all trials+subjects
        trialLevel.event_related_baseline_correction_nuisance()           # per event of interest, baseline corrrects evoked responses
# -----------------------
# 2AFC Decision Task, MEAN responses and group level statistics 
# ----------------------- 
if higher_level:  
    higherLevel = higher.higherLevel(
        subjects                = subjects, 
        experiment_name         = experiment_name,
        project_directory       = data_dir, 
        sample_rate             = sample_rate,
        time_locked             = time_locked,
        pupil_step_lim          = pupil_step_lim,                
        baseline_window         = baseline_window,              
        pupil_time_of_interest  = [[[0.75,1.25], [2.5,3.0]]], # time windows to average phasic pupil, per event, in higher.plot_evoked_pupil
        freq_cond               = 'frequency'   # determines how to group the conditions based on actual frequencies
        )
    higherLevel.compute_blink_percentages()      # computes the amount of interpolated data per trial and excludes based on threshold
    higherLevel.higherlevel_get_phasics()       # computes phasic pupil for each subject (adds to log files)
    higherLevel.create_subjects_dataframe(blocks=break_trials+[240], exclude_interp=0)  # add baselines, concantenates all subjects, flags missed trials, saves higher level data frame
    ''' Supplementary analysis: Run the conservative pipeline
    higherLevel.create_subjects_dataframe(blocks=break_trials+[240], exclude_interp=1)  # adds baseline pupil, combines all subjects' behavioral files into one large data frame, flags outliers, drops phase 2 trials
    '''
    
    ''' Note: the functions after this are using: task-letter_color_visual_decision_subjects.csv
    '''
    higherLevel.average_conditions()           # group level data frames for all main effects + interaction
    higherLevel.plot_phasic_pupil_pe()         # plots the interaction between the frequency and accuracy
    higherLevel.plot_behavior()                # simple bar plots of accuracy and RT per mapping condition
    higherLevel.individual_differences()       # individual differences correlation between behavior and pupil
    
    ''' Evoked pupil response
    '''
    higherLevel.dataframe_evoked_pupil_higher()  # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    higherLevel.plot_evoked_pupil()              # plots evoked pupil per event of interest, group level, main effects + interaction
    
    ''' Ideal learner model
    '''
    higherLevel.information_theory_estimates(flat_prior=False) # compute the Shannon surprise, entropy, and KL divergence per trial for each subject, add to subjects' dataframe
    higherLevel.average_information_conditions()    # average model parameters across conditions of interest
    higherLevel.plot_information()                  # plot the model parameters as a function of task conditions
    
    ''' Ideal learner model fits
    '''
    higherLevel.pupil_information_correlation_matrix()      # plot the correlation between the model parameters
    higherLevel.dataframe_evoked_correlation()              # compute the correlation of pupil to model parameters for each timepoint of evoked response
    higherLevel.plot_pupil_information_regression_evoked()  # plot the correlation between pupil to model parameters for each timepoint of evoked response
    
    ''' Supplementary analysis: Ideal learner model with uniform prior distribution
    '''
    higherLevel.information_theory_estimates(flat_prior=True) # run model with uniform prior distribution
    higherLevel.average_information_conditions()
    higherLevel.plot_information()
    higherLevel.pupil_information_correlation_matrix()
    higherLevel.dataframe_evoked_correlation()
    higherLevel.plot_pupil_information_regression_evoked()

    ''' Supplementary analysis: Sanity checks on pupil pre-processing
    '''
    higherLevel.dataframe_evoked_pupil_higher_raw_bp()      # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    higherLevel.dataframe_evoked_pupil_higher_interp_bp()   # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    higherLevel.dataframe_evoked_pupil_higher_nuisance()    # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    higherLevel.plot_evoked_pupil_raw_bp()                  # plots evoked pupil per event of interest, group level, main effects + interaction
    higherLevel.plot_evoked_pupil_interp_bp()               # plots evoked pupil per event of interest, group level, main effects + interaction
    higherLevel.plot_evoked_pupil_nuisance()                # plots evoked pupil per event of interest, group level, main effects + interaction
    higherLevel.compute_phasics_interp_bp()                 # compute the average feedback response in the time windows of interest (INTERP BP timeseries)
    higherLevel.correlation_interp_clean()                  # check the correlation in the phasic pupil averages between the clean and "unclean" data
    higherLevel.group_r2_deconvolution()                     # average r2 from preprocessing deconvolution
    
    
    