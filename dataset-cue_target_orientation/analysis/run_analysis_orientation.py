#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Pupil dilation offers a time-window in prediction error

Data set #1 Cue-target orientation 2AFC task - RUN ANALYSIS HERE
Python code O.Colizoli 2023 (olympia.colizoli@donders.ru.nl)
Python 3.6

Notes
-----
>> conda install matplotlib # fixed the matplotlib crashing error in 3.6
================================================
"""

import os, sys, datetime, time
import numpy as np
import pandas as pd
from IPython import embed as shell # for debugging
import preprocessing_functions_orientation as pupil_preprocessing
import higher_level_functions_orientation as higher

# -----------------------
# Levels (toggle True/False)
# ----------------------- 
pre_process     = False # pupil preprocessing is done on entire time series
trial_process   = False # cut out events for each trial and calculate trial-wise baselines, baseline correct evoked responses
higher_level    = True  # all subjects' dataframe, pupil and behavior higher level analyses & figures

# -----------------------
# Paths
# ----------------------- 
# set path to home directory
home_dir        = os.path.dirname(os.getcwd()) # one level up from analysis folder
source_dir      = os.path.join(home_dir, 'raw')
data_dir        = os.path.join(home_dir, 'derivatives')
experiment_name = 'task-cue_target_orientation'

# copy 'raw' to derivatives if it doesn't exist:
if not os.path.isdir(data_dir):
    shutil.copytree(source_dir, data_dir) 
else:
    print('Derivatives directory exists. Continuing...')
    
# -----------------------
# Participants
# -----------------------
ppns     = pd.read_csv(os.path.join(home_dir, 'analysis', 'participants_orientation.csv'))
subjects = ['sub-{}'.format(s) for s in ppns['subject']]
group    = ppns['normal_order']

# -----------------------
# Event-locked pupil parameters (shared)
# -----------------------
time_locked             = ['target_locked'] # events to consider
phases                  = ['target']
baseline_window         = 0.5 # seconds before event of interest
pupil_step_lim          = [[-baseline_window,3]] # size of pupil trial kernels in seconds with respect to first event, first element should max = 0!
sample_rate             = 500 # Hz

# -----------------------
# Pupil preprocessing, full time series
# -----------------------
if pre_process:  
    # preprocessing-specific parameters
    tw_blinks = 0.15    # seconds before and after blink periods for interpolation
    mph       = 10      # detect peaks that are greater than minimum peak height
    mpd       = 1       # blinks separated by minimum number of samples
    threshold = 0       # detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors
    # process 1 subject at a time
    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj, experiment_name)
        # initialize class
        pupilPreprocess = pupil_preprocessing.pupilPreprocess(
            subject             = subj,
            edf                 = edf,
            project_directory   = data_dir,
            sample_rate         = sample_rate,
            tw_blinks           = tw_blinks,
            mph                 = mph,
            mpd                 = mpd,
            threshold           = threshold
            )
        pupilPreprocess.housekeeping(experiment_name)   # rename files
        pupilPreprocess.read_trials()                   # change read_trials for different message strings
        pupilPreprocess.preprocess_pupil()              # blink interpolation, filtering, remove blinks/saccades, percent signal change, plots output

# -----------------------
# Pupil evoked responses, all trials
# -----------------------      
if trial_process:  
    # process 1 subject at a time
    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj, experiment_name)
        # initialize class
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

# -----------------------
# Behavior and responses, GROUP-level statistics
# ----------------------- 
if higher_level:  
    higherLevel = higher.higherLevel(
        subjects                = subjects, 
        group                   = group, # counterbalancing conditions
        experiment_name         = experiment_name,
        project_directory       = data_dir, 
        sample_rate             = sample_rate,
        time_locked             = time_locked,
        pupil_step_lim          = pupil_step_lim,                
        baseline_window         = baseline_window,              
        pupil_time_of_interest  = [[[0.75, 1.25], [2.5, 3.0]]] # time windows to average phasic pupil, per event, in higher.plot_evoked_pupil     
        )

    # higherLevel.higherlevel_log_conditions()     # computes mappings, accuracy, and missing trials
    # higherLevel.higherlevel_get_phasics()        # computes phasic pupil for each subject (adds to log files)
    # higherLevel.create_subjects_dataframe()      # adds baseline pupil, combines all subjects' behavioral files: task-predictions_subjects.csv, flags outliers, drops phase 2 trials
    ''' Note: the functions after this are using: task-cue_target_orientation_subjects.csv
    '''
    ''' DV averages within bin windows
    '''
    # higherLevel.average_conditions()           # group level data frames for all main effects + interaction
    # higherLevel.plot_phasic_pupil_pe()         # plots the interaction between the frequency and accuracy
    # higherLevel.plot_behavior()                # simple bar plots of accuracy and RT per mapping condition
    # higherLevel.individual_differences()       # individual differences correlation between behavior and pupil
    
    ''' Evoked pupil response
    '''
    # higherLevel.dataframe_evoked_pupil_higher()  # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    # higherLevel.plot_evoked_pupil()              # plots evoked pupil per event of interest, group level, main effects + interaction
    
    ''' Ideal learner model
    '''
    # higherLevel.information_theory_estimates()
    # higherLevel.pupil_information_correlation_matrix()
    # higherLevel.dataframe_evoked_correlation()
    # higherLevel.plot_pupil_information_regression_evoked()
    # higherLevel.plot_phasic_pupil_information_scatter()
    # higherLevel.average_information_conditions()
    # higherLevel.plot_information()
    higherLevel.plot_information_frequency()         # plots the interaction between the frequency and accuracy of the model parameters

    # not using
    # higherLevel.information_evoked_get_phasics()
    # higherLevel.plot_information_phasics()
    # higherLevel.plot_information_phasics_accuracy_split()
    # higherLevel.plot_information_pe()         # plots the interaction between the frequency and accuracy of the model parameters
    
    
    
    
    