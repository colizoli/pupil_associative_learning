#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Pupil dilation offers a time-window in prediction error

Control Experiment - RUN ANALYSIS HERE
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
import preprocessing_functions_control_exp as pupil_preprocessing
import higher_level_functions_control_exp as higher
from IPython import embed as shell # for debugging

# -----------------------
# Levels (toggle True/False)
# ----------------------- 
pre_process     = False # pupil preprocessing is done on entire time series
trial_process   = False # cut out events for each trial and calculate trial-wise baselines, baseline correct evoked responses
higher_level    = True # all subjects' dataframe, pupil and behavior higher level analyses & figures

# -----------------------
# Paths
# ----------------------- 
# set path to home directory
home_dir        = os.path.dirname(os.getcwd()) # one level up from analysis folder
source_dir      = os.path.join(home_dir, 'raw')
data_dir        = os.path.join(home_dir, 'derivatives') # copy source directory and rename
control_experiments = ['task-control_exp_colors', 'task-control_exp_sounds'] # experiment_name

# copy 'raw' to derivatives if it doesn't exist:
if not os.path.isdir(data_dir):
    shutil.copytree(source_dir, data_dir) 
else:
    print('Derivatives directory exists. Continuing...')
    
# -----------------------
# Participants
# -----------------------
ppns     = pd.read_csv(os.path.join(home_dir, 'analysis', 'participants.csv'))
subjects = ['sub-{}'.format(s) for s in ppns['subject']]

# -----------------------
# Task: Colors
# ----------------------- 
colors_rgb = [
    [  3, 121, 112], # need hexidecimal, sorted by first column (r)
    [ 75, 124,  89],
    [ 76, 154,  68],
    [138, 154,  91],
    [  0, 168, 107],
    [157, 193, 131]
       ]
colors = [
    '#037970',
    '#4B7C59',
    '#4C9A44',
    '#8A9A5B',
    '#00A86B',
    '#9DC183'
]

# -----------------------
# Event-locked pupil parameters (shared)
# -----------------------
## note for response locked, need to use pre-stim baseline! not implemented!!
msgs                    = ['start recording', 'stop recording','phase 2']; # this will change for each task (keep phase 1 for locking to breaks)
phases                  = ['phase 2'] # of interest for analysis
time_locked             = ['stim_locked'] # events to consider (note: these have to match phases variable above)
baseline_window         = 0.5 # seconds before event of interest
pupil_step_lim          = [[-baseline_window, 3.0]] # size of pupil trial kernels in seconds with respect to first event, first element should max = 0!
sample_rate             = 1000 # Hz

# -----------------------
# Pupil preprocessing 
# -----------------------
if pre_process:
    # -----------------------
    # Pupil preprocessing specific parameters
    # phase 2 color onset
    # phase 3 response
    # -----------------------
    sample_rate = 1000 # Hz
    eye = 'R'
    msgs = ['start recording', 'stop recording', 'phase 2']; # this will change for each task
    tolkens = ['ESACC', 'EBLINK' ]      # check saccades and blinks based on EyeLink
    tw_blinks = 0.15                    # seconds before and after blink periods for interpolation
    mph       = 10      # detect peaks that are greater than minimum peak height
    mpd       = 1       # blinks separated by minimum number of samples
    threshold = 0       # detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors
    phases = ['phase 2'] # all phases of current task
    
    for experiment_name in control_experiments:
        for s,subj in enumerate(subjects):
            edf = '{}_{}_recording-eyetracking_physio'.format(subj, experiment_name)

            pupilPreprocess = pupil_preprocessing.pupilPreprocess(
                subject             = subj,
                edf                 = edf,
                project_directory   = data_dir,
                eye                 = ppns['eye'][s],
                msgs                = msgs, 
                tolkens             = tolkens,
                sample_rate         = sample_rate,
                tw_blinks           = tw_blinks,
                mph                 = mph,
                mpd                 = mpd,
                threshold           = threshold,
                )
            pupilPreprocess.convert_edfs()            # converts EDF to asc, msg and gaze files (run locally)
            pupilPreprocess.extract_pupil()             # read trials, and saves time locked pupil series as NPY array in processed folder
            pupilPreprocess.preprocess_pupil()          # blink interpolation, filtering, remove blinks/saccades, split blocks, percent signal change, plots output

# -----------------------
# Pupil trials & mean response per event type
# -----------------------      
if trial_process:  
    for experiment_name in control_experiments:
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
            trialLevel.event_related_subjects(pupil_dv = 'pupil_psc')  # psc: percent signal change, per event of interest, 1 output for all trials+subjects
            trialLevel.event_related_baseline_correction()           # per event of interest, baseline corrrects evoked responses

# -----------------------
# MEAN responses and group level statistics
# ----------------------- 
if higher_level:  
    pupil_time_of_interest = [[[0.75,1.25], [2.5,3.0]]]  # test same time window as in decision task!
    
    higherLevel = higher.higherLevel(
        subjects                = subjects, 
        experiment_name         = control_experiments[0], # colors
        project_directory       = data_dir, 
        sample_rate             = sample_rate,
        time_locked             = time_locked,
        pupil_step_lim          = pupil_step_lim,                
        baseline_window         = baseline_window,              
        pupil_time_of_interest  = pupil_time_of_interest, # time windows to average phasic pupil, per event, in higher.plot_evoked_pupil
        colors                  = colors   # determines how to group the conditions
        )
    higherLevel.higherlevel_get_phasics()             # compute phasic pupil in time window of interest
    higherLevel.create_subjects_dataframe()           # creates a single large dataframe all subjects
    higherLevel.average_conditions_colors()           # averages dvs in conditions of interest
    higherLevel.dataframe_evoked_pupil_higher_colors()  # averages evoked pupil responses by conditions of interest
    higherLevel.plot_evoked_pupil_higher_colors()       # averages evoked pupil responses by conditions of interest
    
        
    higherLevel = higher.higherLevel(
        subjects                = subjects, 
        experiment_name         = control_experiments[1], # sounds
        project_directory       = data_dir, 
        sample_rate             = sample_rate,
        time_locked             = time_locked,
        pupil_step_lim          = pupil_step_lim,                
        baseline_window         = baseline_window,              
        pupil_time_of_interest  = pupil_time_of_interest, # time windows to average phasic pupil, per event, in higher.plot_evoked_pupil
        colors                  = colors                  # determines how to group the conditions
        )
    higherLevel.higherlevel_get_phasics()             # compute phasic pupil in time window of interest
    higherLevel.create_subjects_dataframe()           # creates a single large dataframe all subjects
    higherLevel.average_conditions_sounds()           # averages dvs in conditions of interest
    higherLevel.dataframe_evoked_pupil_higher_sounds()  # averages evoked pupil responses by conditions of interest
    higherLevel.plot_evoked_pupil_higher_sounds()       # averages evoked pupil responses by conditions of interest
        