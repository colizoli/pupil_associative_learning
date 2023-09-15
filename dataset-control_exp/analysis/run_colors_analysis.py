#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LISA VERSION
"""
Analysis control experiment - Colors, pupillometry
O.Colizoli 2019
python 3.6
"""

############################################################################
# PUPIL ANALYSES
############################################################################
# importing python packages
import os, sys, datetime, time
import numpy as np
import pandas as pd
import preprocessing_functions_control_exp as pupil_preprocessing
# import colors_higher_gpe
# conda install matplotlib # fixed the matplotlib crashing error in 3.6

from IPython import embed as shell # for debugging

# -----------------------
# Levels
# ----------------------- 
pre_process     = True
trial_process   = False
higher_level    = False

# -----------------------
# Paths
# ----------------------- 
# set path to home directory
home_dir        = os.path.dirname(os.getcwd()) # one level up from analysis folder
data_dir        = os.path.join(home_dir, 'derivatives') # copy source directory and rename
experiment_name = 'task-control_exp_colors'

# -----------------------
# Participants
# -----------------------
ppns     = pd.read_csv(os.path.join(home_dir,'derivatives','participants.csv'))
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
        # pupilPreprocess.convert_edfs()            # converts EDF to asc, msg and gaze files (run locally)
        pupilPreprocess.extract_pupil()             # read trials, and saves time locked pupil series as NPY array in processed folder
        # pupilPreprocess.preprocess_pupil()          # blink interpolation, filtering, remove blinks/saccades, split blocks, percent signal change, plots output

# -----------------------
# Pupil trials & mean response per event type
# -----------------------      
if trial_process:  
    trialLevel = pupil_control.trials(
        subjects = subjects, 
        experiment_name = log_name,
        project_directory = data_dir, 
        sample_rate=sample_rate,
        phases = phases, 
        time_locked = time_locked,
        time_locked_phases = time_locked_phases,
        pupil_step_lim = pupil_step_lim,                
        baseline_window = baseline_window,              
        pupil_time_of_interest = pupil_time_of_interest     
        )
    trialLevel.event_related_subjects()               # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    trialLevel.event_related_baseline_correction()    # per event of interest, baseline corrrects evoked responses
    trialLevel.plot_event_related()                   # per event of interest, plots events across all conditions, tests against 0
    trialLevel.plot_event_related_subjects()          # per subject, per event of interest, plots events across all conditions, tests against 0
    trialLevel.phasic_pupil()                         # per event of interest, extracts phasic pupil in time window

# -----------------------
# MEAN responses and group level statistics
# ----------------------- 
if higher_level:  
    pupil_time_of_interest = [[[0.75,1.25],[2.5,3.0]]]  # test same time window as in decision task!
    
    higherLevel = colors_higher_gpe.higherLevel(
        subjects=subjects, 
        experiment_name = log_name,
        project_directory=data_dir, 
        sample_rate=sample_rate,
        time_locked = time_locked,
        pupil_step_lim = pupil_step_lim,                
        baseline_window = baseline_window,              
        pupil_time_of_interest = pupil_time_of_interest,
        colors = colors  
        )
    # higherLevel.higherlevel_dataframe()  # adds a column for blocks to subjects' logfiles, drops missed trials, saves higher level data frame
    # higherLevel.dataframe_behavior_higher()      # group level data frames for all main effects + interaction
    # higherLevel.dataframe_phasic_pupil_higher()  # group level data frames for all main effects + interaction
    # higherLevel.dataframe_evoked_pupil_higher()  # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    higherLevel.run_anova_behav()                # run anovas on behavioral dvs
    higherLevel.run_anova_pupil()                # run anovas on pupil dvs (time window)
    # higherLevel.plot_behav()                     # plots behavior, group level, main effects + interaction
    # higherLevel.plot_phasic_pupil()              # plots phasic pupil per event of interest, group level, main effects + interaction
    # higherLevel.plot_evoked_pupil()              # plots evoked pupil per event of interest, group level, main effects + interaction
        