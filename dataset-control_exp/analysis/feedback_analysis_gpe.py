#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LISA VERSION
"""
Analysis control experiment - Tones (Feedback), pupillometry
O.Colizoli 2019
python 3.6
"""

############################################################################
# PUPIL ANALYSES
############################################################################
# importing python packages
import os, sys, datetime, time
import numpy as np
from IPython import embed as shell # for debugging
import pupil_control
import feedback_higher_gpe
# conda install matplotlib # fixed the matplotlib crashing error in 3.6

############################################################################
# LEVELS ANALYSES
pre_process = False
trial_process = True
higher_level = True
############################################################################

# set path to home directory
home_dir = '/project/colizoli/GPE/Control_Exp1/'  # LISA
data_dir = os.path.join(home_dir, 'derivatives')
source_dir = os.path.join(home_dir, 'source')
log_name = 'task-feedback'

ppn = [206,220] # subject number range
# 201 have to combine two files somehow, EDF problem, not behavior
exclude = ['sub-202','sub-203','sub-204','sub-205'] # subjects #202-205 missing EDF files, technical error
subjects = ['sub-{}'.format(s) for s in np.arange(ppn[0],ppn[1]+1)]
subjects = [e for e in subjects if e not in exclude] 

########################################
# TASK COLORS
# -----------------------
# Pupil preprocessing, full time series 
# -----------------------
sample_rate = 1000 # Hz
eye = 'R'
msgs = ['start recording', 'stop recording', 'phase 2']; # this will change for each task
tolkens = ['ESACC', 'EBLINK' ]      # check saccades and blinks based on EyeLink
tw_blinks = 0.15                    # seconds before and after blink periods for interpolation
phases = ['phase 2']                # all phases of current task
time_locked = ['stim_locked']       # events to consider 
time_locked_phases = ['phase 2']    # phase onset of events
baseline_window = 0.5               # seconds before event of interest
pupil_step_lim = [-baseline_window,3]   # size of pupil trial kernels in seconds with respect to first event, first element should max = 0!
pupil_time_of_interest = [.541,1.541]   # test same time window as in decision task!
# pupil_time_of_interest = [.35,2.2]    # whole significant mean response

if pre_process:
    for s,subj in enumerate(subjects):
        # if subj == 'sub-106':
        #     eye = 'L'
        # else:
        #     eye = 'R'
        edfs = [os.path.join(source_dir,subj,'{}_{}_eye.EDF'.format(subj,log_name))]
        logs = [os.path.join(source_dir,subj,'{}_{}_events.csv'.format(subj,log_name))]
        
        for i in range(len(edfs)): # process one EDF at a time
            A = '{}_{}'.format(subj,log_name) # A = '{}_{}_{}'.format(exp_dir, subj, i)
        
            pupilPreprocess = pupil_control.pupilPreprocess(
                subject = subj,
                edf = A,
                experiment_name = log_name,
                project_directory = data_dir,
                eye = eye,
                msgs = msgs, 
                tolkens = tolkens,
                sample_rate = sample_rate,
                tw_blinks = tw_blinks
                )
            # pupilPreprocess.import_raw_data(edf_file=edfs[i],log_file=logs[i],alias=A) # only need to run once, copies files from source directory
            # pupilPreprocess.convert_edfs()              # converts EDF to asc, msg and gaze files
            pupilPreprocess.extract_pupil()             # read trials, and saves time locked pupil series as NPY array in processed folder
            pupilPreprocess.preprocess_pupil()          # blink interpolation, filtering, remove blinks/saccades, split blocks, percent signal change, plots output

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
    higherLevel = feedback_higher_gpe.higherLevel(
        subjects=subjects, 
        experiment_name = log_name,
        project_directory=data_dir, 
        sample_rate=sample_rate,
        time_locked = time_locked,
        pupil_step_lim = pupil_step_lim,                
        baseline_window = baseline_window,              
        pupil_time_of_interest = pupil_time_of_interest,
        )
    # higherLevel.higherlevel_dataframe()  # adds a column for blocks to subjects' logfiles, aves higher level data frame
    # higherLevel.dataframe_phasic_pupil_higher()  # group level data frames for all main effects + interaction
    # higherLevel.dataframe_evoked_pupil_higher()  # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    higherLevel.run_anova_pupil()                # run anovas on pupil dvs (time window)
    # higherLevel.plot_phasic_pupil()              # plots phasic pupil per event of interest, group level, main effects + interaction
    # higherLevel.plot_evoked_pupil()              # plots evoked pupil per event of interest, group level, main effects + interaction
        