#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Pupil dilation offers a time-window in prediction error

Data set #1 Cue-target orientation 2AFC task - Higher Level Functions
Python code O.Colizoli 2023 (olympia.colizoli@donders.ru.nl)
Python 3.6

Notes
-----
>>> conda install -c conda-forge/label/gcc7 mne
================================================
"""

import os, sys, datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
import scipy as sp
import scipy.stats as stats
import statsmodels.api as sm

from copy import deepcopy
from IPython import embed as shell # used for debugging

pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas


""" Plotting Format
############################################
# PLOT SIZES: (cols,rows)
# a single plot, 1 row, 1 col (2,2)
# 1 row, 2 cols (2*2,2*1)
# 2 rows, 2 cols (2*2,2*2)
# 2 rows, 3 cols (2*3,2*2)
# 1 row, 4 cols (2*4,2*1)
# Nsubjects rows, 2 cols (2*2,Nsubjects*2)

############################################
# Define parameters
############################################
"""
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 1, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 7, 
    'ytick.labelsize': 7, 
    'legend.fontsize': 7, 
    'xtick.major.width': 1, 
    'ytick.major.width': 1,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()


class higherLevel(object):
    """Define a class for the higher level analysis.

    Parameters
    ----------
    subjects : list
        List of subject numbers
    group : int or boolean
        Indicating group 0 (flipped) or 1 (normal order) for the counterbalancing of the mapping conditions
    experiment_name : string
        Name of the experiment for output files
    project_directory : str
        Path to the derivatives data directory
    sample_rate : int
        Sampling rate of pupil measurements in Hertz
    time_locked : list
        List of strings indiciting the events for time locking that should be analyzed (e.g., ['cue_locked','target_locked'])
    pupil_step_lim : list 
        List of arrays indicating the size of pupil trial kernels in seconds with respect to first event, first element should max = 0! (e.g., [[-baseline_window,3],[-baseline_window,3]] )
    baseline_window : float
        Number of seconds before each event in self.time_locked that are averaged for baseline correction
    pupil_time_of_interest : list
        List of arrays indicating the time windows in seconds in which to average evoked responses, per event in self.time_locked, see in higher.plot_evoked_pupil (e.g., [[1.0,2.0],[1.0,2.0]])

    Attributes
    ----------
    subjects : list
        List of subject numbers
    group : int or boolean
        Indicating group 0 (flipped) or 1 (normal order) for the counterbalancing of the mapping conditions
    exp : string
        Name of the experiment for output files
    project_directory : str
        Path to the derivatives data directory
    figure_folder : str
        Path to the figure directory
    dataframe_folder : str
        Path to the dataframe directory
    trial_bin_folder : str
        Path to the trial bin directory for conditions 
    jasp_folder : str
        Path to the jasp directory for stats
    sample_rate : int
        Sampling rate of pupil measurements in Hertz
    time_locked : list
        List of strings indiciting the events for time locking that should be analyzed (e.g., ['cue_locked','target_locked'])
    pupil_step_lim : list 
        List of arrays indicating the size of pupil trial kernels in seconds with respect to first event, first element should max = 0! (e.g., [[-baseline_window,3],[-baseline_window,3]] )
    baseline_window : float
        Number of seconds before each event in self.time_locked that are averaged for baseline correction
    pupil_time_of_interest : list
        List of arrays indicating the time windows in seconds in which to average evoked responses, per event in self.time_locked, see in higher.plot_evoked_pupil (e.g., [[1.0,2.0],[1.0,2.0]])
    """
    
    def __init__(self, subjects, group, experiment_name, project_directory, sample_rate, time_locked, pupil_step_lim, baseline_window, pupil_time_of_interest):        
        """Constructor method
        """
        self.subjects = subjects
        self.group = group
        self.exp = experiment_name
        self.project_directory = project_directory
        self.figure_folder = os.path.join(project_directory, 'figures')
        self.dataframe_folder = os.path.join(project_directory, 'data_frames')
        self.trial_bin_folder = os.path.join(self.dataframe_folder,'trial_bins_pupil') # for average pupil in different trial bin windows
        self.jasp_folder = os.path.join(self.dataframe_folder,'jasp') # for dataframes to input into JASP
        self.sample_rate = sample_rate
        self.time_locked = time_locked
        self.pupil_step_lim = pupil_step_lim                
        self.baseline_window = baseline_window              
        self.pupil_time_of_interest = pupil_time_of_interest
        
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
            
        if not os.path.isdir(self.dataframe_folder):
            os.mkdir(self.dataframe_folder)
        
        if not os.path.isdir(self.trial_bin_folder):
            os.mkdir(self.trial_bin_folder)
            
        if not os.path.isdir(self.jasp_folder):
            os.mkdir(self.jasp_folder)
        
        
    def tsplot(self, ax, data, alpha_fill=0.2,alpha_line=1, **kw):
        """Time series plot replacing seaborn tsplot
            
        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The subplot handle to plot in

        data : array
            The data in matrix of format: subject x timepoints

        alpha_line : int
            The thickness of the mean line (default 1)

        kw : list
            Optional keyword arguments for matplotlib.plot().
        """
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        ## confidence intervals
        # cis = self.bootstrap(data)
        # ax.fill_between(x,cis[0],cis[1],alpha=alpha_fill,**kw) # debug double label!
        ## standard error mean
        sde = np.true_divide(sd, np.sqrt(data.shape[0]))
        fill_color = kw['color']
        ax.fill_between(x, est-sde, est+sde, alpha=alpha_fill, color=fill_color, linewidth=0.0) # debug double label!
        
        ax.plot(x, est,alpha=alpha_line,**kw)
        ax.margins(x=0)
    
    
    def bootstrap(self, data, n_boot=10000, ci=68):
        """Bootstrap confidence interval for new tsplot.
        
        Parameters
        ----------
        data : array
            The data in matrix of format: subject x timepoints

        n_boot : int
            Number of iterations for bootstrapping

        ci : int
            Confidence interval range

        Returns
        -------
        (s1,s2) : tuple
            Confidence interval.
        """
        boot_dist = []
        for i in range(int(n_boot)):
            resampler = np.random.randint(0, data.shape[0], data.shape[0])
            sample = data.take(resampler, axis=0)
            boot_dist.append(np.mean(sample, axis=0))
        b = np.array(boot_dist)
        s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
        s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
        return (s1,s2)
        
        
    # common functions
    def cluster_sig_bar_1samp(self,array, x, yloc, color, ax, threshold=0.05, nrand=5000, cluster_correct=True):
        """Add permutation-based cluster-correction bar on time series plot.
        
        Parameters
        ----------
        array : array
            The data in matrix of format: subject x timepoints

        x : array
            x-axis of plot

        yloc : int
            Location on y-axis to draw bar

        color : string
            Color of bar

        ax : matplotlib.axes._subplots.AxesSubplot
            The subplot handle to plot in

        threshold : float
            Alpha value for p-value significance (default 0.05)

        nrand : int 
            Number of permutations (default 5000)

        cluster_correct : bool 
            Perform cluster-based multiple comparison correction if True (default True).
        """
        if yloc == 1:
            yloc = 10
        if yloc == 2:
            yloc = 20
        if yloc == 3:
            yloc = 30
        if yloc == 4:
            yloc = 40
        if yloc == 5:
            yloc = 50

        if cluster_correct:
            whatever, clusters, pvals, bla = mne.stats.permutation_cluster_1samp_test(array, n_permutations=nrand, n_jobs=10)
            for j, cl in enumerate(clusters):
                if len(cl) == 0:
                    pass
                else:
                    if pvals[j] < threshold:
                        for c in cl:
                            sig_bool_indices = np.arange(len(x))[c]
                            xx = np.array(x[sig_bool_indices])
                            try:
                                xx[0] = xx[0] - (np.diff(x)[0] / 2.0)
                                xx[1] = xx[1] + (np.diff(x)[0] / 2.0)
                            except:
                                xx = np.array([xx - (np.diff(x)[0] / 2.0), xx + (np.diff(x)[0] / 2.0),]).ravel()
                            ax.plot(xx, np.ones(len(xx)) * ((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0], color, alpha=1, linewidth=2.5)
        else:
            p = np.zeros(array.shape[1])
            for i in range(array.shape[1]):
                p[i] = sp.stats.ttest_rel(array[:,i], np.zeros(array.shape[0]))[1]
            sig_indices = np.array(p < 0.05, dtype=int)
            sig_indices[0] = 0
            sig_indices[-1] = 0
            s_bar = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0])
            for sig in s_bar:
                ax.hlines(((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0], x[int(sig[0])]-(np.diff(x)[0] / 2.0), x[int(sig[1])]+(np.diff(x)[0] / 2.0), color=color, alpha=1, linewidth=2.5)
    
    
    def fisher_transform(self,r):
        """Compute Fisher transform on correlation coefficient.
        
        Parameters
        ----------
        r : array_like
            The coefficients to normalize
        
        Returns
        -------
        0.5*np.log((1+r)/(1-r)) : ndarray
            Array of shape r with normalized coefficients.
        """
        return 0.5*np.log((1+r)/(1-r))
    
       
    def compute_blink_percentages(self, thresh = 0.40):
        """Computes the percentage of interpolated data per trial and marks trials to be excluded based on treshold
        
        Notes
        -----
        Overwrites original log file (this_log).
        """
        
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory, subj, 'beh', '{}_{}_beh.csv'.format(subj,self.exp)) # derivatives folder            
            this_df = pd.read_csv(this_log, float_precision='%.16f') 
            this_df = this_df.loc[:, ~this_df.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            # loop through each type of event to lock events to...
            for t,time_locked in enumerate(self.time_locked):
                pupil_step_lim = self.pupil_step_lim[t] # kernel size is always the same for each event type
                
                this_blinks = pd.read_csv(os.path.join(self.project_directory, subj, 'beh', '{}_{}_recording-eyetracking_physio_{}_evoked_blinks.csv'.format(subj, self.exp, time_locked)))
                this_blinks = this_blinks.loc[:, ~this_blinks.columns.str.contains('^Unnamed')] # remove all unnamed columns
                
                blinks_per_trial = np.sum(this_blinks, axis=1)
                
                this_df['blinks_ratio'] = blinks_per_trial / (self.sample_rate*(pupil_step_lim[1]-pupil_step_lim[0])) # fraction of whole trial
                this_df['blinks_exclude'] = this_df['blinks_ratio'] > thresh
                
                excluded_trials = this_df[this_df['blinks_exclude']].index.tolist()

                print("{} {} Trials to exclude (>{} interpolated): {}".format(time_locked, subj, thresh, len(excluded_trials)))
                #################################
                # compute within baseline pupil
                
                this_blinks = np.array(this_blinks) # trials x timewindow
                
                base_start = 0
                base_end = self.baseline_window*self.sample_rate
                blinks_baseline = np.sum(this_blinks[:,int(base_start):int(base_end)], axis=1) 
                
                this_df['blink_ratio_{}_baseline'.format(time_locked)] = blinks_baseline / (self.baseline_window*self.sample_rate)
                #################################
                # compute within time windows of interest phasic
                for twi,pupil_time_of_interest in enumerate(self.pupil_time_of_interest[t]): # multiple time windows to average
                    SAVE_TRIALS = []
                    for trial in np.arange(len(this_blinks)):
                        ### PHASIC TIME WINDOWS
                        # in seconds
                        phase_start = -pupil_step_lim[0] + pupil_time_of_interest[0]
                        phase_end = -pupil_step_lim[0] + pupil_time_of_interest[1]
                        # in sample rate units
                        phase_start = int(phase_start*self.sample_rate)
                        phase_end = int(phase_end*self.sample_rate)
                        # sum of blink samples within phasic time window
                        this_phasic = np.sum(this_blinks[trial,phase_start:phase_end]) 
                        SAVE_TRIALS.append(this_phasic)
                    # save phasics
                    this_df['blink_ratio_{}_t{}'.format(time_locked,twi+1)] = np.array(SAVE_TRIALS) / ((pupil_time_of_interest[1]-pupil_time_of_interest[0])*self.sample_rate)

            #######################
            this_df.to_csv(this_log, float_format='%.16f') # save per subject
 
        print('success: compute_blink_percentages')
       
       
    def higherlevel_log_conditions(self,):
        """For each LOG file for each subject, computes mappings, accuracy, RT outliers (3 STD group level)
        
        Notes
        -----
        It was not possible to miss a trial (no max response window).
        Overwrites original log file (this_log).
        
        #############
        # ACCURACY COMPUTATIONS
        #############
        # cue 'cue_ori': 0 = square, 45 = diamond
        # tone 'play_tone': TRUE or FALSE
        # target 'target_ori': 45 degrees  = right orientation, 315 degrees = left orientation
        # counterbalancing: 'normal'
        # "mapping1" always refers to the high frequency cue-target contingencies in phase 1
        # trials 1-200 phase1, trials 201-400 phase2
        # phase1 = np.arange(1,201) # excluding 201
        # phase2 = np.arange(201,401) # excluding 401
        """
        # normal congruency phase1: combinations of cue, tone and target:
        mapping1 = ['0_True_45','0_False_45','45_True_315','45_False_315']
        mapping2 = ['0_True_315','0_False_315','45_True_45','45_False_45']
        
        # loop through subjects' log files
        # make a copy in derivatives folder to add phasics to
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory,subj,'beh','{}_{}_beh.csv'.format(subj,self.exp)) # derivatives folder            
            this_df = pd.read_csv(this_log, float_precision='%.16f') 
            # drop 'updating' column if it exists'
            this_df = this_df.drop(['updating'], axis=1, errors='ignore')
            
            ###############################
            # compute column for MAPPING
            # col values 'mapping1': mapping1 = 1, mapping2 = 0
            mapping_normal = [
                # KEEP ORIGINAL MAPPINGS TO SEE 'FLIP'
                (this_df['cue_ori'] == 0) & (this_df['play_tone'] == True) & (this_df['target_ori'] == 45), #'0_True_45'
                (this_df['cue_ori'] == 0) & (this_df['play_tone'] == False) & (this_df['target_ori'] == 45), #'0_False_45'
                (this_df['cue_ori'] == 45) & (this_df['play_tone'] == True) & (this_df['target_ori'] == 315), #'45_True_315'
                (this_df['cue_ori'] == 45) & (this_df['play_tone'] == False) & (this_df['target_ori'] == 315), #'45_False_315'

                ]
                
            mapping_counter = [
                # KEEP ORIGINAL MAPPINGS TO SEE 'FLIP'
                (this_df['cue_ori'] == 0) & (this_df['play_tone'] == True) & (this_df['target_ori'] == 315), #'0_True_315'
                (this_df['cue_ori'] == 0) & (this_df['play_tone'] == False) & (this_df['target_ori'] == 315), #'0_False_315',
                (this_df['cue_ori'] == 45) & (this_df['play_tone'] == True) & (this_df['target_ori'] == 45), #'45_True_45'
                (this_df['cue_ori'] == 45) & (this_df['play_tone'] == False) & (this_df['target_ori'] == 45), #'45_False_45'
                ]
                
            values = [1,1,1,1]
            
            if self.group[s]: # 1 for normal_order
                this_df['mapping1'] = np.select(mapping_normal, values)
            else:
                this_df['mapping1'] = np.select(mapping_counter, values)
            
            ###############################
            # compute column for MODEL PHASE
            this_df['phase1'] = np.array(this_df['trial_counter'] <= 200, dtype=int) # phase = 1, revision phase = 0
            
            ###############################
            # compute column for MAPPING FREQUENCY
            frequency = [
                # phase 1
                (this_df['phase1'] == 1) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 1), # mapping 1 phase1 tone 80%
                (this_df['phase1'] == 1) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 0), # mapping 1 phase1 no tone 80%
                (this_df['phase1'] == 1) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 1), # mapping 2 phase1 tone 20%
                (this_df['phase1'] == 1) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 0), # mapping 2 phase1 no tone 20%
                # phase 2
                (this_df['phase1'] == 0) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 1), # mapping 1 phase2 tone 20% FLIP!!
                (this_df['phase1'] == 0) & (this_df['mapping1'] == 1) & (this_df['play_tone'] == 0), # mapping 1 phase2 no tone 80%
                (this_df['phase1'] == 0) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 1), # mapping 2 phase2 tone 80% FLIP
                (this_df['phase1'] == 0) & (this_df['mapping1'] == 0) & (this_df['play_tone'] == 0), # mapping 2 phase2 no tone 20%
                ]
            values = [80,80,20,20,20,80,80,20]
            this_df['frequency'] = np.select(frequency, values)
            
            ###############################
            # compute column for ACCURACY
            accuracy = [
                (this_df['target_ori'] == 45) & (this_df['keypress'] == 'right'), 
                (this_df['target_ori'] == 315) & (this_df['keypress'] == 'left')
                ]
            values = [1,1]
            this_df['correct'] = np.select(accuracy, values)
            
            ###############################
            # add column for SUBJECT
            this_df['subject'] = np.repeat(subj,this_df.shape[0])
            
            # resave log file with new columns in derivatives folder
            this_df = this_df.loc[:, ~this_df.columns.str.contains('^Unnamed')] # remove all unnamed columns
            this_df.to_csv(os.path.join(this_log), float_format='%.16f')
        print('success: higherlevel_log_conditions')
       
       
    def higherlevel_get_phasics(self,):
        """Computes phasic pupil (evoked average) in selected time window per trial and add phasics to behavioral data frame. 
        
        Notes
        -----
        Overwrites original log file (this_log).
        """
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory,subj,'beh','{}_{}_beh.csv'.format(subj,self.exp)) # derivatives folder
            B = pd.read_csv(this_log, float_precision = '%.16f') # behavioral file
            ### DROP EXISTING PHASICS COLUMNS TO PREVENT OLD DATA
            try: 
                B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                B = B.loc[:, ~B.columns.str.contains('_locked')] # remove all old phasic pupil columns
            except:
                pass
                
            # loop through each type of event to lock events to...
            for t,time_locked in enumerate(self.time_locked):
                
                pupil_step_lim = self.pupil_step_lim[t] # kernel size is always the same for each event type
                
                for twi,pupil_time_of_interest in enumerate(self.pupil_time_of_interest[t]): # multiple time windows to average
                    # load evoked pupil file (all trials)
                    P = pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f') 
                    P = P.loc[:, ~P.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    P = np.array(P)
                
                    SAVE_TRIALS = []
                    for trial in np.arange(len(P)):
                        # in seconds
                        phase_start = -pupil_step_lim[0] + pupil_time_of_interest[0] # have to add the baseline window in to reference event onset from at zero!
                        phase_end = -pupil_step_lim[0] + pupil_time_of_interest[1]
                        # in sample rate units
                        phase_start = int(phase_start*self.sample_rate)
                        phase_end = int(phase_end*self.sample_rate)
                        # mean within phasic time window
                        this_phasic = np.nanmean(P[trial,phase_start:phase_end]) 
                        SAVE_TRIALS.append(this_phasic)
                    # save phasics
                    B['pupil_{}_t{}'.format(time_locked,twi+1)] = np.array(SAVE_TRIALS)

                    #######################
                    B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    B.to_csv(this_log, float_format='%.16f')
                    print('subject {}, {} phasic pupil extracted {}'.format(subj, time_locked, pupil_time_of_interest))
        print('success: higherlevel_get_phasics')
        
                
    def create_subjects_dataframe(self, exclude_interp=0):
        """Combine behavior and phasic pupil dataframes including pupil baselines of all subjects into a single large dataframe. 
        
        Parameters
        ----------
        exclude_inter : boolean (default = 0)
            Exclude the trials the have too much interpolate data (1) or not (0). 
        
        Notes
        -----
        Drops phase 2 trials.
        Adds target_locked baselines to dataframe.
        Flag outliers based on RT (+-3 STD, separate column) per subject. 
        Output in dataframe folder: task-predictions_subjects.csv
        """     
        DF = pd.DataFrame() # ALL SUBJECTS phasic pupil + behavior 
        
        # loop through subjects, get behavioral log files
        for s,subj in enumerate(self.subjects):
            this_data = pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_beh.csv'.format(subj,self.exp)), float_precision='%.16f')
            this_data = this_data.loc[:, ~this_data.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            # open baseline pupil to add to dataframes as well
            this_baseline = pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_baselines.csv'.format(subj,self.exp,'target_locked')), float_precision='%.16f')
            this_baseline = this_baseline.loc[:, ~this_baseline.columns.str.contains('^Unnamed')] # remove all unnamed columns
            this_data['pupil_baseline_target_locked'] = np.array(this_baseline)
            
            ###############################
            # compute column for OUTLIER REACTION TIMES: transform to Z and exclude +- 3*STD seconds
            RT = stats.zscore(this_data['reaction_time']) # use STD based on z transform first
            outlier_rt = [
                (RT < -3), # lower limit < -3 STD zscore 
                (RT > 3) # upper limit > 3 STD zscore above mean
                ]
            values = [1,1]
            this_data['outlier_rt'] = np.select(outlier_rt, values)
                        
            ###############################            
            # concatenate all subjects
            DF = pd.concat([DF,this_data],axis=0)
        
        # drop phase 2 trials
        DF = DF[DF['trial_counter']<=200]
        
        ### mark all trials to exclude 
        if exclude_interp:
            DF['exclude'] = DF['outlier_rt']+DF['blinks_exclude']
            DF['exclude'] = (DF['exclude'] > 0).astype(int)
        else: # only exclude RT outliers
            DF['exclude'] = DF['outlier_rt']
            
        # how many trials excluded due to blinks? 
        print('Total blink trials excluded = {}%'.format(np.true_divide(np.sum(DF['blinks_exclude']),DF.shape[0])*100))
        # per subject?
        blinks_excluded = DF.groupby(['subject',])['blinks_exclude'].sum()
        blinks_excluded.to_csv(os.path.join(self.dataframe_folder,'{}_blinks_excluded_counts_subject.csv'.format(self.exp)), float_format='%.16f')
        # blinks per condition per subject
        blinks_excluded = DF.groupby(['subject', 'mapping1', 'correct'])['blinks_exclude'].sum()
        blinks_excluded.to_csv(os.path.join(self.dataframe_folder,'{}_blinks_excluded_conditions_subject.csv'.format(self.exp)), float_format='%.16f')
                
        ### print how many outliers in phase 1
        print('Phase 1 outliers = {}%'.format(np.true_divide(np.sum(DF['outlier_rt']),DF.shape[0])*100))

        # trial counts (note: no missing trials because no maximum response window!)
        missing = DF.groupby(['subject','keypress'])['keypress'].value_counts()
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_counts_subject.csv'.format(self.exp)), float_format='%.16f')
        # combination of conditions
        missing = DF.groupby(['subject','mapping1','play_tone','correct','phase1'])['keypress'].count()
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_counts_conditions.csv'.format(self.exp)), float_format='%.16f')

        #####################
        # save whole dataframe with all subjects
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_format='%.16f')
        #####################
        print('success: create_subjects_dataframe')


    def average_conditions(self,):
        """Average the DVs per subject per condition of interest. 

        Notes
        -----
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject','trial_counter'],inplace=True)
        DF.reset_index()
                    
        ############################
        # drop bad trials
        DF = DF[DF['exclude']==0]
        ############################
        
        '''
        ######## CORRECT x MAPPING1 x TIME WINDOW ########
        '''
        DFOUT = DF.groupby(['subject','correct','mapping1']).aggregate({'pupil_target_locked_t1':'mean', 'pupil_target_locked_t2':'mean'})
        # DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct-mapping1-timewindow_{}.csv'.format(self.exp,pupil_dv))) # FOR PLOTTING
        # save for RMANOVA format
        DFANOVA =  DFOUT.unstack(['mapping1','correct']) 
        print(DFANOVA.columns)
        DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
        cols = ['20_error_t1', '80_error_t1', '20_correct_t1', '80_correct_t1', '20_error_t2', '80_error_t2', '20_correct_t2', '80_correct_t2']
        print(cols)
        DFANOVA.columns = cols
        DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-mapping1-timewindow_rmanova.csv'.format(self.exp)), float_format='%.16f') # for stats
        '''
        ######## CORRECT x MAPPING1 ########
        '''
        for pupil_dv in ['reaction_time', 'pupil_target_locked_t1', 'pupil_target_locked_t2', 'pupil_baseline_target_locked']:
            DFOUT = DF.groupby(['subject','correct','mapping1'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct-mapping1_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['mapping1','correct']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            cols = ['20_error', '80_error', '20_correct', '80_correct']
            print(cols)
            DFANOVA.columns = cols
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-mapping1_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        '''
        ######## MAPPING1 ########
        '''
        for pupil_dv in ['correct','reaction_time','pupil_target_locked_t1','pupil_target_locked_t2', 'pupil_baseline_target_locked']: 
            DFOUT = DF.groupby(['subject','mapping1'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_mapping1_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['mapping1']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_mapping1_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        print('success: average_conditions')
        
        
    def plot_phasic_pupil_pe(self,):
        """Plot the phasic pupil target_locked interaction frequency and accuracy.

        Notes
        -----
        GROUP LEVEL DATA
        Separate lines for correct, x-axis is frequency (mapping) conditions.
        Figure output as PDF in figure folder.
        """
        ylim = [ 
            [-3.25, 2.25], # t1
            [-3.25, 2.25], # t2
            [-3, 3], # baseline
            [0.6,1.5] # RT
        ]
        tick_spacer = [1, 1, 2, .2]
        
        dvs = ['pupil_target_locked_t1', 'pupil_target_locked_t2', 'pupil_baseline_target_locked', 'reaction_time' ]
        ylabels = ['Pupil response\n(% signal change)', 'Pupil response\n(% signal change)', 'Pupil response\n(% signal change)', 'RT (s)']
        factor = ['mapping1','correct']
        xlabel = 'Cue-target frequency'
        xticklabels = ['20%','80%'] 
        labels = ['Error','Correct']
        colors = ['red','blue'] 
        
        xind = np.arange(len(xticklabels))
        dot_offset = [0.1,-0.1]
        
        for dvi, pupil_dv in enumerate(dvs):
            
            fig = plt.figure(figsize=(2, 2))
            ax = fig.add_subplot(111) # 1 subplot per bin window
            
            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_correct-mapping1_{}.csv'.format(self.exp,pupil_dv)), float_precision='%.16f')
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby(factor)[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
            
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
 
            # plot line graph
            for x in[0,1]: # split by error, correct
                D = GROUP[GROUP['correct']==x]
                print(D)
                ax.errorbar(xind, np.array(D['mean']), yerr=np.array(D['sem']), marker='o', markersize=3, fmt='-', elinewidth=1, label=labels[x], capsize=3, color=colors[x], alpha=1)
                
            # set figure parameters
            ax.set_title('{}'.format(pupil_dv))                
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            ax.set_ylim(ylim[dvi])
            # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer[dvi]))
            ax.legend()
        
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_correct-mapping1_{}_lines.pdf'.format(self.exp, pupil_dv)))
        print('success: plot_phasic_pupil_pe')
        

    def plot_behavior(self, ):
        """Plot the group level means of accuracy and RT per frequency (mapping) condition.

        Notes
        -----
        2 figures, GROUP LEVEL DATA
        x-axis is frequency conditions.
        Figure output as PDF in figure folder.
        """
        dvs = ['correct','reaction_time']
        ylabels = ['Accuracy', 'RT (s)']
        factor = 'mapping1'
        xlabel = 'Cue-target frequency'
        xticklabels = ['20%','80%'] 
        color = 'black'        
        bar_width = 0.7
        xind = np.arange(len(xticklabels))
                
        for dvi,pupil_dv in enumerate(dvs):
            
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111) # 1 subplot per bin windo

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_{}_{}.csv'.format(self.exp,factor,pupil_dv)), float_precision='%.16f')
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby([factor])[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
                        
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
                       
            # plot bar graph
            for x in GROUP[factor]:
                ax.bar(xind[x],np.array(GROUP['mean'][x]), width=bar_width, yerr=np.array(GROUP['sem'][x]), capsize=3, color=(0,0,0,0), edgecolor='black', ecolor='black')
                
            # individual points, repeated measures connected with lines
            DFIN = DFIN.groupby(['subject',factor])[pupil_dv].mean() # hack for unstacking to work
            DFIN = DFIN.unstack(factor)
            for s in np.array(DFIN):
                ax.plot(xind, s, linestyle='-', marker='o', markersize=3, fillstyle='full', color='black', alpha=.2) # marker, line, black

            # set figure parameters
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            if pupil_dv == 'correct':
                ax.set_ylim([0.0,1.])
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.2))
            else:
                ax.set_ylim([0.2,1.8]) #RT
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.4))

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_mapping1_{}.pdf'.format(self.exp, pupil_dv)))
        print('success: plot_behav')

    
    def individual_differences(self,):
       """Correlate frequency effect in pupil DV with frequency effect in accuracy across participants, then plot.
       
       Notes
       -----
       3 figures: 1 per pupil DV
       """
       dvs = ['pupil_target_locked_t1','pupil_target_locked_t2', 'pupil_baseline_target_locked']
              
       for sp,pupil_dv in enumerate(dvs):
           fig = plt.figure(figsize=(2,2))
           ax = fig.add_subplot(111) # 1 subplot per bin window
           
           B = pd.read_csv(os.path.join(self.jasp_folder,'{}_mapping1_correct_rmanova.csv'.format(self.exp)), float_precision='%.16f')
           P = pd.read_csv(os.path.join(self.jasp_folder,'{}_mapping1_{}_rmanova.csv'.format(self.exp, pupil_dv)), float_precision='%.16f')

           # frequency effect
           P['main_effect_freq'] = (P['1']-P['0']) # mapping1=1 is 80% condition
           B['main_effect_freq'] = (B['1']-B['0']) # fraction correct
           
           x = np.array(B['main_effect_freq'])
           y = np.array(P['main_effect_freq'])           
           # all subjects
           r,pval = stats.spearmanr(x,y)
           print('all subjects')
           print(pupil_dv)
           print('r={}, p-val={}'.format(r,pval))

           # all subjects
           ax.plot(x, y, 'o', markersize=3, color='green') # marker, line, black
           m, b = np.polyfit(x, y, 1)
           ax.plot(x, m*x+b, color='green',alpha=.5, label='all participants')
           
           # set figure parameters
           ax.set_title('r={}, p-val={}'.format(np.round(r,2),np.round(pval,3)))
           ax.set_ylabel('{} (80%-20%)'.format(pupil_dv))
           ax.set_xlabel('accuracy (80%-20%)')
           # ax.legend()
           
           plt.tight_layout()
           fig.savefig(os.path.join(self.figure_folder,'{}_frequency_individual_differences_{}.pdf'.format(self.exp, pupil_dv)))
       print('success: individual_differences')
        
    
    def dataframe_evoked_pupil_higher(self):
        """Compute evoked pupil responses.
        
        Notes
        -----
        Split by conditions of interest. Save as higher level dataframe per condition of interest. 
        Evoked dataframes need to be combined with behavioral data frame, looping through subjects. 
        DROP PHASE 2 trials.
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        csv_names = deepcopy(['subject','correct','mapping1','correct-mapping1'])
        factors = [['subject'],['correct'],['mapping1'],['correct','mapping1']]
        
        for t,time_locked in enumerate(self.time_locked):
            # Loop through conditions                
            for c,cond in enumerate(csv_names):
                # intialize dataframe per condition

                COND = pd.DataFrame()
                g_idx = deepcopy(factors)[c]       # need to add subject idx for groupby()
                
                if not cond == 'subject':
                    g_idx.insert(0, 'subject') # get strings not list element
                
                for s,subj in enumerate(self.subjects):
                    SBEHAV = DF[DF['subject']==subj].reset_index()
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f'))
                    SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
                    #############################
                    # DROP THE LAST 200 trials from evoked DF
                    SPUPIL = SPUPIL.iloc[:200,:]
                    
                    # merge behavioral and evoked dataframes so we can group by conditions
                    SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
                    
                    #### DROP OMISSIONS HERE ####
                    SDATA = SDATA[SDATA['exclude'] == 0] # drop outliers based on RT
                    #############################
                    
                    evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    df = SDATA.groupby(g_idx)[evoked_cols].mean() # only get kernels out
                    df = pd.DataFrame(df).reset_index()
                    # add to condition dataframe
                    COND = pd.concat([COND,df],join='outer',axis=0) # can also do: this_cond = this_cond.append()  
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,cond)), float_format='%.16f')
        print('success: dataframe_evoked_pupil_higher')
    
    
    def plot_evoked_pupil(self):
        """Plot evoked pupil time courses.
        
        Notes
        -----
        4 figures: mean response, accuracy, frequency, accuracy*frequency.
        Always target_locked pupil response.
        """
        ylim_feed = [-2.5,2.5]
        tick_spacer = 2.5
        
        #######################
        # FEEDBACK MEAN RESPONSE
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
                
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,factor)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
    
        xticklabels = ['mean response']
        colors = ['black'] # black
        alphas = [1]

        # plot time series
        i=0
        TS = np.array(COND.iloc[:,-kernel:]) # index from back to avoid extra unnamed column pandas
        self.tsplot(ax, TS, color='k', label=xticklabels[i])
        self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
    
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            
        # shade baseline pupil
        twb = [-self.baseline_window, 0]
        baseline_onset = int(abs(twb[0]*self.sample_rate))
        twb_begin = int(baseline_onset + (twb[0]*self.sample_rate))
        twb_end = int(baseline_onset + (twb[1]*self.sample_rate))
        ax.axvspan(twb_begin,twb_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
                
        # compute peak of mean response to center time window around
        m = np.mean(TS,axis=0)
        argm = np.true_divide(np.argmax(m),self.sample_rate) + self.pupil_step_lim[t][0] # subtract pupil baseline to get timing
        print('mean response = {} peak @ {} seconds'.format(np.max(m),argm))
        # ax.axvline(np.argmax(m), lw=0.25, alpha=0.5, color = 'k')
        
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # CORRECT
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'correct'
        factor = 'correct'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['Error','Correct']
        colorsts = ['r','b',]
        alpha_fills = [0.2,0.2] # fill
        alpha_lines = [1,1]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_difference = save_conds[0]-save_conds[1]
        self.cluster_sig_bar_1samp(array=pe_difference, x=pd.Series(range(pe_difference.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # FREQUENCY
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'mapping1'
        factor = 'mapping1'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['20%','80%']
        colorsts = ['indigo','indigo',]
        alpha_fills = [0.2,0.2] # fill
        alpha_lines = [0.2,1]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_difference = save_conds[0]-save_conds[1]
        self.cluster_sig_bar_1samp(array=pe_difference, x=pd.Series(range(pe_difference.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # CORRECT x MAPPING1
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'correct-mapping1'
        factor = ['correct','mapping1']
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        ########
        # make unique labels for each of the 4 conditions
        conditions = [
            (COND['correct'] == 0) & (COND['mapping1'] == 1), # Easy Error 1
            (COND['correct'] == 1) & (COND['mapping1'] == 1), # Easy Correct 2
            (COND['correct'] == 0) & (COND['mapping1'] == 0), # Hard Error 3
            (COND['correct'] == 1) & (COND['mapping1'] == 0), # Hard Correct 4
            ]
        values = [1,2,3,4]
        conditions = np.select(conditions, values) # don't add as column to time series otherwise it gets plotted
        ########
                    
        xticklabels = ['Error 80%','Correct 80%','Error 20%','Correct 20%']
        colorsts = ['r','b','r','b']
        alpha_fills = [0.2,0.2,0.1,0.1] # fill
        alpha_lines = [1,1,.8,.8]
        linestyle= ['solid','solid','dashed','dashed']
        save_conds = []
        # plot time series
        
        for i,x in enumerate(values):
            TS = COND[conditions==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, linestyle=linestyle[i], color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_interaction = (save_conds[0]-save_conds[1]) - (save_conds[2]-save_conds[3])
        self.cluster_sig_bar_1samp(array=pe_interaction, x=pd.Series(range(pe_interaction.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        ax.legend(loc='best')
                
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked_{}.pdf'.format(self.exp, csv_name)))
        print('success: plot_evoked_pupil')
        
    
    def information_theory_code_stimuli(self, fn_in):
        """Add a new column in the subjects dataframe to give each cue-target pair a unique identifier.
        
        Parameters
        ----------
        fn_in : str
            The path to the subjects' dataframe.
        
        Notes
        -----
        square-left = 0
        square-right = 1
        diamond-left = 2
        diamond-right = 3
        
        New column name is "cue_target_pair"
        """
        
        df_in = pd.read_csv(fn_in, float_precision='%.16f')
                
        # make new column to give each cue-target combination a unique identifier (1, 2, 3 or 4)        
        mapping = [
            # KEEP ORIGINAL MAPPINGS TO SEE 'FLIP'
            (df_in['cue_ori'] == 0) & (df_in['target_ori'] == 315), # square left
            (df_in['cue_ori'] == 0) & (df_in['target_ori'] == 45), # sqaure right
            (df_in['cue_ori'] == 45) & (df_in['target_ori'] == 315), # diamond left
            (df_in['cue_ori'] == 45) & (df_in['target_ori'] == 45), # diamond right
            ]
        
        elements = [0,1,2,3] # also elements is the same as priors (start with 0 so they can be indexed by element)
        
        df_in['cue_target_pair'] = np.select(mapping, elements)
        
        df_in.to_csv(fn_in, float_format='%.16f') # save with new columns
        print('success: information_theory_code_stimuli')
        
    
    def idt_model(self, df, df_data_column, elements):
        """Process Ideal Learner Model.
        
        Parameters
        ----------
        df : pandas dataframe
            The dataframe to apply the Ideal Learner Model to.
        
        df_data_column : str
            The name of the column that refers to the cue-target pairs for all trials in the experiment.
        
        elements : list
            The list of unique indentifiers for the cue-target pairs.
        
        Returns
        -------
        [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D]: list
            A list containing all model parameters (see notes).
            
        Notes
        -----
        Ideal Learner Model adapted from Poli, Mars, & Hunnius (2020).
        See also: https://github.com/FrancescPoli/eye_processing/blob/master/ITDmodel.m
        
        Using uniform priors.
        
        Model Output Notes:
        model_e = trial sequence
        model_P = probabilities of all elements at each trial
        model_p = probability of current element at current trial
        model_I = surprise of all elements at each trial (i.e., complexity)
        model_i = surprise of current element at current trial
        model_H = entropy at current trial
        model_CH = cross-entropy at current trial
        model_D = KL-divergence at current trial
        """
        
        data = np.array(df[df_data_column])
    
        # initialize output variables for current subject
        model_e = [] # trial sequence
        model_P = [] # probabilities of all elements
        model_p = [] # probability of current element 
        model_I = [] # surprise of all elements 
        model_i = [] # surprise of current element 
        model_H = [] # entropy at current trial
        model_CH = [] # cross-entropy at current trial
        model_D = []  # KL-divergence at current trial
    
        # loop trials
        for t,trial_counter in enumerate(df['trial_counter']):
            vector = data[:t+1] #  trial number starts at 0, all the targets that have been seen so far
            
            model_e.append(vector[-1])  # element in current trial = last element in the vector
            
            # print(vector)
            if t < 1: # if it's the first trial, our expectations are based only on the prior (values)
                # FLAT PRIORS
                alpha1 = np.ones(len(elements)) # np.sum(alpha) == len(elements), flat prior
                p1 = alpha1 / len(elements) # probablity, i.e., np.sum(p1) == 1
                p = p1
            
            # at every trial, we compute surprise based on the probability
            model_P.append(p)             # probability (all elements) 
            model_p.append(p[vector[-1]]) # probability of current element
            # Surprise is defined by the negative log of the probability of the current trial given the previous trials.
            I = -np.log2(p)     # complexity of every event (each cue_target_pair is a potential event)
            i = I[vector[-1]]   # surprise of the current event (last element in vector)
            model_I.append(I)
            model_i.append(i)
            
            # EVERYTHING AFTER HERE IS CALCULATED INCLUDING CURRENT EVENT
            # Updated estimated probabilities (posterior)
            p = []
            for k in elements:
                # +1 because in the prior there is one element of the same type; +len(alpha) because in the prior there are #alpha elements
                # The influence of the prior should be sampled by a distribution or
                # set to a certain value based on Kidd et al. (2012, 2014)
                p.append((np.sum(vector == k) + alpha1[k]) / (len(vector) + len(alpha1)))       

            H = -np.sum(p * np.log2(p)) # entropy (note that np.log2(1/p) is equivalent to multiplying the whole sum by -1)
            model_H.append(H)   # entropy
            
            # once we have the updated probabilities, we can compute KL Divergence, Entropy and Cross-Entropy
            prevtrial = t-1
            if prevtrial < 0: # first trial
                D = np.sum(p * (np.log2(p / np.array(p1)))) # KL divergence, after vs. before, same direction as Poli et al. 2020
            else:
                D = np.sum(p * (np.log2(p / np.array(model_P[prevtrial])))) # KL divergence, after vs. before, same direction as Poli et al. 2020
            
            CH = H + D # Cross-entropy
    
            model_CH.append(CH) # cross-entropy
            model_D.append(D)   # KL divergence
        
        return [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D]
        
        
    def information_theory_estimates(self, ):
        """Run subject loop on Ideal Learner Model and save model estimates.
        
        Notes
        -----
        Ideal Learner Model adapted from Poli, Mars, & Hunnius (2020).
        See also: https://github.com/FrancescPoli/eye_processing/blob/master/ITDmodel.m
        
        Model estimates that are saved in subject's dataframe:
        model_i = surprise of current element at current trial
        model_H = entropy at current trial
        model_D = KL-divergence at current trial
        """
        
        elements = [0,1,2,3]
        
        fn_in = os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp))
        self.information_theory_code_stimuli(fn_in) # code stimuli based on predictions and based on targets

        df_in = pd.read_csv(fn_in, float_precision='%.16f')
        df_in = df_in.loc[:, ~df_in.columns.str.contains('^Unnamed')]
        # sort by subjects then trial_counter in ascending order
        df_in.sort_values(by=['subject', 'trial_counter'], ascending=True, inplace=True)
        
        df_out = pd.DataFrame()
        
        # loop subjects
        for s,subj in enumerate(self.subjects):
            
            # get current subjects data only
            this_df = df_in[df_in['subject']==subj].copy()
            
            # the input to the model is the trial sequence = the order of cue_target/prediction_pair for each participant
            [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D] = self.idt_model(this_df, 'cue_target_pair', elements)
            
            this_df['model_p'] = np.array(model_p)
            this_df['model_i'] = np.array(model_i)
            this_df['model_H'] = np.array(model_H)
            this_df['model_D'] = np.array(model_D)
            df_out = pd.concat([df_out, this_df])    # add current subject df to larger df
        
        # save whole DF
        df_out.to_csv(fn_in, float_format='%.16f') # overwrite subjects dataframe
        print('success: information_theory_estimates')
    

    def average_information_conditions(self,):
        """Average the DVs per subject per condition of interest. 

        Notes
        -----
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder, '{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject', 'trial_counter'],inplace=True)
        DF.reset_index()
                    
        ############################
        # drop outliers
        DF = DF[DF['exclude']==0]
        ############################

        '''
        ######## CORRECT x MAPPING1 ########
        '''
        for pupil_dv in ['model_i', 'model_H', 'model_D']:
            DFOUT = DF.groupby(['subject', 'correct', 'mapping1'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder, '{}_correct-mapping1_{}.csv'.format(self.exp, pupil_dv)), float_format='%.16f') # FOR PLOTTING
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['mapping1', 'correct']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            cols = ['20_error', '80_error', '20_correct', '80_correct']
            print(cols)
            DFANOVA.columns = cols
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-mapping1_{}_rmanova.csv'.format(self.exp, pupil_dv)), float_format='%.16f') # for stats
        '''
        ######## MAPPING1 ########
        '''
        for pupil_dv in ['model_i','model_H','model_D']: 
            DFOUT = DF.groupby(['subject', 'mapping1'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_mapping1_{}.csv'.format(self.exp, pupil_dv)), float_format='%.16f') # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['mapping1']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_mapping1_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        print('success: average_information_conditions')


    def plot_information(self, ):
        """Plot the model parameters across trials and average over subjects.
            Then, plot the model parameters by frequency (mapping1)

        Notes
        -----
        1 figure, GROUP LEVEL DATA
        x-axis is trials or frequency conditions.
        Figure output as PDF in figure folder.
        """
        dvs = ['model_D', 'model_i','model_H']
        ylabels = ['KL divergence', 'Surprise', 'Entropy', ]
        xlabel = 'Trials'
        colors = [ 'purple', 'teal', 'orange',]    
        
        fig = plt.figure(figsize=(4,4))
        
        subplot_counter = 1
        # PLOT ACROSS TRIALS
        for dvi, pupil_dv in enumerate(dvs):

            ax = fig.add_subplot(3, 3, subplot_counter) # 1 subplot per bin windo
            
            DFIN = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
                        
            subject_array = np.zeros((len(self.subjects), np.max(DFIN['trial_counter'])))
            
            for s, subj in enumerate(self.subjects):
                this_df = DFIN[DFIN['subject']==subj].copy()                
                subject_array[s,:] = np.ravel(this_df[[pupil_dv]])
                            
            self.tsplot(ax, subject_array, color=colors[dvi], label=ylabels[dvi])
    
            # set figure parameters
            ax.set_xlim([0,200])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabels[dvi])
            # ax.legend()
            subplot_counter += 1
            
        # PLOT ACROSS FREQUENCY CONDITIONS
        factor = 'mapping1'
        xlabel = 'Cue-target frequency'
        xticklabels = ['20%','80%'] 
        bar_width = 0.6
        xind = np.arange(len(xticklabels))
        
        for dvi, pupil_dv in enumerate(dvs):

                ax = fig.add_subplot(3, 3, subplot_counter) # 1 subplot per bin windo
            
                DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_{}_{}.csv'.format(self.exp,factor,pupil_dv)), float_precision='%.16f')
                DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
                # Group average per BIN WINDOW
                GROUP = pd.DataFrame(DFIN.groupby([factor])[pupil_dv].agg(['mean','std']).reset_index())
                GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
                print(GROUP)
                        
                # ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
                       
                # plot bar graph
                for x in GROUP[factor]:
                    ax.bar(xind[x],np.array(GROUP['mean'][x]), width=bar_width, yerr=np.array(GROUP['sem'][x]), capsize=3, color=colors[dvi], edgecolor='black', ecolor='black')
                
                # # individual points, repeated measures connected with lines
                # DFIN = DFIN.groupby(['subject',factor])[pupil_dv].mean() # hack for unstacking to work
                # DFIN = DFIN.unstack(factor)
                # for s in np.array(DFIN):
                #     ax.plot(xind, s, linestyle='-', marker='o', markersize=3, fillstyle='full', color='black', alpha=.2) # marker, line, black

                # set figure parameters
                ax.set_ylabel(ylabels[dvi])
                ax.set_xlabel(xlabel)
                ax.set_xticks(xind)
                ax.set_xticklabels(xticklabels)
                if pupil_dv == 'model_H':
                    ax.set_ylim([1.7, 1.8])
                subplot_counter += 1
                
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_information.pdf'.format(self.exp)))
        print('success: plot_information')
        
        
    def pupil_information_correlation_matrix(self,):
        """Correlate information variables to evaluate multicollinearity.
        
        Notes
        -----
        Model estimates that are correlated per subject the tested at group level:
        model_i = surprise of current element at current trial
        model_H = entropy at current trial
        model_D = KL-divergence at current trial
        
        See figure folder for plot and output of t-test.
        """
        
        ivs = ['model_i', 'model_H', 'model_D']
        labels = ['i' , 'H', 'KL']

        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')

        ############################
        # drop outliers
        DF = DF[DF['exclude']==0]
        ############################
        
        corr_out = []

        # loop subjects
        for s, subj in enumerate(np.unique(DF['subject'])):
            # get current subject's data only
            this_df = DF[DF['subject']==subj].copy()
                            
            x = this_df[ivs] # select information variable columns
            x_corr = x.corr() # correlation matrix
            
            corr_out.append(x_corr) # beta KLdivergence (target-prediction)
        
        corr_subjects = np.array(corr_out)
        corr_mean = np.mean(corr_subjects, axis=0)
        corr_std = np.std(corr_subjects, axis=0)
        
        t, pvals = sp.stats.ttest_1samp(corr_subjects, 0, axis=0)
        
        f = open(os.path.join(self.figure_folder, '{}_pupil_information_correlation_matrix.txt'.format(self.exp)), "w")
        f.write('corr_mean')
        f.write('\n')
        f.write('{}'.format(corr_mean))
        f.write('\n')
        f.write('\n')
        f.write('tvals')
        f.write('\n')
        f.write('{}'.format(t))
        f.write('\n')
        f.write('\n')
        f.write('pvals')
        f.write('\n')
        f.write('{}'.format(pvals))
        f.close
        
        ### PLOT ###
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(121)
        cbar_ax = fig.add_subplot(122)
        
        # mask for significance
        mask_pvals = pvals < 0.05
        mask_pvals = ~mask_pvals # True means mask this cell
        
        # plot only lower triangle
        mask = np.triu(np.ones_like(corr_mean))
        mask = mask + mask_pvals # only show sigificant correlations in heatmap
        
        ax = sns.heatmap(corr_mean, vmin=-1, vmax=1, mask=mask, cmap='bwr', cbar_ax=cbar_ax, xticklabels=labels, yticklabels=labels, square=True, annot=True, ax=ax)
        
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_pupil_information_correlation_matrix.pdf'.format(self.exp)))
        print('success: pupil_information_correlation_matrix')
                

    def dataframe_evoked_correlation(self):
        """Compute correlation of pupil response with model estimate at each time point (with other model estimates removed).
        
        Notes
        -----
        DROP PHASE 2 trials.
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        
        Correlations are done for all trials as well as for correct and error trials separately.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        
        ivs = ['model_i', 'model_H', 'model_D']
        
        df_out = pd.DataFrame() # timepoints x subjects
        for t,time_locked in enumerate(self.time_locked):
            
            for cond in ['correct', 'error', 'all_trials']:
            
                # Loop through IVs                
                for i,iv in enumerate(ivs):
                
                    # loop subjects
                    for s,subj in enumerate(self.subjects):
                        SBEHAV = DF[DF['subject']==subj].reset_index()
                        SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f'))
                        SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
                        #############################
                        # DROP THE LAST 200 trials from evoked DF
                        SPUPIL = SPUPIL.iloc[:200,:]
                    
                        # merge behavioral and evoked dataframes so we can group by conditions
                        SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
                    
                        #### DROP OMISSIONS HERE ####
                        SDATA = SDATA[SDATA['exclude'] == 0] # drop excluded
                        #############################
                    
                        evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    
                        save_timepoint_r = []
                    
                        # loop timepoints, regress
                        for col in evoked_cols:
                            Y = SDATA[col] # pupil
                            X = SDATA[iv] # iv

                            if cond == 'correct':
                                mask = SDATA['correct']==True
                                Y = Y[mask] # pupil 
                                X = X[mask] # IV
                            elif cond == 'error':
                                mask = SDATA['correct']==False
                                Y = Y[mask] # pupil 
                                X = X[mask] # IV

                            r, pval = sp.stats.pearsonr(np.array(X), np.array(Y))

                            save_timepoint_r.append(self.fisher_transform(r))
                            
                        # add column for each subject with timepoints as rows
                        df_out[subj] = np.array(save_timepoint_r)
                        # df_out[subj] = df_out[subj].apply(lambda x: '%.16f' % x) # remove scientific notation from df
                    
                    # save output file
                    df_out.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_{}_{}.csv'.format(self.exp, time_locked, cond, iv)), float_format='%.16f')
        print('success: dataframe_evoked_regression')
        
        
    def plot_pupil_information_regression_evoked(self):
        """Plot correlation between pupil response and model estimates.
        
        Notes
        -----
        Always target_locked pupil response.
        Correlations are done for all trials as well as for correct and error trials separately.
        """
        ylim_feed = [-0.2, 0.2]
        tick_spacer = 0.1
        
        ivs = ['model_i', 'model_H', 'model_D']
    
        # xticklabels = ['mean response']
        colors = ['teal', 'orange', 'purple']
        alphas = [1]
        
        #######################
        # FEEDBACK PLOT BETAS FOR EACH MODEL DV
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
                
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        for i,iv in enumerate(ivs):
            # Compute means, sems across group
            COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_{}_{}.csv'.format(self.exp, time_locked, 'all_trials', iv)), float_precision='%.16f')
            COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns

            # plot time series
            TS = np.array(COND.T) # flip so subjects are rows
            self.tsplot(ax, TS, color=colors[i], label=iv)
            self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1+i, color=colors[i], ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
    
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            
        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('r')
        ax.set_title(time_locked)
        ax.legend()
        
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked_correlation.pdf'.format(self.exp)))
        
        #######################
        # Model IVs split by Error and Correct
        #######################
        for iv in ['model_i', 'model_H', 'model_D']:
        
            fig = plt.figure(figsize=(4,2))
            ax = fig.add_subplot(111)
            t = 0
            time_locked = 'target_locked'
            xticklabels = ['Error', 'Correct']
            colorsts = ['red', 'blue']
            alpha_fills = [0.2,0.2] # fill
            alpha_lines = [1, 1]
        
            kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
            # determine time points x-axis given sample rate
            event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
            end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
            mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
        
            save_conds = []
            # plot time series
            for i, cond in enumerate(['error', 'correct']):
            
                # Compute means, sems across group
                COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_{}_{}.csv'.format(self.exp, time_locked, cond, iv)), float_precision='%.16f')
                COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns

                TS = np.array(COND.T)
                self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
                save_conds.append(TS) # for stats
                # single condition against 0
                self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1+i, color=colorsts[i], ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
            
            # test difference
            self.cluster_sig_bar_1samp(array=np.subtract(save_conds[1], save_conds[0]), x=pd.Series(range(TS.shape[-1])), yloc=1, color='purple', ax=ax, threshold=0.05, nrand=5000, cluster_correct=False)
            self.cluster_sig_bar_1samp(array=np.subtract(save_conds[1], save_conds[0]), x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
        
            # set figure parameters
            ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
            # Shade all time windows of interest in grey, will be different for events
            for twi in self.pupil_time_of_interest[t]:       
                tw_begin = int(event_onset + (twi[0]*self.sample_rate))
                tw_end = int(event_onset + (twi[1]*self.sample_rate))
                ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            
            xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
            ax.set_xticks(xticks)
            ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
            ax.set_xlabel('Time from feedback (s)')
            ax.set_ylabel('r')
            ax.set_title(iv)
            ax.legend()
            # whole figure format
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_evoked_correlation_{}.pdf'.format(self.exp, iv)))
        print('success: plot_pupil_information_regression_evoked')


    def dataframe_evoked_pupil_higher_raw_bp(self):
        """Compute evoked pupil responses for the RAW BP pupil time series.
        
        Notes
        -----
        Split by conditions of interest. Save as higher level dataframe per condition of interest. 
        Evoked dataframes need to be combined with behavioral data frame, looping through subjects. 
        DROP PHASE 2 trials.
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        csv_names = deepcopy(['subject','correct','mapping1','correct-mapping1'])
        factors = [['subject'],['correct'],['mapping1'],['correct','mapping1']]
        
        for t,time_locked in enumerate(self.time_locked):
            # Loop through conditions                
            for c,cond in enumerate(csv_names):
                # intialize dataframe per condition

                COND = pd.DataFrame()
                g_idx = deepcopy(factors)[c]       # need to add subject idx for groupby()
                
                if not cond == 'subject':
                    g_idx.insert(0, 'subject') # get strings not list element
                
                for s,subj in enumerate(self.subjects):
                    SBEHAV = DF[DF['subject']==subj].reset_index()
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_raw_bp_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f'))
                    SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
                    #############################
                    # DROP THE LAST 200 trials from evoked DF
                    SPUPIL = SPUPIL.iloc[:200,:]
                    
                    # merge behavioral and evoked dataframes so we can group by conditions
                    SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
                    
                    #### DROP OMISSIONS HERE ####
                    SDATA = SDATA[SDATA['exclude'] == 0] # drop outliers based on RT
                    #############################
                    
                    evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    df = SDATA.groupby(g_idx)[evoked_cols].mean() # only get kernels out
                    df = pd.DataFrame(df).reset_index()
                    # add to condition dataframe
                    COND = pd.concat([COND,df],join='outer',axis=0) # can also do: this_cond = this_cond.append()  
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_raw_bp_evoked_{}.csv'.format(self.exp,time_locked,cond)), float_format='%.16f')
        print('success: dataframe_evoked_pupil_higher RAW BP')


    def dataframe_evoked_pupil_higher_interp_bp(self):
        """Compute evoked pupil responses for the INTERPOLATE BP pupil time series.
        
        Notes
        -----
        Split by conditions of interest. Save as higher level dataframe per condition of interest. 
        Evoked dataframes need to be combined with behavioral data frame, looping through subjects. 
        DROP PHASE 2 trials.
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        csv_names = deepcopy(['subject','correct','mapping1','correct-mapping1'])
        factors = [['subject'],['correct'],['mapping1'],['correct','mapping1']]
        
        for t,time_locked in enumerate(self.time_locked):
            # Loop through conditions                
            for c,cond in enumerate(csv_names):
                # intialize dataframe per condition

                COND = pd.DataFrame()
                g_idx = deepcopy(factors)[c]       # need to add subject idx for groupby()
                
                if not cond == 'subject':
                    g_idx.insert(0, 'subject') # get strings not list element
                
                for s,subj in enumerate(self.subjects):
                    SBEHAV = DF[DF['subject']==subj].reset_index()
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_interp_bp_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f'))
                    SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
                    #############################
                    # DROP THE LAST 200 trials from evoked DF
                    SPUPIL = SPUPIL.iloc[:200,:]
                    
                    # merge behavioral and evoked dataframes so we can group by conditions
                    SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
                    
                    #### DROP OMISSIONS HERE ####
                    SDATA = SDATA[SDATA['exclude'] == 0] # drop outliers based on RT
                    #############################
                    
                    evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    df = SDATA.groupby(g_idx)[evoked_cols].mean() # only get kernels out
                    df = pd.DataFrame(df).reset_index()
                    # add to condition dataframe
                    COND = pd.concat([COND,df],join='outer',axis=0) # can also do: this_cond = this_cond.append()  
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_interp_bp_evoked_{}.csv'.format(self.exp,time_locked,cond)), float_format='%.16f')
        print('success: dataframe_evoked_pupil_higher INTERP BP')
    
    
    def dataframe_evoked_pupil_higher_nuisance(self):
        """Compute evoked pupil responses for the NUISANCE pupil time series.
        
        Notes
        -----
        Split by conditions of interest. Save as higher level dataframe per condition of interest. 
        Evoked dataframes need to be combined with behavioral data frame, looping through subjects. 
        DROP PHASE 2 trials.
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        csv_names = deepcopy(['subject','correct','mapping1','correct-mapping1'])
        factors = [['subject'],['correct'],['mapping1'],['correct','mapping1']]
        
        for t,time_locked in enumerate(self.time_locked):
            # Loop through conditions                
            for c,cond in enumerate(csv_names):
                # intialize dataframe per condition

                COND = pd.DataFrame()
                g_idx = deepcopy(factors)[c]       # need to add subject idx for groupby()
                
                if not cond == 'subject':
                    g_idx.insert(0, 'subject') # get strings not list element
                
                for s,subj in enumerate(self.subjects):
                    SBEHAV = DF[DF['subject']==subj].reset_index()
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'{}_{}_recording-eyetracking_physio_{}_nuisance_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f'))
                    SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
                    #############################
                    # DROP THE LAST 200 trials from evoked DF
                    SPUPIL = SPUPIL.iloc[:200,:]
                    
                    # merge behavioral and evoked dataframes so we can group by conditions
                    SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
                    
                    #### DROP OMISSIONS HERE ####
                    SDATA = SDATA[SDATA['exclude'] == 0] # drop outliers based on RT
                    #############################
                    
                    evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    df = SDATA.groupby(g_idx)[evoked_cols].mean() # only get kernels out
                    df = pd.DataFrame(df).reset_index()
                    # add to condition dataframe
                    COND = pd.concat([COND,df],join='outer',axis=0) # can also do: this_cond = this_cond.append()  
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_nuisance_evoked_{}.csv'.format(self.exp,time_locked,cond)), float_format='%.16f')
        print('success: dataframe_evoked_pupil_higher NUISANCE')
        

    def plot_evoked_pupil_raw_bp(self):
        """Plot evoked pupil time courses.
        
        Notes
        -----
        4 figures: mean response, accuracy, frequency, accuracy*frequency.
        Always target_locked pupil response.
        """
        from matplotlib.ticker import FixedLocator
        
        ylim_feed = [-2.5,2.5]
        tick_spacer = 2.5
        
        #######################
        # FEEDBACK MEAN RESPONSE
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
                
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_raw_bp_evoked_{}.csv'.format(self.exp,time_locked,factor)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
    
        xticklabels = ['mean response']
        colors = ['black'] # black
        alphas = [1]
        
        # plot time series
        i=0
        TS = np.array(COND.iloc[:,-kernel:]) # index from back to avoid extra unnamed column pandas
        self.tsplot(ax, TS, color='k', label=xticklabels[i])
        self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
    
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            
        # shade baseline pupil
        twb = [-self.baseline_window, 0]
        baseline_onset = int(abs(twb[0]*self.sample_rate))
        twb_begin = int(baseline_onset + (twb[0]*self.sample_rate))
        twb_end = int(baseline_onset + (twb[1]*self.sample_rate))
        ax.axvspan(twb_begin,twb_end, facecolor='k', alpha=0.1)
        
        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        
        # ax.set_ylim(ylim_feed)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
                
        # compute peak of mean response to center time window around
        m = np.mean(TS,axis=0)
        argm = np.true_divide(np.argmax(m),self.sample_rate) + self.pupil_step_lim[t][0] # subtract pupil baseline to get timing
        print('mean response = {} peak @ {} seconds'.format(np.max(m),argm))
        # ax.axvline(np.argmax(m), lw=0.25, alpha=0.5, color = 'k')
                
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_raw_bp_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # CORRECT
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'correct'
        factor = 'correct'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_raw_bp_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['Error','Correct']
        colorsts = ['r','b',]
        alpha_fills = [0.2,0.2] # fill
        alpha_lines = [1,1]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_difference = save_conds[0]-save_conds[1]
        self.cluster_sig_bar_1samp(array=pe_difference, x=pd.Series(range(pe_difference.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_raw_bp_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # FREQUENCY
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'mapping1'
        factor = 'mapping1'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_raw_bp_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['20%','80%']
        colorsts = ['indigo','indigo',]
        alpha_fills = [0.2,0.2] # fill
        alpha_lines = [0.2,1]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_difference = save_conds[0]-save_conds[1]
        self.cluster_sig_bar_1samp(array=pe_difference, x=pd.Series(range(pe_difference.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim_feed)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_raw_bp_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # CORRECT x MAPPING1
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'correct-mapping1'
        factor = ['correct','mapping1']
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_raw_bp_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        ########
        # make unique labels for each of the 4 conditions
        conditions = [
            (COND['correct'] == 0) & (COND['mapping1'] == 1), # Easy Error 1
            (COND['correct'] == 1) & (COND['mapping1'] == 1), # Easy Correct 2
            (COND['correct'] == 0) & (COND['mapping1'] == 0), # Hard Error 3
            (COND['correct'] == 1) & (COND['mapping1'] == 0), # Hard Correct 4
            ]
        values = [1,2,3,4]
        conditions = np.select(conditions, values) # don't add as column to time series otherwise it gets plotted
        ########
                    
        xticklabels = ['Error 80%','Correct 80%','Error 20%','Correct 20%']
        colorsts = ['r','b','r','b']
        alpha_fills = [0.2,0.2,0.1,0.1] # fill
        alpha_lines = [1,1,.8,.8]
        linestyle= ['solid','solid','dashed','dashed']
        save_conds = []
        # plot time series
        
        for i,x in enumerate(values):
            TS = COND[conditions==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, linestyle=linestyle[i], color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_interaction = (save_conds[0]-save_conds[1]) - (save_conds[2]-save_conds[3])
        self.cluster_sig_bar_1samp(array=pe_interaction, x=pd.Series(range(pe_interaction.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim_feed)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        ax.legend(loc='best')
                
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_raw_bp_evoked_{}.pdf'.format(self.exp, csv_name)))
        print('success: plot_evoked_pupil RAW BP')
        

    def plot_evoked_pupil_interp_bp(self):
        """Plot evoked pupil time courses: INTERPOLATE and BP time series
        
        Notes
        -----
        4 figures: mean response, accuracy, frequency, accuracy*frequency.
        Always target_locked pupil response.
        """
        ylim_feed = [-2.5,2.5]
        tick_spacer = 2.5
        
        #######################
        # FEEDBACK MEAN RESPONSE
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
                
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_interp_bp_evoked_{}.csv'.format(self.exp,time_locked,factor)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
    
        xticklabels = ['mean response']
        colors = ['black'] # black
        alphas = [1]

        # plot time series
        i=0
        TS = np.array(COND.iloc[:,-kernel:]) # index from back to avoid extra unnamed column pandas
        self.tsplot(ax, TS, color='k', label=xticklabels[i])
        self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
    
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            
        # shade baseline pupil
        twb = [-self.baseline_window, 0]
        baseline_onset = int(abs(twb[0]*self.sample_rate))
        twb_begin = int(baseline_onset + (twb[0]*self.sample_rate))
        twb_end = int(baseline_onset + (twb[1]*self.sample_rate))
        ax.axvspan(twb_begin,twb_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        
        # ax.set_ylim(ylim_feed)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
                
        # compute peak of mean response to center time window around
        m = np.mean(TS,axis=0)
        argm = np.true_divide(np.argmax(m),self.sample_rate) + self.pupil_step_lim[t][0] # subtract pupil baseline to get timing
        print('mean response = {} peak @ {} seconds'.format(np.max(m),argm))
        # ax.axvline(np.argmax(m), lw=0.25, alpha=0.5, color = 'k')
        
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_interp_bp_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # CORRECT
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'correct'
        factor = 'correct'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_interp_bp_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['Error','Correct']
        colorsts = ['r','b',]
        alpha_fills = [0.2,0.2] # fill
        alpha_lines = [1,1]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_difference = save_conds[0]-save_conds[1]
        self.cluster_sig_bar_1samp(array=pe_difference, x=pd.Series(range(pe_difference.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_interp_bp_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # FREQUENCY
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'mapping1'
        factor = 'mapping1'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_interp_bp_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['20%','80%']
        colorsts = ['indigo','indigo',]
        alpha_fills = [0.2,0.2] # fill
        alpha_lines = [0.2,1]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_difference = save_conds[0]-save_conds[1]
        self.cluster_sig_bar_1samp(array=pe_difference, x=pd.Series(range(pe_difference.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim_feed)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_interp_bp_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # CORRECT x MAPPING1
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'correct-mapping1'
        factor = ['correct','mapping1']
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_interp_bp_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        ########
        # make unique labels for each of the 4 conditions
        conditions = [
            (COND['correct'] == 0) & (COND['mapping1'] == 1), # Easy Error 1
            (COND['correct'] == 1) & (COND['mapping1'] == 1), # Easy Correct 2
            (COND['correct'] == 0) & (COND['mapping1'] == 0), # Hard Error 3
            (COND['correct'] == 1) & (COND['mapping1'] == 0), # Hard Correct 4
            ]
        values = [1,2,3,4]
        conditions = np.select(conditions, values) # don't add as column to time series otherwise it gets plotted
        ########
                    
        xticklabels = ['Error 80%','Correct 80%','Error 20%','Correct 20%']
        colorsts = ['r','b','r','b']
        alpha_fills = [0.2,0.2,0.1,0.1] # fill
        alpha_lines = [1,1,.8,.8]
        linestyle= ['solid','solid','dashed','dashed']
        save_conds = []
        # plot time series
        
        for i,x in enumerate(values):
            TS = COND[conditions==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, linestyle=linestyle[i], color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_interaction = (save_conds[0]-save_conds[1]) - (save_conds[2]-save_conds[3])
        self.cluster_sig_bar_1samp(array=pe_interaction, x=pd.Series(range(pe_interaction.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim_feed)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        ax.legend(loc='best')
                
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_interp_bp_evoked_{}.pdf'.format(self.exp, csv_name)))
        print('success: plot_evoked_pupil_interp_bp')


    def plot_evoked_pupil_nuisance(self):
        """Plot evoked pupil time courses: NUISANCE time series
        
        Notes
        -----
        4 figures: mean response, accuracy, frequency, accuracy*frequency.
        Always target_locked pupil response.
        """
        ylim_feed = [-2.5,2.5]
        tick_spacer = 2.5
        
        #######################
        # FEEDBACK MEAN RESPONSE
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
                
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_nuisance_evoked_{}.csv'.format(self.exp,time_locked,factor)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
    
        xticklabels = ['mean response']
        colors = ['black'] # black
        alphas = [1]

        # plot time series
        i=0
        TS = np.array(COND.iloc[:,-kernel:]) # index from back to avoid extra unnamed column pandas
        self.tsplot(ax, TS, color='k', label=xticklabels[i])
        self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
    
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            
        # shade baseline pupil
        twb = [-self.baseline_window, 0]
        baseline_onset = int(abs(twb[0]*self.sample_rate))
        twb_begin = int(baseline_onset + (twb[0]*self.sample_rate))
        twb_end = int(baseline_onset + (twb[1]*self.sample_rate))
        ax.axvspan(twb_begin,twb_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        
        # ax.set_ylim(ylim_feed)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
                
        # compute peak of mean response to center time window around
        m = np.mean(TS,axis=0)
        argm = np.true_divide(np.argmax(m),self.sample_rate) + self.pupil_step_lim[t][0] # subtract pupil baseline to get timing
        print('mean response = {} peak @ {} seconds'.format(np.max(m),argm))
        # ax.axvline(np.argmax(m), lw=0.25, alpha=0.5, color = 'k')
        
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_nuisance_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # CORRECT
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'correct'
        factor = 'correct'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_nuisance_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['Error','Correct']
        colorsts = ['r','b',]
        alpha_fills = [0.2,0.2] # fill
        alpha_lines = [1,1]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_difference = save_conds[0]-save_conds[1]
        self.cluster_sig_bar_1samp(array=pe_difference, x=pd.Series(range(pe_difference.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_nuisance_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # FREQUENCY
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'mapping1'
        factor = 'mapping1'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_nuisance_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['20%','80%']
        colorsts = ['indigo','indigo',]
        alpha_fills = [0.2,0.2] # fill
        alpha_lines = [0.2,1]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_difference = save_conds[0]-save_conds[1]
        self.cluster_sig_bar_1samp(array=pe_difference, x=pd.Series(range(pe_difference.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim_feed)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_nuisance_evoked_{}.pdf'.format(self.exp, factor)))
        
        #######################
        # CORRECT x MAPPING1
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'target_locked'
        csv_name = 'correct-mapping1'
        factor = ['correct','mapping1']
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_nuisance_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        ########
        # make unique labels for each of the 4 conditions
        conditions = [
            (COND['correct'] == 0) & (COND['mapping1'] == 1), # Easy Error 1
            (COND['correct'] == 1) & (COND['mapping1'] == 1), # Easy Correct 2
            (COND['correct'] == 0) & (COND['mapping1'] == 0), # Hard Error 3
            (COND['correct'] == 1) & (COND['mapping1'] == 0), # Hard Correct 4
            ]
        values = [1,2,3,4]
        conditions = np.select(conditions, values) # don't add as column to time series otherwise it gets plotted
        ########
                    
        xticklabels = ['Error 80%','Correct 80%','Error 20%','Correct 20%']
        colorsts = ['r','b','r','b']
        alpha_fills = [0.2,0.2,0.1,0.1] # fill
        alpha_lines = [1,1,.8,.8]
        linestyle= ['solid','solid','dashed','dashed']
        save_conds = []
        # plot time series
        
        for i,x in enumerate(values):
            TS = COND[conditions==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, linestyle=linestyle[i], color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_interaction = (save_conds[0]-save_conds[1]) - (save_conds[2]-save_conds[3])
        self.cluster_sig_bar_1samp(array=pe_interaction, x=pd.Series(range(pe_interaction.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(250*1), event_onset+(250*2), event_onset+(250*3), event_onset+(250*4), event_onset+(250*5), event_onset+(250*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim_feed)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        ax.legend(loc='best')
                
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_nuisance_evoked_{}.pdf'.format(self.exp, csv_name)))
        print('success: plot_evoked_pupil_nuisance')
        
    
    def compute_phasics_interp_bp(self,):
        """Compute the correlation between the 'clean' and 'unclean' post-feedback pupil response in the time windows of interest.
        """
        
        for s,subj in enumerate(self.subjects):
            # save as new file
            out_log = os.path.join(self.project_directory,subj, 'beh', '{}_{}_interp_bp_phasics.csv'.format(subj,self.exp)) # derivatives folder
            
            this_log = os.path.join(self.project_directory,subj, 'beh', '{}_{}_beh.csv'.format(subj,self.exp)) # derivatives folder
            B = pd.read_csv(this_log, float_precision = '%.16f') # behavioral file

            ### DROP EXISTING PHASICS COLUMNS TO PREVENT OLD DATA
            try: 
                B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                # B = B.loc[:, ~B.columns.str.contains('_locked')] # remove all old phasic pupil columns
            except:
                pass
                
            # loop through each type of event to lock events to...
            for t,time_locked in enumerate(self.time_locked):
                
                pupil_step_lim = self.pupil_step_lim[t] # kernel size is always the same for each event type
                
                for twi,pupil_time_of_interest in enumerate(self.pupil_time_of_interest[t]): # multiple time windows to average
                    # load evoked pupil file (all trials)
                    P = pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_interp_bp_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f') 
                    P = P.loc[:, ~P.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    P = np.array(P)
                
                    SAVE_TRIALS = []
                    for trial in np.arange(len(P)):
                        # in seconds
                        phase_start = -pupil_step_lim[0] + pupil_time_of_interest[0]
                        phase_end = -pupil_step_lim[0] + pupil_time_of_interest[1]
                        # in sample rate units
                        phase_start = int(phase_start*self.sample_rate)
                        phase_end = int(phase_end*self.sample_rate)
                        # mean within phasic time window
                        this_phasic = np.nanmean(P[trial,phase_start:phase_end]) 
                        SAVE_TRIALS.append(this_phasic)
                    # save phasics
                    B['pupil_{}_t{}_interp_bp'.format(time_locked,twi+1)] = np.array(SAVE_TRIALS)
                                        
                    #######################
                    B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    B.to_csv(out_log, float_format='%.16f')
                    print('subject {}, {} phasic pupil extracted {}'.format(subj, time_locked, pupil_time_of_interest))
        print('success: compute_phasics_interp_bp')
        
    
    def correlation_interp_clean(self,):
        """Compute the correlation between the 'clean' and 'unclean' post-feedback pupil response in the time windows of interest.
        """
        
        df_out = pd.DataFrame()
        # loop through each type of event to lock events to...
        for t,time_locked in enumerate(self.time_locked):
            
            for twi,pupil_time_of_interest in enumerate(self.pupil_time_of_interest[t]): # multiple time windows to average
            
                save_coeffs = []
                for s,subj in enumerate(self.subjects):
                    this_log = os.path.join(self.project_directory,subj, 'beh', '{}_{}_interp_bp_phasics.csv'.format(subj,self.exp)) # derivatives folder
                    B = pd.read_csv(this_log, float_precision = '%.16f') # behavioral file
                
                    x = B['pupil_{}_t{}_interp_bp'.format(time_locked,twi+1)]
                    y = B['pupil_{}_t{}'.format(time_locked,twi+1)]
                    x = np.array(x)
                    y = np.array(y)
                    
                    mask = ~np.isnan(x) & ~np.isnan(y)

                    # Apply the mask
                    x_clean = x[mask]
                    y_clean = y[mask]
                
                    r, pvalue = stats.pearsonr(x_clean, y_clean)

                
                    save_coeffs.append(r)
                df_out['r_pupil_{}_t{}'.format(time_locked,twi+1)] = save_coeffs
                
        df_out.to_csv(os.path.join(self.jasp_folder,'{}_preprocessing_sanity_correlation_clean_interp.csv'.format(self.exp)), float_format='%.16f') # for stats
        print('success: correlation_interp_clean')
    
    
    def group_r2_deconvolution(self,):
        """Average r2 from preprocessing deconvolution at the group level.
        
        Notes
        -----
        Overwrites original log file (this_log).
        """
        r2_all = []
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory,subj, 'beh', '{}_{}_recording-eyetracking_physio_r2_deconvolution.csv'.format(subj, self.exp)) # derivatives folder
            B = pd.read_csv(this_log, float_precision='%.16f') # behavioral file
            B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
            r2_all.append(B['r2_deconvolution'])
        
        df_out = pd.DataFrame(r2_all)        
        df_out.to_csv(os.path.join(self.jasp_folder,'{}_preprocessing_sanity_r2_deconvolution.csv'.format(self.exp)), float_format='%.16f') # for stats
        print('success: group_r2_deconvolution')
        
        
        