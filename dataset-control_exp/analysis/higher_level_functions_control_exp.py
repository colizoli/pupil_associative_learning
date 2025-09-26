#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Pupil dilation offers a time-window in prediction error

Control Experiment - Higher Level Functions
Python code O.Colizoli 2023 (olympia.colizoli@donders.ru.nl)
Python 3.6

Notes
-----
>>> conda install -c conda-forge/label/gcc7 mne
================================================
"""

import os, sys, datetime
import numpy as np
import scipy as sp
from scipy.signal import decimate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
import re
from copy import deepcopy
import itertools
from IPython import embed as shell # for debugging only


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
    colors : list
        List of colors to use as conditions of interest

    Attributes
    ----------
    subjects : list
        List of subject numbers
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
    colors : list
        List of colors to use as conditions of interest
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
    
    def __init__(self, subjects, experiment_name, project_directory, sample_rate, time_locked, pupil_step_lim, baseline_window, pupil_time_of_interest, colors):        
        """Constructor method
        """
        self.subjects           = subjects
        self.exp                = experiment_name
        self.project_directory  = project_directory
        self.figure_folder      = os.path.join(project_directory, 'figures')
        self.dataframe_folder   = os.path.join(project_directory, 'data_frames')
        self.trial_bin_folder   = os.path.join(self.dataframe_folder,'trial_bins_pupil') # for average pupil in different trial bin windows
        self.jasp_folder        = os.path.join(self.dataframe_folder,'jasp') # for dataframes to input into JASP
        self.colors             = colors # determines how to group the conditions
        ##############################    
        # Pupil time series information:
        ##############################
        self.sample_rate        = sample_rate
        self.time_locked        = time_locked
        self.pupil_step_lim     = pupil_step_lim                
        self.baseline_window    = baseline_window              
        self.pupil_time_of_interest = pupil_time_of_interest
        
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
            
        if not os.path.isdir(self.dataframe_folder):
            os.mkdir(self.dataframe_folder)
        
        if not os.path.isdir(self.trial_bin_folder):
            os.mkdir(self.trial_bin_folder)
            
        if not os.path.isdir(self.jasp_folder):
            os.mkdir(self.jasp_folder)
        
        
    def tsplot(self, ax, data, alpha_fill=0.2, alpha_line=1, **kw):
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
        # shell()
        fill_color = kw['color']
        ax.fill_between(x, est-sde, est+sde, alpha=alpha_fill, color=fill_color, linewidth=0.0) # debug double label!
        
        ax.plot(x, est, alpha=alpha_line, **kw)
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


    def cluster_sig_bar_1samp(self, array, x, yloc, color, ax, threshold=0.05, nrand=5000, cluster_correct=True):
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
                
                            
    def higherlevel_get_phasics(self,):
        """Computes phasic pupil (evoked average) in selected time window per trial and add phasics to behavioral data frame. 
        
        Notes
        -----
        Overwrites original log file (this_log).
        """
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory,subj, 'beh', '{}_{}_beh.csv'.format(subj, self.exp)) # derivatives folder
            B = pd.read_csv(this_log) # behavioral file
            ### DROP EXISTING PHASICS COLUMNS TO PREVENT OLD DATA
            try: 
                B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                B = B.loc[:, ~B.columns.str.contains('_locked')] # remove all old phasic pupil columns
            except:
                pass
                
            # loop through each type of event to lock events to...
            for t, time_locked in enumerate(self.time_locked):
                
                pupil_step_lim = self.pupil_step_lim[t] # kernel size is always the same for each event type
                
                for twi, pupil_time_of_interest in enumerate(self.pupil_time_of_interest[t]): # multiple time windows to average
                
                    # load evoked pupil file (all trials)
                    P = pd.read_csv(os.path.join(self.project_directory,subj, 'beh', '{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked))) 
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
                    B['pupil_{}_t{}'.format(time_locked,twi+1)] = np.array(SAVE_TRIALS)

                    #######################
                    B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    B.to_csv(this_log)
                    print('subject {}, {} phasic pupil extracted {}'.format(subj,time_locked, pupil_time_of_interest))
        print('success: higherlevel_get_phasics')


    def create_subjects_dataframe(self, ):
        """Combine behavior and phasic pupil dataframes including pupil baselines of all subjects into a single large dataframe. 
        
        Notes
        -----
        Flag outliers based on RT (separate column) per subject. 
        Output in dataframe folder: task-predictions_subjects.csv
        """
        DF = pd.DataFrame()
        
        # loop through subjects, get behavioral log files
        for s,subj in enumerate(self.subjects):
            
            this_data = pd.read_csv(os.path.join(self.project_directory, subj, 'beh', '{}_{}_beh.csv'.format(subj,self.exp)))
            this_data = this_data.loc[:, ~this_data.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            # open baseline pupil to add to dataframes as well
            this_baseline = pd.read_csv(os.path.join(self.project_directory, subj, 'beh', '{}_{}_recording-eyetracking_physio_{}_baselines.csv'.format(subj, self.exp, 'stim_locked')))
            this_baseline = this_baseline.loc[:, ~this_baseline.columns.str.contains('^Unnamed')] # remove all unnamed columns
            this_data['pupil_baseline_stim_locked'] = np.array(this_baseline)
                        
            ###############################            
            # concatenate all subjects
            DF = pd.concat([DF,this_data],axis=0)

        #####################
        # save whole dataframe with all subjects
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        #####################
        print('success: create_subjects_dataframe')
        
    
    def get_stimulus_durations(self, ):
        """Get the color stimulus duration per participant and compute group mean.
        
        Notes
        -----
        Output in dataframe folder
        """
        durations = []
        
        # loop through subjects, get behavioral log files
        for s,subj in enumerate(self.subjects):
            
            subj_num = re.findall(r'\d+', subj)[0]
            this_data = pd.read_csv(os.path.join(self.project_directory, subj, 'beh', 'sub-{}_task-decision_meanRT.csv'.format(int(subj_num)-100)))
            this_data = this_data.loc[:, ~this_data.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            ###############################            
            # concatenate all subjects
            durations.append(this_data.meanRT[0])
        
        print('Across participants Mean stimulus duration = {} s, SD = {}'.format(np.mean(durations), np.std(durations)))
        #####################
        # save whole dataframe with all subjects
        DF = pd.DataFrame()
        DF['subject'] = self.subjects
        DF['stim_duration'] = durations
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_stimulus_durations.csv'.format(self.exp)))
        #####################
        print('success: get_stimulus_durations')
        
        
    def average_conditions_colors(self, ):
        """Average the DVs per subject per condition of interest. 

        Notes
        -----
        self.colors argument determines how the trials were split.
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject','trial_num'],inplace=True)
        DF.reset_index()
        
        ############################ 
        # drop outliers and missing trials
        DF = DF[DF['RT']!=np.nan]
        ############################
        
        '''
        ######## COLORS x TIME WINDOW ########
        '''
        DFOUT = DF.groupby(['subject', 'r']).aggregate({'pupil_stim_locked_t1':'mean', 'pupil_stim_locked_t2':'mean'})
        # save for RMANOVA format
        DFANOVA =  DFOUT.unstack(['r']) 
        print(DFANOVA.columns)
        DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
        DFANOVA.to_csv(os.path.join(self.jasp_folder, '{}_colors-timewindow_rmanova.csv'.format(self.exp))) # for stats
        
        '''
        ######## COLORS ########
        '''
        for pupil_dv in ['pupil_stim_locked_t1', 'pupil_stim_locked_t2', 'pupil_baseline_stim_locked']: # mean accuracy
            DFOUT = DF.groupby(['subject', 'r'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder, '{}_colors_{}.csv'.format(self.exp, pupil_dv))) # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['r']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder, '{}_colors_{}_rmanova.csv'.format(self.exp, pupil_dv))) # for stats
        print('success: average_conditions')
    
    
    def dataframe_evoked_pupil_higher_colors(self):
        """Compute evoked pupil responses.
        
        Notes
        -----
        Split by conditions of interest. Save as higher level dataframe per condition of interest. 
        Evoked dataframes need to be combined with behavioral data frame, looping through subjects. 
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder, '{}_subjects.csv'.format(self.exp)))
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        csv_names = deepcopy(['subject', 'colors'])
        factors = [['subject'],['r']]
        
        for t,time_locked in enumerate(self.time_locked):
            # Loop through conditions                
            for c,cond in enumerate(csv_names):
                # intialize dataframe per condition
                COND = pd.DataFrame()
                g_idx = deepcopy(factors)[c]       # need to add subject idx for groupby()
                
                if not cond == 'subject':
                    g_idx.insert(0, 'subject') # get strings not list element
                
                for s,subj in enumerate(self.subjects):
                    subj_num = re.findall(r'\d+', subj)[0]
                    SBEHAV = DF[DF['subject']==int(subj_num)].reset_index() # not 'sub-' in DF
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked))))
                    SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
                    # merge behavioral and evoked dataframes so we can group by conditions
                    SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
                    
                    #### DROP OMISSIONS HERE ####
                    SDATA = SDATA[SDATA['RT'] != np.nan] # drop outliers based on RT
                    #############################
                    
                    evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    df = SDATA.groupby(g_idx)[evoked_cols].mean() # only get kernels out
                    df = pd.DataFrame(df).reset_index()
                    # add to condition dataframe
                    COND = pd.concat([COND,df],join='outer',axis=0) # can also do: this_cond = this_cond.append()  
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,cond)))
        print('success: dataframe_evoked_pupil_higher_colors')


    def plot_evoked_pupil_higher_colors(self):
        """Plot evoked pupil time courses.
        
        Notes
        -----
        2 figures. 
        Plot the group level mean and then split per color.
        """
        ylim = [-4,15]
        tick_spacer = 3
        
        #######################
        # MEAN RESPONSE
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'stim_locked'
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,factor)))
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
            
        # Shade average color stimulus duration (0.7 s)
        fb_begin = int(event_onset)
        fb_end = int(event_onset + (0.7*self.sample_rate))
        ax.axvspan(fb_begin,fb_end, facecolor='k', alpha=0.3)
        
        # shade baseline pupil
        # twb = [-self.baseline_window, 0]
        # baseline_onset = int(abs(twb[0]*self.sample_rate))
        # twb_begin = int(baseline_onset + (twb[0]*self.sample_rate))
        # twb_end = int(baseline_onset + (twb[1]*self.sample_rate))
        # ax.axvspan(twb_begin,twb_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(500*1), event_onset+(500*2), event_onset+(500*3), event_onset+(500*4), event_onset+(500*5), event_onset+(500*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from stimulus (s)')
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
        # COLORS
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'stim_locked'
        csv_name = 'colors'
        factor = 'r'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=self.colors[i], label=self.colors[i], alpha_fill=0.2, alpha_line=1)
            # stats        
            self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=i+1, color=self.colors[i], ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
        
        # Shade average color stimulus duration (0.7 s)
        fb_begin = int(event_onset)
        fb_end = int(event_onset + (0.7*self.sample_rate))
        ax.axvspan(fb_begin,fb_end, facecolor='k', alpha=0.3)

        xticks = [event_onset, event_onset+(500*1), event_onset+(500*2), event_onset+(500*3), event_onset+(500*4), event_onset+(500*5), event_onset+(500*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from stimulus (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked_{}.pdf'.format(self.exp, csv_name)))
        print('success: plot_evoked_pupil_higher_colors')
    
    
    def average_conditions_sounds(self, ):
        """Average the DVs per subject per condition of interest. 

        Notes
        -----
        Split by tone factor.
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        """       
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject','trial_num'],inplace=True)
        DF.reset_index()
        
        '''
        ######## TONE x TIME WINDOW ########
        '''
        DFOUT = DF.groupby(['subject', 'tone']).aggregate({'pupil_stim_locked_t1':'mean', 'pupil_stim_locked_t2':'mean'})
        # save for RMANOVA format
        DFANOVA =  DFOUT.unstack(['tone']) 
        print(DFANOVA.columns)
        DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
        DFANOVA.to_csv(os.path.join(self.jasp_folder, '{}_tone-timewindow_rmanova.csv'.format(self.exp))) # for stats
        
        '''
        ######## TONE ########
        '''
        for pupil_dv in ['pupil_stim_locked_t1', 'pupil_stim_locked_t2', 'pupil_baseline_stim_locked']: # mean accuracy
            DFOUT = DF.groupby(['subject', 'tone'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder, '{}_tone_{}.csv'.format(self.exp, pupil_dv))) # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['tone']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder, '{}_tone_{}_rmanova.csv'.format(self.exp, pupil_dv))) # for stats
        print('success: average_conditions_sounds')
        
    
    def dataframe_evoked_pupil_higher_sounds(self):
        """Compute evoked pupil responses.
        
        Notes
        -----
        Split by conditions of interest. Save as higher level dataframe per condition of interest. 
        Evoked dataframes need to be combined with behavioral data frame, looping through subjects. 
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder, '{}_subjects.csv'.format(self.exp)))
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        csv_names = deepcopy(['subject', 'tone'])
        factors = [['subject'],['tone']]
        
        for t,time_locked in enumerate(self.time_locked):
            # Loop through conditions                
            for c,cond in enumerate(csv_names):
                # intialize dataframe per condition
                COND = pd.DataFrame()
                g_idx = deepcopy(factors)[c]       # need to add subject idx for groupby()
                
                if not cond == 'subject':
                    g_idx.insert(0, 'subject') # get strings not list element
                
                for s,subj in enumerate(self.subjects):
                    subj_num = re.findall(r'\d+', subj)[0]
                    SBEHAV = DF[DF['subject']==int(subj_num)].reset_index() # not 'sub-' in DF
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked))))
                    SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
                    # merge behavioral and evoked dataframes so we can group by conditions
                    SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
                    
                    evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    df = SDATA.groupby(g_idx)[evoked_cols].mean() # only get kernels out
                    df = pd.DataFrame(df).reset_index()
                    # add to condition dataframe
                    COND = pd.concat([COND,df],join='outer',axis=0) # can also do: this_cond = this_cond.append()  
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp, time_locked, cond)))
        print('success: dataframe_evoked_pupil_higher_colors')
    

    def plot_evoked_pupil_higher_sounds(self):
        """Plot evoked pupil time courses.
        
        Notes
        -----
        2 figures. 
        Plot the group level mean and then split per tone.
        """
        ylim = [-4,15]
        tick_spacer = 3
        
        #######################
        # MEAN RESPONSE
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'stim_locked'
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,factor)))
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
        
        # Shade aduitory feedback duration (0.3 s)
        fb_begin = int(event_onset)
        fb_end = int(event_onset + (0.3*self.sample_rate))
        ax.axvspan(fb_begin,fb_end, facecolor='k', alpha=0.3)
        
        # shade baseline pupil
        # twb = [-self.baseline_window, 0]
        # baseline_onset = int(abs(twb[0]*self.sample_rate))
        # twb_begin = int(baseline_onset + (twb[0]*self.sample_rate))
        # twb_end = int(baseline_onset + (twb[1]*self.sample_rate))
        # ax.axvspan(twb_begin,twb_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(500*1), event_onset+(500*2), event_onset+(500*3), event_onset+(500*4), event_onset+(500*5), event_onset+(500*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from stimulus (s)')
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
        # TONE
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'stim_locked'
        csv_name = 'tone'
        factor = 'tone'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        xticklabels = ['tone_error','tone_correct']
        colorsts = ['r','b']        
        
        save_conds = []
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=0.2, alpha_line=1)
            save_conds.append(TS)
        
        # stats (plot corrected and uncorrected) 
        cond_diff = save_conds[0]-save_conds[1]  
        self.cluster_sig_bar_1samp(array=cond_diff, x=pd.Series(range(cond_diff.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
        self.cluster_sig_bar_1samp(array=cond_diff, x=pd.Series(range(cond_diff.shape[-1])), yloc=1, color='purple', ax=ax, threshold=0.05, nrand=5000, cluster_correct=False)
        
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
        
        # shade duration of auditory stimulus (0.3 s)
        fb_begin = int(event_onset)
        fb_end = int(event_onset + (0.3*self.sample_rate))
        ax.axvspan(fb_begin,fb_end, facecolor='k', alpha=0.3)

        xticks = [event_onset, event_onset+(500*1), event_onset+(500*2), event_onset+(500*3), event_onset+(500*4), event_onset+(500*5), event_onset+(500*6)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim)
        # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from stimulus (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked_{}.pdf'.format(self.exp, csv_name)))
        print('success: plot_evoked_pupil_higher_sounds')
        
        
