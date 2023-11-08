#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Pupil dilation offers a time-window in prediction error

Data set #2 Letter-color 2AFC task - Higher Level Functions
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
import scipy.stats as stats
import statsmodels.formula.api as sm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
import re
from statsmodels.stats.anova import AnovaRM
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
    freq_cond : str
        Frequency condition of interest ('frequency' or 'actual_frequency')

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
    freq_cond : str
        Frequency condition of interest ('frequency' or 'actual_frequency')
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
    
    def __init__(self, subjects, experiment_name, project_directory, sample_rate, time_locked, pupil_step_lim, baseline_window, pupil_time_of_interest, freq_cond):        
        """Constructor method
        """
        self.subjects           = subjects
        self.exp                = experiment_name
        self.project_directory  = project_directory
        self.figure_folder      = os.path.join(project_directory, 'figures')
        self.dataframe_folder   = os.path.join(project_directory, 'data_frames')
        self.trial_bin_folder   = os.path.join(self.dataframe_folder,'trial_bins_pupil') # for average pupil in different trial bin windows
        self.jasp_folder        = os.path.join(self.dataframe_folder,'jasp') # for dataframes to input into JASP
        self.freq_cond          = freq_cond # determines how to group the conditions based on actual frequencies
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


    def timeseries_fdr_correction(self,  xind, color, ax, pvals, alpha=0.05, method='negcorr'):
        """Add False Discovery Rate-based correction bar on time series plot.
        
        Parameters
        ----------
        xind : array
            x indices of plat
        
        color : string
            Color of bar

        ax : matplotlib.axes._subplots.AxesSubplot
            The subplot handle to plot in
        
        pvals : array
            Input for FDR correction
        
        alpha : float
            Alpha value for p-value significance (default 0.05)

        method : 'negcorr' 
            Method for FDR correction (default 'negcorr')
        
        Notes
        -----
        Plot corrected (black) and uncorrected (purple) on timecourse
        https://mne.tools/stable/generated/mne.stats.fdr_correction.html
        """
        # UNCORRECTED
        yloc = 5
        sig_indices = np.array(pvals < alpha, dtype=int)
        yvalues = sig_indices * (((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0])
        yvalues[yvalues == 0] = np.nan # or use np.nan
        ax.plot(xind, yvalues, linestyle='None', marker='.', color='purple', alpha=0.2)
        
        # FDR CORRECTED
        yloc = 8
        reject, pval_corrected = mne.stats.fdr_correction(pvals, alpha=alpha, method=method)
        sig_indices = np.array(reject, dtype=int)
        yvalues = sig_indices * (((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0])
        yvalues[yvalues == 0] = np.nan # or use np.nan
        ax.plot(xind, yvalues, linestyle='None', marker='.', color=color, alpha=1)


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
        
    
    def higherlevel_get_phasics(self,):
        """Computes phasic pupil (evoked average) in selected time window per trial and add phasics to behavioral data frame. 
        
        Notes
        -----
        Overwrites original log file (this_log).
        """
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory,subj,'beh','{}_{}_beh.csv'.format(subj,self.exp)) # derivatives folder
            B = pd.read_csv(this_log) # behavioral file
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
                    P = pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked))) 
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
                    print('subject {}, {} phasic pupil extracted {}'.format(subj,time_locked,pupil_time_of_interest))
        print('success: higherlevel_get_phasics')
        
        
    def create_subjects_dataframe(self,blocks):
        """Combine behavior and phasic pupil dataframes of all subjects into a single large dataframe. 
        
        Notes
        -----
        Flag missing trials from concantenated dataframe.
        Output in dataframe folder: task-experiment_name_subjects.csv
        Merge with actual frequencies
        """
        DF = pd.DataFrame()
        
        # loop through subjects, get behavioral log files
        for s,subj in enumerate(self.subjects):
            
            this_data = pd.read_csv(os.path.join(self.project_directory, subj, 'beh', '{}_{}_beh.csv'.format(subj,self.exp)))
            this_data = this_data.loc[:, ~this_data.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            # open baseline pupil to add to dataframes as well
            this_baseline = pd.read_csv(os.path.join(self.project_directory, subj, 'beh', '{}_{}_recording-eyetracking_physio_{}_baselines.csv'.format(subj, self.exp, 'feed_locked')))
            this_baseline = this_baseline.loc[:, ~this_baseline.columns.str.contains('^Unnamed')] # remove all unnamed columns
            this_data['pupil_baseline_feed_locked'] = np.array(this_baseline)
            
            ###############################
            # flag missing trials
            this_data['missing'] = this_data['button']=='missing'
            this_data['drop_trial'] = np.array(this_data['missing']) #logical or
                        
            ###############################            
            # concatenate all subjects
            DF = pd.concat([DF,this_data],axis=0)
       
        # count missing
        M = DF[DF['button']!='missing'] 
        missing = M.groupby(['subject','button'])['button'].count()
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_missing.csv'.format(self.exp)))
        
        ### print how many outliers
        print('Missing = {}%'.format(np.true_divide(np.sum(DF['missing']),DF.shape[0])*100))
        print('Dropped trials = {}%'.format(np.true_divide(np.sum(DF['drop_trial']),DF.shape[0])*100))

        #####################
        # merges the actual_frequencies and bins calculated from the oddball task logfiles into subjects' dataframe
        FREQ = pd.read_csv(os.path.join(self.dataframe_folder,'{}_actual_frequencies.csv'.format('task-letter_color_visual_training')))
        FREQ = FREQ.drop(['frequency'],axis=1) # otherwise get double
        FREQ = FREQ.loc[:, ~FREQ.columns.str.contains('^Unnamed')] # drop all unnamed columns
        # inner merge on subject, letter, and color (r)
        M = DF.merge(FREQ,how='inner',on=['subject','letter','r'])
        
        # actual frequencies average:
        AF = M.groupby(['frequency','match'])['actual_frequency'].mean()
        AF.to_csv(os.path.join(self.dataframe_folder,'{}_actual_frequencies_mean.csv'.format(self.exp)))
        
        print('actual frequencies per matching condition')
        print(AF) 
        #####################
        # save whole dataframe with all subjects
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        #####################
        print('success: higherlevel_dataframe')
        
        
    def average_conditions(self, ):
        """Average the phasic pupil per subject per condition of interest. 

        Notes
        -----
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        self.freq_cond argument determines how the trials were split
        """     
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject','trial_num'],inplace=True)
        DF.reset_index()
        
        ############################
        # drop outliers and missing trials
        DF = DF[DF['drop_trial']==0]
        ############################
        
        '''
        ######## CORRECT x FREQUENCY x TIME WINDOW ########
        '''
        DFOUT = DF.groupby(['subject','correct',self.freq_cond]).aggregate({'pupil_feed_locked_t1':'mean', 'pupil_feed_locked_t2':'mean'})
        # DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct-mapping1-timewindow_{}.csv'.format(self.exp,pupil_dv))) # FOR PLOTTING
        # save for RMANOVA format
        DFANOVA =  DFOUT.unstack(['frequency','correct']) 
        print(DFANOVA.columns)
        DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
        DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-frequency-timewindow_rmanova.csv'.format(self.exp))) # for stats
        
        #interaction accuracy and frequency
        for pupil_dv in ['RT', 'pupil_feed_locked_t1', 'pupil_feed_locked_t2', 'pupil_baseline_feed_locked']: #interaction accuracy and frequency
            
            '''
            ######## CORRECT x FREQUENCY ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject','correct',self.freq_cond])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct-frequency_{}.csv'.format(self.exp,pupil_dv))) # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack([self.freq_cond,'correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-frequency_{}_rmanova.csv'.format(self.exp,pupil_dv))) # for stats
            '''
            ######## CORRECT ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject','correct'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct_{}.csv'.format(self.exp,pupil_dv))) # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct_{}_rmanova.csv'.format(self.exp,pupil_dv))) # for stats
        
        '''
        ######## FREQUENCY ########
        '''
        for pupil_dv in ['correct', 'RT', 'pupil_feed_locked_t1', 'pupil_feed_locked_t2', 'pupil_baseline_feed_locked']: # mean accuracy
            DFOUT = DF.groupby(['subject',self.freq_cond])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_frequency_{}.csv'.format(self.exp,pupil_dv))) # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack([self.freq_cond]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp,pupil_dv))) # for stats
        print('success: average_conditions')


    def plot_phasic_pupil_pe(self,):
        """Plot the phasic pupil target_locked interaction frequency and accuracy in each trial bin window.
        
        Notes
        -----
        4 figures: per DV
        GROUP LEVEL DATA
        Separate lines for correct, x-axis is frequency conditions.
        """
        ylim = [ 
            [-1.5,6.5], # t1
            [-3.25,2.25], # t2
            [-3, 5], # baseline
            [0.6,1.5] # RT
        ]
        tick_spacer = [1, 1, 2, .2]
        
        dvs = ['pupil_feed_locked_t1', 'pupil_feed_locked_t2', 'pupil_baseline_feed_locked', 'RT']
        ylabels = ['Pupil response\n(% signal change)', 'Pupil response\n(% signal change)', 'Pupil response\n(% signal change)', 'RT (s)']
        factor = [self.freq_cond,'correct'] 
        xlabel = 'Letter-color frequency'
        xticklabels = ['20%','40%','80%'] 
        labels = ['Error','Correct']
        colors = ['red','blue'] 
        
        xind = np.arange(len(xticklabels))
        dot_offset = [0.05,-0.05]
                
        for dvi,pupil_dv in enumerate(dvs):
            
            fig = plt.figure(figsize=(2, 2))
            ax = fig.add_subplot(111)
            
            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder, '{}_correct-frequency_{}.csv'.format(self.exp, pupil_dv)))
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby(factor)[pupil_dv].agg(['mean', 'std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'], np.sqrt(len(self.subjects)))
            print(GROUP)
            
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
            # # plot line graph
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
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer[dvi]))
            # ax.legend()

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, '{}_correct-frequency_{}_lines.pdf'.format(self.exp, pupil_dv)))
        print('success: plot_phasic_pupil_pe')
        
        
    def plot_behavior(self,):
        """Plot the group level means of accuracy and RT per mapping condition.

        Notes
        -----
        GROUP LEVEL DATA
        x-axis is frequency conditions.
        Figure output as PDF in figure folder.
        """
        #######################
        # Frequency
        #######################
        dvs = ['correct','RT']
        ylabels = ['Accuracy', 'RT (s)']
        factor = self.freq_cond
        xlabel = 'Letter-color frequency'
        xticklabels = ['20%','40%','80%'] 
        bar_width = 0.7
        xind = np.arange(len(xticklabels))
                
        for dvi,pupil_dv in enumerate(dvs):
            
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111) # 1 subplot per bin window

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_{}_{}.csv'.format(self.exp,'frequency',pupil_dv)))
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby([factor])[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
            
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
                       
            # plot bar graph
            for xi,x in enumerate(GROUP[factor]):
                ax.bar(xind[xi],np.array(GROUP['mean'][xi]), width=bar_width, yerr=np.array(GROUP['sem'][xi]), capsize=3, color=(0,0,0,0), edgecolor='black', ecolor='black')
                
            # individual points, repeated measures connected with lines
            DFIN = DFIN.groupby(['subject',factor])[pupil_dv].mean() # hack for unstacking to work
            DFIN = DFIN.unstack(factor)
            for s in np.array(DFIN):
                ax.plot(xind, s, linestyle='-', marker='o', markersize=3, fillstyle='full', color='black', alpha=.2) # marker, line, black
                
            # set figure parameters
            ax.set_title(ylabels[dvi]) # repeat for consistent formatting
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            if pupil_dv == 'correct':
                ax.set_ylim([0.0,1.])
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.2))
                ax.axhline(0.5, linestyle='--', lw=1, alpha=1, color = 'k') # Add dashed horizontal line at chance level
            else:
                ax.set_ylim([0.2,1.8]) #RT
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.4))

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_{}.pdf'.format(self.exp, pupil_dv)))
        print('success: plot_behav')
    
    
    def individual_differences(self,):
       """Correlate frequency effect in pupil DV with frequency effect in accuracy across participants, then plot.
       
       Notes
       -----
       3 figures: 1 per pupil DV
       """
       dvs = ['pupil_feed_locked_t1', 'pupil_feed_locked_t2', 'pupil_baseline_feed_locked']
              
       for sp,pupil_dv in enumerate(dvs):
           fig = plt.figure(figsize=(2,2))
           ax = fig.add_subplot(111) # 1 subplot per bin window
           
           B = pd.read_csv(os.path.join(self.jasp_folder,'{}_frequency_correct_rmanova.csv'.format(self.exp)))
           P = pd.read_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp, pupil_dv)))

           # frequency effect
           P['main_effect_freq'] = (P['80']-P['20'])
           B['main_effect_freq'] = (B['80']-B['20']) # fraction correct
           
           x = np.array(B['main_effect_freq'])
           y = np.array(P['main_effect_freq'])           
           # all subjects
           r,pval = stats.spearmanr(x,y)
           print('all subjects')
           print(pupil_dv)
           print('r={}, p-val={}'.format(r,pval))
           # shell()
           # all subjects in grey
           ax.plot(x, y, 'o', markersize=3, color='green') # marker, line, black
           m, b = np.polyfit(x, y, 1)
           ax.plot(x, m*x+b, color='green',alpha=.5, label='all participants')
           
           # set figure parameters
           ax.set_title('rs = {}, p = {}'.format(np.round(r,2),np.round(pval,3)))
           ax.set_ylabel('{} (80-20%)'.format(pupil_dv))
           ax.set_xlabel('accuracy (80-20%)')
           # ax.legend()
           
           plt.tight_layout()
           fig.savefig(os.path.join(self.figure_folder,'{}_frequency_individual_differences_{}.pdf'.format(self.exp, pupil_dv)))
       print('success: individual_differences')
       
    
    def confound_rt_pupil(self,):
        """Compute single-trial correlation between RT and pupil_dvs, subject and group level
       
        Notes
        -----
        Plots a random subject.
        """
        dvs = ['pupil_feed_locked_t1', 'pupil_feed_locked_t2', 'pupil_baseline_feed_locked']
        DFOUT = pd.DataFrame() # subjects x pupil_dv (fischer z-transformed correlation coefficients)       
        for sp, pupil_dv in enumerate(dvs):

            DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
            
            ############################
            # drop outliers and missing trials
            DF = DF[DF['drop_trial']==0]
            ############################

            plot_subject = np.random.randint(0, len(self.subjects)) # plot random subject
            save_coeff = []
            for s, subj in enumerate(np.unique(DF['subject'])):
                this_df = DF[DF['subject']==subj].copy()

                x = np.array(this_df['RT'])
                y = np.array(this_df[pupil_dv])  
                r,pval = stats.pearsonr(x,y)
                save_coeff.append(self.fisher_transform(r))
                
                if s==plot_subject:  # plot one random subject
                    fig = plt.figure(figsize=(2,2))
                    ax = fig.add_subplot(111)
                    ax.plot(x, y, 'o', markersize=3, color='grey') # marker, line, black
                    m, b = np.polyfit(x, y, 1)
                    ax.plot(x, m*x+b, color='grey',alpha=1)
                    # set figure parameters
                    ax.set_title('subject={}, r = {}, p = {}'.format(subj, np.round(r,2),np.round(pval,3)))
                    ax.set_ylabel(pupil_dv)
                    ax.set_xlabel('RT (s)')
                    # ax.legend()
                    plt.tight_layout()
                    fig.savefig(os.path.join(self.figure_folder,'{}_confound_RT_{}.pdf'.format(self.exp, pupil_dv)))
            DFOUT[pupil_dv] = np.array(save_coeff)
        DFOUT.to_csv(os.path.join(self.jasp_folder, '{}_confound_RT.csv'.format(self.exp)))
        print('success: confound_rt_pupil')


    def confound_baseline_phasic(self,):
        """Compute single-trial correlation between feedback_baseline and phasic t1 and t2.
       
        Notes
        -----
        Plots a random subject.
        """
        dvs = ['pupil_feed_locked_t1', 'pupil_feed_locked_t2']
        DFOUT = pd.DataFrame() # subjects x pupil_dv (fischer z-transformed correlation coefficients)       
        for sp, pupil_dv in enumerate(dvs):

            DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
            
            ############################
            # drop outliers and missing trials
            DF = DF[DF['drop_trial']==0]
            ############################

            plot_subject = np.random.randint(0, len(self.subjects)) # plot random subject
            save_coeff = []
            for s, subj in enumerate(np.unique(DF['subject'])):
                this_df = DF[DF['subject']==subj].copy()

                x = np.array(this_df['pupil_baseline_feed_locked'])
                y = np.array(this_df[pupil_dv])  
                r,pval = stats.pearsonr(x,y)
                save_coeff.append(self.fisher_transform(r))
                
                if s==plot_subject:  # plot one random subject
                    fig = plt.figure(figsize=(2,2))
                    ax = fig.add_subplot(111)
                    ax.plot(x, y, 'o', markersize=3, color='grey') # marker, line, black
                    m, b = np.polyfit(x, y, 1)
                    ax.plot(x, m*x+b, color='grey',alpha=1)
                    # set figure parameters
                    ax.set_title('subject={}, r = {}, p = {}'.format(subj, np.round(r,2),np.round(pval,3)))
                    ax.set_ylabel(pupil_dv)
                    ax.set_xlabel('pupil_baseline_feed_locked')
                    # ax.legend()
                    plt.tight_layout()
                    fig.savefig(os.path.join(self.figure_folder,'{}_confound_baseline_phasic_{}.pdf'.format(self.exp, pupil_dv)))
            DFOUT[pupil_dv] = np.array(save_coeff)
        DFOUT.to_csv(os.path.join(self.jasp_folder, '{}_confound_baseline_phasic.csv'.format(self.exp)))
        print('success: confound_baseline_phasic')
    
    
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
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        csv_names = deepcopy(['subject','correct','frequency','correct*frequency'])
        factors = [['subject'],['correct'],[self.freq_cond],['correct',self.freq_cond]]
        
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
                    SDATA = SDATA[SDATA['drop_trial'] == 0] # drop outliers based on RT
                    #############################
                    
                    evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    df = SDATA.groupby(g_idx)[evoked_cols].mean() # only get kernels out
                    df = pd.DataFrame(df).reset_index()
                    # add to condition dataframe
                    COND = pd.concat([COND,df],join='outer',axis=0) # can also do: this_cond = this_cond.append()  
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,cond)))
        print('success: dataframe_evoked_pupil_higher')
    
    
    def plot_evoked_pupil(self):
        """Plot evoked pupil time courses.
        
        Notes
        -----
        4 figures: mean response, accuracy, frequency, accuracy*frequency.
        Always feed_locked pupil response.
        """
        ylim_feed = [-3,8]
        tick_spacer = 3
        
        #######################
        # FEEDBACK MEAN RESPONSE
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'feed_locked'
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
        
        # shade baseline pupil
        twb = [-self.baseline_window, 0]
        baseline_onset = int(abs(twb[0]*self.sample_rate))
        twb_begin = int(baseline_onset + (twb[0]*self.sample_rate))
        twb_end = int(baseline_onset + (twb[1]*self.sample_rate))
        ax.axvspan(twb_begin,twb_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, ((mid_point-event_onset)/2)+event_onset, mid_point, ((end_sample-mid_point)/2)+mid_point, end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0, self.pupil_step_lim[t][1]*.25, self.pupil_step_lim[t][1]*.5, self.pupil_step_lim[t][1]*.75, self.pupil_step_lim[t][1]])
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
        time_locked = 'feed_locked'
        csv_name = 'correct'
        factor = 'correct'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)))
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

        xticks = [event_onset, ((mid_point-event_onset)/2)+event_onset, mid_point, ((end_sample-mid_point)/2)+mid_point, end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0, self.pupil_step_lim[t][1]*.25, self.pupil_step_lim[t][1]*.5, self.pupil_step_lim[t][1]*.75, self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked_{}.pdf'.format(self.exp, csv_name)))
        
        #######################
        # FREQUENCY
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'feed_locked'
        csv_name = 'frequency'
        factor = 'frequency'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['20%','40%','80%']
        colorsts = ['indigo','indigo','indigo']
        alpha_fills = [0.2,0.2,0.2] # fill
        alpha_lines = [.3,.6,1.]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        ### STATS - RM_ANOVA ###
        # loop over time points, run anova, save F-statistic for cluster correction
        # first 3 columns are subject, correct, frequency
        # get pval for the interaction term (last element in res.anova_table)
        interaction_pvals = np.empty(COND.shape[-1]-3)
        for timepoint in np.arange(COND.shape[-1]-3):            
            this_df = COND.iloc[:,:timepoint+4]
            aovrm = AnovaRM(this_df, str(timepoint), 'subject', within=['frequency'])
            res = aovrm.fit()
            interaction_pvals[timepoint] = np.array(res.anova_table)[-1][-1] # last row, last element
            
        # stats        
        self.timeseries_fdr_correction(pvals=interaction_pvals, xind=pd.Series(range(interaction_pvals.shape[-1])), color='black', ax=ax)
    
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, ((mid_point-event_onset)/2)+event_onset, mid_point, ((end_sample-mid_point)/2)+mid_point, end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0, self.pupil_step_lim[t][1]*.25, self.pupil_step_lim[t][1]*.5, self.pupil_step_lim[t][1]*.75, self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        ax.legend(loc='best')
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked_{}.pdf'.format(self.exp, csv_name)))
        
        #######################
        # CORRECT x FREQUENCY
        #######################
        fig = plt.figure(figsize=(4,2))
        ax = fig.add_subplot(111)
        t = 0
        time_locked = 'feed_locked'
        csv_name = 'correct-frequency'
        factor = ['correct',self.freq_cond]
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        labels_frequences = np.unique(COND[self.freq_cond])
        
        ########
        # make unique labels for each of the 4 conditions
        conditions = [
            (COND['correct'] == 0) & (COND[self.freq_cond] == labels_frequences[2]), # Easy Error 1
            (COND['correct'] == 1) & (COND[self.freq_cond] == labels_frequences[2]), # Easy Correct 2
            (COND['correct'] == 0) & (COND[self.freq_cond] == labels_frequences[0]), # Hard Error 3
            (COND['correct'] == 1) & (COND[self.freq_cond] == labels_frequences[0]), # Hard Correct 4
            (COND['correct'] == 0) & (COND[self.freq_cond] == labels_frequences[1]), # Medium Error 5 # coded like this to keep in order with other experiments
            (COND['correct'] == 1) & (COND[self.freq_cond] == labels_frequences[1]), # Medium Correct 6
            ]
        values = [1,2,3,4,5,6]
        conditions = np.select(conditions, values) # don't add as column to time series otherwise it gets plotted
        ########
                    
        xticklabels = ['Error 80%', 'Correct 80%', 'Error 20%', 'Correct 20%', 'Error 40%', 'Correct 40%']
        colorsts = ['r', 'b', 'r', 'b', 'r', 'b']
        alpha_fills = [0.2, 0.2, 0.1, 0.1, 0.15, .15] # fill
        alpha_lines = [1, 1, 0.6, 0.6, 0.8, 0.8]
        linestyle= ['solid', 'solid', 'dotted', 'dotted', 'dashed', 'dashed']
        save_conds = []
        # plot time series
        
        for i,x in enumerate(values):
            TS = COND[conditions==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, linestyle=linestyle[i], color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        ### STATS - RM_ANOVA ###
        # loop over time points, run anova, save F-statistic for cluster correction
        # first 3 columns are subject, correct, frequency
        # get pval for the interaction term (last element in res.anova_table)
        interaction_pvals = np.empty(COND.shape[-1]-3)
        for timepoint in np.arange(COND.shape[-1]-3):            
            this_df = COND.iloc[:,:timepoint+4]
            aovrm = AnovaRM(this_df, str(timepoint), 'subject', within=['correct', 'frequency'])
            res = aovrm.fit()
            interaction_pvals[timepoint] = np.array(res.anova_table)[-1][-1] # last row, last element
            
        # stats        
        self.timeseries_fdr_correction(pvals=interaction_pvals, xind=pd.Series(range(interaction_pvals.shape[-1])), color='black', ax=ax)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            
        xticks = [event_onset, ((mid_point-event_onset)/2)+event_onset, mid_point, ((end_sample-mid_point)/2)+mid_point, end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0, self.pupil_step_lim[t][1]*.25, self.pupil_step_lim[t][1]*.5, self.pupil_step_lim[t][1]*.75, self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
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
    
 