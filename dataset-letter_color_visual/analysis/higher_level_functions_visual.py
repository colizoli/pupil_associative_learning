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
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from copy import deepcopy
import itertools
from IPython import embed as shell # for debugging only

pd.set_option('display.float_format', lambda x: '%.8f' % x) # suppress scientific notation in pandas

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
            B = pd.read_csv(this_log, float_precision='%.16f') # behavioral file
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
                    B.to_csv(this_log, float_format='%.16f')
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
            
            this_data = pd.read_csv(os.path.join(self.project_directory, subj, 'beh', '{}_{}_beh.csv'.format(subj,self.exp)), float_precision='%.16f')
            this_data = this_data.loc[:, ~this_data.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            # open baseline pupil to add to dataframes as well
            this_baseline = pd.read_csv(os.path.join(self.project_directory, subj, 'beh', '{}_{}_recording-eyetracking_physio_{}_baselines.csv'.format(subj, self.exp, 'feed_locked')), float_precision='%.16f')
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
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_missing.csv'.format(self.exp)), float_format='%.16f')
        
        ### print how many outliers
        print('Missing = {}%'.format(np.true_divide(np.sum(DF['missing']),DF.shape[0])*100))
        print('Dropped trials = {}%'.format(np.true_divide(np.sum(DF['drop_trial']),DF.shape[0])*100))

        #####################
        # merges the actual_frequencies and bins calculated from the oddball task logfiles into subjects' dataframe
        FREQ = pd.read_csv(os.path.join(self.dataframe_folder,'{}_actual_frequencies.csv'.format('task-letter_color_visual_training')), float_precision='%.16f')
        FREQ = FREQ.drop(['frequency'],axis=1) # otherwise get double
        FREQ = FREQ.loc[:, ~FREQ.columns.str.contains('^Unnamed')] # drop all unnamed columns
        # inner merge on subject, letter, and color (r)
        M = DF.merge(FREQ,how='inner',on=['subject','letter','r'])
        
        # actual frequencies average:
        AF = M.groupby(['frequency','match'])['actual_frequency'].mean()
        AF.to_csv(os.path.join(self.dataframe_folder,'{}_actual_frequencies_mean.csv'.format(self.exp)), float_format='%.16f')
        
        print('actual frequencies per matching condition')
        print(AF) 
        #####################
        # save whole dataframe with all subjects
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_format='%.16f')
        #####################
        print('success: higherlevel_dataframe')
        
        
    def average_conditions(self, ):
        """Average the phasic pupil per subject per condition of interest. 

        Notes
        -----
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        self.freq_cond argument determines how the trials were split
        """     
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
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
        DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-frequency-timewindow_rmanova.csv'.format(self.exp)), float_format='%.16f') # for stats
        
        #interaction accuracy and frequency
        for pupil_dv in ['RT', 'pupil_feed_locked_t1', 'pupil_feed_locked_t2', 'pupil_baseline_feed_locked']: #interaction accuracy and frequency
            
            '''
            ######## CORRECT x FREQUENCY ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject','correct',self.freq_cond])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct-frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack([self.freq_cond,'correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
            '''
            ######## CORRECT ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject','correct'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        
        '''
        ######## FREQUENCY ########
        '''
        for pupil_dv in ['correct', 'RT', 'pupil_feed_locked_t1', 'pupil_feed_locked_t2', 'pupil_baseline_feed_locked']: # mean accuracy
            DFOUT = DF.groupby(['subject',self.freq_cond])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack([self.freq_cond]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
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
            
            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder, '{}_correct-frequency_{}.csv'.format(self.exp, pupil_dv)), float_precision='%.16f')
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

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_{}_{}.csv'.format(self.exp,'frequency',pupil_dv)), float_precision='%.16f')
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
           
           B = pd.read_csv(os.path.join(self.jasp_folder,'{}_frequency_correct_rmanova.csv'.format(self.exp)), float_precision='%.16f')
           P = pd.read_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp, pupil_dv)), float_precision='%.16f')

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
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f'))
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
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,cond)), float_format='%.16f')
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
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
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
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)), float_precision='%.16f')
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
    
    
    def information_theory_code_stimuli(self, fn_in):
        """Add a new column in the subjects dataframe to give each letter-color pair a unique identifier.
        
        Parameters
        ----------
        fn_in : str
            The path to the subjects' dataframe.
        
        Notes
        -----
        6 letters and 6 shades of green -> 36 different letter-color pair combinations.
        
        New column name is "letter_color_pair"
        """
        df_in = pd.read_csv(fn_in, float_precision='%.16f')
        
        # make new column to give each letter-color combination a unique identifier (1 - 36)        
        mapping = [
            # KEEP ORIGINAL MAPPINGS TO SEE 'FLIP'
            (df_in['letter'] == 'A') & (df_in['r'] == 76), 
            (df_in['letter'] == 'A') & (df_in['r'] == 157), 
            (df_in['letter'] == 'A') & (df_in['r'] == 0), 
            (df_in['letter'] == 'A') & (df_in['r'] == 3), 
            (df_in['letter'] == 'A') & (df_in['r'] == 138), 
            (df_in['letter'] == 'A') & (df_in['r'] == 75), 
            #
            (df_in['letter'] == 'D') & (df_in['r'] == 76), 
            (df_in['letter'] == 'D') & (df_in['r'] == 157), 
            (df_in['letter'] == 'D') & (df_in['r'] == 0), 
            (df_in['letter'] == 'D') & (df_in['r'] == 3), 
            (df_in['letter'] == 'D') & (df_in['r'] == 138), 
            (df_in['letter'] == 'D') & (df_in['r'] == 75), 
            #
            (df_in['letter'] == 'I') & (df_in['r'] == 76), 
            (df_in['letter'] == 'I') & (df_in['r'] == 157), 
            (df_in['letter'] == 'I') & (df_in['r'] == 0), 
            (df_in['letter'] == 'I') & (df_in['r'] == 3), 
            (df_in['letter'] == 'I') & (df_in['r'] == 138), 
            (df_in['letter'] == 'I') & (df_in['r'] == 75), 
            #
            (df_in['letter'] == 'O') & (df_in['r'] == 76), 
            (df_in['letter'] == 'O') & (df_in['r'] == 157), 
            (df_in['letter'] == 'O') & (df_in['r'] == 0), 
            (df_in['letter'] == 'O') & (df_in['r'] == 3), 
            (df_in['letter'] == 'O') & (df_in['r'] == 138), 
            (df_in['letter'] == 'O') & (df_in['r'] == 75), 
            #
            (df_in['letter'] == 'R') & (df_in['r'] == 76), 
            (df_in['letter'] == 'R') & (df_in['r'] == 157), 
            (df_in['letter'] == 'R') & (df_in['r'] == 0), 
            (df_in['letter'] == 'R') & (df_in['r'] == 3), 
            (df_in['letter'] == 'R') & (df_in['r'] == 138), 
            (df_in['letter'] == 'R') & (df_in['r'] == 75), 
            #
            (df_in['letter'] == 'T') & (df_in['r'] == 76),  
            (df_in['letter'] == 'T') & (df_in['r'] == 157), 
            (df_in['letter'] == 'T') & (df_in['r'] == 0), 
            (df_in['letter'] == 'T') & (df_in['r'] == 3), 
            (df_in['letter'] == 'T') & (df_in['r'] == 138), 
            (df_in['letter'] == 'T') & (df_in['r'] == 75), 
            ]
        
        elements = np.arange(36) # also elements is the same as priors (start with 0 so they can be indexed by element)
        df_in['letter_color_pair'] = np.select(mapping, elements)
        
        df_in.to_csv(fn_in, float_format='%.16f') # save with new columns
        print('success: information_theory_code_stimuli')   
    

    def idt_model(self, df, df_data_column, elements, priors, flat_prior=False):
        """Process Ideal Learner Model.
        
        Parameters
        ----------
        df : pandas dataframe
            The dataframe to apply the Ideal Learner Model to.
        
        df_data_column : str
            The name of the column that refers to the cue-target pairs for all trials in the experiment.
        
        elements : list
            The list of unique indentifiers for the cue-target pairs.
        
        priors : list
            The list of priors as probabilities.
        
        Returns
        -------
        [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D]: list
            A list containing all model parameters (see notes).
            
        Notes
        -----
        Ideal Learner Model adapted from Poli, Mars, & Hunnius (2020).
        See also: https://github.com/FrancescPoli/eye_processing/blob/master/ITDmodel.m
        
        Priors are generated from the probabilities of the letter-color pair in the odd-ball task.
        
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
        for t in np.arange(df.shape[0]):
            vector = data[:t+1] #  trial number starts at 0, all the targets that have been seen so far
            
            model_e.append(vector[-1])  # element in current trial = last element in the vector
            
            # print(vector)
            if t < 1: # if it's the first trial, our expectations are based only on the prior (values)
                
                if not flat_prior:
                    # PRIORS BASED ON ODDBALL TASK
                    alpha1 = priors*len(elements) # np.sum(alpha) == len(elements), priors from odd-ball task
                    p1 = priors # priors based on odd-ball task, np.sum(priors) should equal 1
                    p = p1
                else:
                    # UNIFORM PRIOR DISTRIBUTION
                    alpha1 = np.ones(len(elements)) # np.sum(alpha) == len(elements), flat prior
                    p1 = alpha1 / len(elements) # probablity, i.e., np.sum(p1) == 1
                    p = p1
            
            # at every trial, we compute surprise based on the probability
            model_P.append(p)           # probability (all elements)
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
        
        
    def information_theory_estimates(self, flat_prior=False):
        """Run subject loop on Ideal Learner Model and save model estimates.
        
        Parameters
        ----------
        flat_prior : boolean
            Use a uniform prior (True) or input priors from odd-ball task (False); default False
        
        Notes
        -----
        Ideal Learner Model adapted from Poli, Mars, & Hunnius (2020).
        See also: https://github.com/FrancescPoli/eye_processing/blob/master/ITDmodel.m
        
        Model estimates that are saved in subject's dataframe:
        model_i = surprise of current element at current trial
        model_H = entropy at current trial
        model_D = KL-divergence at current trial
        """
        
        fn_in = os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp))
        priors = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects_priors.csv'.format('task-letter_color_visual_training')), float_precision='%.16f')
        
        self.information_theory_code_stimuli(fn_in) # code stimuli based on predictions and based on targets
        
        df_in = pd.read_csv(fn_in, float_precision='%.16f')
        df_in = df_in.loc[:, ~df_in.columns.str.contains('^Unnamed')]
        # sort by subjects then trial_counter in ascending order
        df_in.sort_values(by=['subject', 'trial_num'], ascending=True, inplace=True)
        
        df_out = pd.DataFrame()
        df_prob_out = pd.DataFrame() # last probabilities all elements saved
        
        elements = np.unique(df_in['letter_color_pair'])
        
        # loop subjects
        for s,subj in enumerate(self.subjects):
            
            this_subj = int(''.join(filter(str.isdigit, subj))) # get number of subject only
            # get current subjects data only
            this_priors = priors[str(this_subj)] # priors for current subject
            this_df = df_in[df_in['subject']==this_subj].copy()
            
            # the input to the model is the trial sequence = the order of letter-color pair for each participant
            [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D] = self.idt_model(this_df, 'letter_color_pair', elements, this_priors, flat_prior)
            
            # add to subject dataframe
            this_df['model_p'] = np.array(model_p)
            this_df['model_i'] = np.array(model_i)
            this_df['model_H'] = np.array(model_H)
            this_df['model_D'] = np.array(model_D)
            df_out = pd.concat([df_out, this_df])    # add current subject df to larger df
            
            df_prob_out['{}'.format(this_subj)] = np.array(model_P[-1])
            print(subj)
        
        # save whole DF
        df_out.to_csv(fn_in, float_format='%.16f') # overwrite subjects dataframe
        print('success: information_theory_estimates')
        

    def average_information_conditions(self, ):
        """Average the model parameters per subject per condition of interest. 

        Notes
        -----
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        self.freq_cond argument determines how the trials were split
        """     
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject','trial_num'],inplace=True)
        DF.reset_index()
        
        ############################
        # drop outliers and missing trials
        DF = DF[DF['drop_trial']==0]
        ############################
        
        #interaction accuracy and frequency
        for pupil_dv in ['model_i', 'model_H', 'model_D']: #interaction accuracy and frequency
            
            '''
            ######## CORRECT x FREQUENCY ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject','correct',self.freq_cond])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct-frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack([self.freq_cond,'correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
            '''
            ######## CORRECT ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject','correct'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        
        '''
        ######## FREQUENCY ########
        '''
        for pupil_dv in ['model_i', 'model_H', 'model_D']: # mean accuracy
            DFOUT = DF.groupby(['subject',self.freq_cond])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack([self.freq_cond]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        print('success: average_information_conditions')


    def plot_information(self, ):
        """Plot the model parameters across trials and average over subjects
        Then, plot the model parameters by frequency

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
                        
            subject_array = np.zeros((len(self.subjects), np.max(DFIN['trial_num'])+1))
        
            for s, subj in enumerate(self.subjects):
                this_subj = int(''.join(filter(str.isdigit, subj)))
                this_df = DFIN[DFIN['subject']==this_subj].copy()        
                subject_array[s,:] = np.ravel(this_df[[pupil_dv]])
                            
            self.tsplot(ax, subject_array, color=colors[dvi], label=ylabels[dvi])
    
            # set figure parameters
            ax.set_xlim([0, np.max(DFIN['trial_num'])+1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabels[dvi])
            # ax.legend()
            subplot_counter += 1
        
        # PLOT ACROSS FREQUENCY CONDITIONS
        factor = self.freq_cond
        xlabel = 'Letter-color frequency'
        xticklabels = ['20%','40%','80%'] 
        bar_width = 0.7
        xind = np.arange(len(xticklabels))
        
        for dvi,pupil_dv in enumerate(dvs):
            
            ax = fig.add_subplot(3, 3, subplot_counter) # 1 subplot per bin window

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_{}_{}.csv'.format(self.exp,'frequency',pupil_dv)), float_precision='%.16f')
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average 
            GROUP = pd.DataFrame(DFIN.groupby([factor])[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
            
            # ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
                       
            # plot bar graph
            for xi,x in enumerate(GROUP[factor]):
                ax.bar(xind[xi],np.array(GROUP['mean'][xi]), width=bar_width, yerr=np.array(GROUP['sem'][xi]), capsize=3, color=colors[dvi], edgecolor='black', ecolor='black')
                
            # individual points, repeated measures connected with lines
            # DFIN = DFIN.groupby(['subject',factor])[pupil_dv].mean() # hack for unstacking to work
            # DFIN = DFIN.unstack(factor)
            # for s in np.array(DFIN):
            #     ax.plot(xind, s, linestyle='-', marker='o', markersize=3, fillstyle='full', color='black', alpha=.1) # marker, line, black
                
            # set figure parameters
            # ax.set_title(ylabels[dvi]) # repeat for consistent formatting
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            if pupil_dv == 'model_D':
                ax.set_ylim([0.004, 0.007])
            if pupil_dv == 'model_i':
                ax.set_ylim([4.9, 5.2])
            if pupil_dv == 'model_H':
                ax.set_ylim([4.5, 4.75])
            # if pupil_dv == 'model_D':
            #     ax.set_ylim([0.004, 0.00475])
            # if pupil_dv == 'model_i':
            #     ax.set_ylim([4.95, 5.05])
            # if pupil_dv == 'model_H':
            #     ax.set_ylim([4.85, 4.9])
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
        
        ivs = ['model_i', 'model_H', 'model_D',]
        labels = ['i' , 'H', 'KL',]

        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')

        #### DROP OMISSIONS HERE ####
        DF = DF[DF['drop_trial'] == 0] # drop outliers based on RT
        #############################
        
        corr_out = []

        # loop subjects
        for s, subj in enumerate(self.subjects):
            
            this_subj = int(''.join(filter(str.isdigit, subj))) 
            # get current subject's data only
            this_df = DF[DF['subject']==this_subj].copy(deep=False)
                            
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
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(121)
        cbar_ax = fig.add_subplot(122)
        
        # mask for significance
        mask_pvals = pvals < 0.05
        mask_pvals = ~mask_pvals # True means mask this cell
        
        # plot only lower triangle
        mask = np.triu(np.ones_like(corr_mean))
        mask = mask + mask_pvals # only show sigificant correlations in heatmap
        
        # ax = sns.heatmap(corr_mean, vmin=-1, vmax=1, mask=mask, cmap='bwr', cbar_ax=cbar_ax, xticklabels=labels, yticklabels=labels, square=True, annot=True, ax=ax)
        ax = sns.heatmap(corr_mean, vmin=-1, vmax=1, cmap='bwr', cbar_ax=cbar_ax, xticklabels=labels, yticklabels=labels, square=True, annot=True, ax=ax)
        
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_pupil_information_correlation_matrix.pdf'.format(self.exp)))
                        
        print('success: pupil_information_correlation_matrix')


    def dataframe_evoked_correlation(self):
        """Correlation of theoretic variables with other variables removed.

        Notes
        -----
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns

        ivs = ['model_i', 'model_H', 'model_D']

        pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas

        for t,time_locked in enumerate(self.time_locked):

            for cond in ['correct', 'error', 'all_trials']:
                                
                # Loop through IVs
                for i,iv in enumerate(ivs):
                    
                    df_out = pd.DataFrame() # timepoints x subjects
                    
                    # loop subjects
                    for s, subj in enumerate(self.subjects):

                        this_subj = int(''.join(filter(str.isdigit, subj)))
                        # get current subject's data only

                        SBEHAV = DF[DF['subject']==this_subj].reset_index()
                        SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f'))
                        SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns

                        # merge behavioral and evoked dataframes so we can group by conditions
                        SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)

                        #### DROP OMISSIONS HERE ####
                        SDATA = SDATA[SDATA['drop_trial'] == 0] # drop outliers based on RT
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
                        df_out[subj] = df_out[subj].apply(lambda x: '%.16f' % x) # remove scientific notation from df

                    # save output file
                    df_out.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_{}_{}.csv'.format(self.exp, time_locked, cond, iv)), float_format='%.16f')
        print('success: dataframe_evoked_regression')


    def plot_pupil_information_regression_evoked(self):
        """Plot partial correlation between pupil response and model estimates.
        
        Notes
        -----
        Always feed_locked pupil response.
        Partial correlations are done for all trials as well as for correct and error trials separately.
        """
        ylim_feed = [-0.2, 0.2]
        tick_spacer = 0.1
        
        ivs = ['model_i', 'model_H', 'model_D']
    
        # xticklabels = ['mean response']
        colors = ['teal', 'orange', 'purple'] # black
        alphas = [1]
        
        #######################
        # FEEDBACK PLOT BETAS FOR EACH MODEL DV
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
        
        for i,iv in enumerate(ivs):
            # Compute means, sems across group
            COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_{}_{}.csv'.format(self.exp, time_locked, 'all_trials', iv)), float_precision='%.16f')
            COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns

            # plot time series
            TS = np.array(COND.T) # flip so subjects are rows
            self.tsplot(ax, TS, color=colors[i], label=iv)
            try:
                self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1+i, color=colors[i], ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)           
            except:
                shell()
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
            time_locked = 'feed_locked'
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
            self.cluster_sig_bar_1samp(array=np.subtract(save_conds[1], save_conds[0]), x=pd.Series(range(TS.shape[-1])), yloc=3, color='purple', ax=ax, threshold=0.05, nrand=5000, cluster_correct=False)
            self.cluster_sig_bar_1samp(array=np.subtract(save_conds[1], save_conds[0]), x=pd.Series(range(TS.shape[-1])), yloc=4, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
            
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
            ax.set_ylabel('r')
            ax.set_title(iv)
            ax.legend()
            # whole figure format
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_evoked_correlation_{}.pdf'.format(self.exp, iv)))
        print('success: plot_pupil_information_regression_evoked')
        


    # def rt_pupil_correlation(self,):
    #     """Correlate the RTs with the post-feedback pupil response in the early time window across participants.
    #
    #     Notes
    #     -----
    #     Outputs a dataframe to enter into JASP
    #     """
    #     dvs = ['pupil_feed_locked_t1', 'pupil_feed_locked_t2', 'pupil_baseline_feed_locked']
    #
    #     df_in = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
    #     df_out = pd.DataFrame()
    #     df_out['subject'] = self.subjects
    #     ############################
    #     # drop outliers and missing trials
    #     df_in = df_in[df_in['drop_trial']==0]
    #
    #     ############################
    #
    #     for correct in [0,1]:
    #         # loop pupil dvs
    #         for sp,pupil_dv in enumerate(dvs):
    #
    #            save_coeff = []
    #            # loop subjects
    #            for s,subj in enumerate(self.subjects):
    #
    #                this_subj = int(''.join(filter(str.isdigit, subj))) # get number of subject only
    #
    #                this_df = df_in[df_in['subject'] == this_subj].copy()
    #                this_df = this_df[this_df['correct']==correct].copy()
    #
    #                x = np.array(this_df['RT'])
    #                y = np.array(this_df[pupil_dv])
    #                # all subjects
    #                r,pval = stats.spearmanr(x,y)
    #
    #                save_coeff.append(r)
    #
    #            df_out[pupil_dv + '_correct{}'.format(correct)] = np.array(save_coeff)
    #     df_out.to_csv(os.path.join(self.jasp_folder,'{}_rt_pupil_correlations.csv'.format(self.exp)))
    #     print('success: rt_pupil_correlation')
        

    # def plot_information_frequency(self,):
   #      """Plot the model parameteres by frequency condition
   #
   #      Notes
   #      -----
   #      GROUP LEVEL DATA
   #      x-axis is frequency conditions.
   #      Figure output as PDF in figure folder.
   #      """
   #      #######################
   #      # Frequency
   #      #######################
   #      dvs = [ 'model_D', 'model_i', 'model_H']
   #      ylabels = ['KL divergence', 'Surprise', 'Entropy']
   #      factor = self.freq_cond
   #      xlabel = 'Letter-color frequency'
   #      xticklabels = ['20%','40%','80%']
   #      bar_width = 0.7
   #      xind = np.arange(len(xticklabels))
   #
   #      colors = ['purple', 'teal', 'orange']
   #
   #      fig = plt.figure(figsize=(4,2))
   #
   #      for dvi,pupil_dv in enumerate(dvs):
   #
   #          ax = fig.add_subplot(1, 3, dvi+1) # 1 subplot per bin window
   #
   #          DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_{}_{}.csv'.format(self.exp,'frequency',pupil_dv)), float_precision='%.16f')
   #          DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
   #
   #          # Group average
   #          GROUP = pd.DataFrame(DFIN.groupby([factor])[pupil_dv].agg(['mean','std']).reset_index())
   #          GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
   #          print(GROUP)
   #
   #          # ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
   #
   #          # plot bar graph
   #          for xi,x in enumerate(GROUP[factor]):
   #              ax.bar(xind[xi],np.array(GROUP['mean'][xi]), width=bar_width, yerr=np.array(GROUP['sem'][xi]), capsize=3, color=colors[dvi], edgecolor='black', ecolor='black')
   #
   #          # individual points, repeated measures connected with lines
   #          # DFIN = DFIN.groupby(['subject',factor])[pupil_dv].mean() # hack for unstacking to work
   #          # DFIN = DFIN.unstack(factor)
   #          # for s in np.array(DFIN):
   #          #     ax.plot(xind, s, linestyle='-', marker='o', markersize=3, fillstyle='full', color='black', alpha=.1) # marker, line, black
   #
   #          # set figure parameters
   #          # ax.set_title(ylabels[dvi]) # repeat for consistent formatting
   #          ax.set_ylabel(ylabels[dvi])
   #          ax.set_xlabel(xlabel)
   #          ax.set_xticks(xind)
   #          ax.set_xticklabels(xticklabels)
   #          # if pupil_dv == 'model_D':
   #          #     ax.set_ylim([0.0042, 0.0048])
   #          # if pupil_dv == 'model_i':
   #          #     ax.set_ylim([4.95, 5.1])
   #          # if pupil_dv == 'model_H':
   #          #     ax.set_ylim([4.85, 4.90])
   #
   #      sns.despine(offset=10, trim=True)
   #      plt.tight_layout()
   #      fig.savefig(os.path.join(self.figure_folder,'{}_information_frequency.pdf'.format(self.exp)))
   #      print('success: plot_information_frequency')
   #

        
# not using
    #
    # def partial_correlation_information(self,):
    #     """Carry out the partial correlations of the three model parameters
    #
    #     Notes
    #     -----
    #     Model estimates that are correlated per subject the tested at group level:
    #     model_i = surprise of current element at current trial
    #     model_H = negative entropy at current trial
    #     model_D = KL-divergence at current trial
    #
    #     """
    #
    #     ivs = ['model_i', 'model_H', 'model_D']
    #     labels = ['i' , 'H', 'KL']
    #
    #     fn_in = os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp))
    #     DF = pd.read_csv(fn_in, float_precision='%.16f')
    #
    #
    #     for i,iv in enumerate(ivs):
    #
    #         save_iv = [] # append clean model parameters here
    #
    #         # loop subjects
    #         for s, subj in enumerate(self.subjects):
    #
    #             this_subj = int(''.join(filter(str.isdigit, subj)))
    #             # get current subject's data only
    #             this_df = DF[DF['subject']==this_subj].copy(deep=False)
    #
    #             # First remove other ivs from current iv with linear regression
    #             remove_ivs = [i for i in ivs if not i == iv]
    #
    #             # model: iv1 ~ constant + iv2 + iv3, take residuals into correlation with pupil
    #             Y = np.array(this_df[iv]) # current iv
    #             X = this_df[remove_ivs]
    #
    #             # partial correlation via ordinary least squares linear regression, get residuals
    #             model = sm.OLS(Y, X, missing='drop')
    #             results = model.fit()
    #             x = results.resid # residuals of theoretic variable regression
    #
    #             shell()
    #             save_iv.append(x)
    #
    #         DF['{}_clean'.format(iv)] = np.concatenate(save_iv)
    #
    #     # save whole DF
    #     DF.to_csv(fn_in, float_format='%.16f') # overwrite subjects dataframe
    #
    #     print('success: partial_correlation_information')
        
    
    #
    # def plot_information_pe(self,):
    #     """Plot the model parameters interaction frequency and accuracy in each trial bin window.
    #
    #     Notes
    #     -----
    #     4 figures: per DV
    #     GROUP LEVEL DATA
    #     Separate lines for correct, x-axis is frequency conditions.
    #     """
    #     tick_spacer = [1, 1, 2, .2]
    #
    #     dvs = ['model_i', 'model_H', 'model_D']
    #     ylabels = ['Surprise', 'Negative entropy', 'KL divergence']
    #     factor = [self.freq_cond,'correct']
    #     xlabel = 'Letter-color frequency'
    #     xticklabels = ['20%','40%','80%']
    #     labels = ['Error','Correct']
    #     colors = ['red','blue']
    #
    #     xind = np.arange(len(xticklabels))
    #     dot_offset = [0.05,-0.05]
    #
    #     for dvi,pupil_dv in enumerate(dvs):
    #
    #         fig = plt.figure(figsize=(2, 2))
    #         ax = fig.add_subplot(111)
    #
    #         DFIN = pd.read_csv(os.path.join(self.trial_bin_folder, '{}_correct-frequency_{}.csv'.format(self.exp, pupil_dv)), float_precision='%.16f')
    #         DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
    #
    #         # Group average per BIN WINDOW
    #         GROUP = pd.DataFrame(DFIN.groupby(factor)[pupil_dv].agg(['mean', 'std']).reset_index())
    #         GROUP['sem'] = np.true_divide(GROUP['std'], np.sqrt(len(self.subjects)))
    #         print(GROUP)
    #
    #         # ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
    #
    #         # # plot line graph
    #         for x in[0,1]: # split by error, correct
    #             D = GROUP[GROUP['correct']==x]
    #             print(D)
    #             ax.errorbar(xind, np.array(D['mean']), yerr=np.array(D['sem']), marker='o', markersize=3, fmt='-', elinewidth=1, label=labels[x], capsize=3, color=colors[x], alpha=1)
    #
    #         # set figure parameters
    #         ax.set_title('{}'.format(pupil_dv))
    #         ax.set_ylabel(ylabels[dvi])
    #         ax.set_xlabel(xlabel)
    #         ax.set_xticks(xind)
    #         ax.set_xticklabels(xticklabels)
    #         # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer[dvi]))
    #         # ax.legend()
    #
    #         sns.despine(offset=10, trim=True)
    #         plt.tight_layout()
    #         fig.savefig(os.path.join(self.figure_folder, '{}_correct-frequency_{}_lines.pdf'.format(self.exp, pupil_dv)))
    #     print('success: plot_information_pe')
    #
    #
    #
    # def information_evoked_get_phasics(self,):
    #     """Compute average partial correlation coefficients in selected time window per trial and adds average to behavioral data frame.
    #
    #     Notes
    #     -----
    #     Always target_locked pupil response.
    #     Partial correlations are done for all trials as well as for correct and error trials separately.
    #     """
    #
    #     ivs = ['model_i', 'model_H', 'model_D']
    #
    #     DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
    #     # sort by subjects then trial_counter in ascending order
    #     DF.sort_values(by=['subject', 'trial_num'], ascending=True, inplace=True)
    #     DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns
    #
    #     # loop theoretic variables
    #     for iv in ivs:
    #
    #         for cond in ['correct', 'error', 'all_trials']:
    #
    #             # loop through each type of event to lock events to...
    #             for t,time_locked in enumerate(self.time_locked):
    #
    #                 df_out = pd.DataFrame()
    #
    #                 # load evoked pupil file (all subjects as columns and time points as rows)
    #                 df_pupil = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_{}_{}.csv'.format(self.exp, time_locked, cond, iv)), float_precision='%.16f')
    #                 df_pupil = df_pupil.loc[:, ~df_pupil.columns.str.contains('^Unnamed')] # remove all unnamed columns
    #
    #                 pupil_step_lim = self.pupil_step_lim[t] # kernel size is always the same for each event type
    #
    #                 for twi,pupil_time_of_interest in enumerate(self.pupil_time_of_interest[t]): # multiple time windows to average
    #
    #                     save_phasics = []
    #
    #                     # loop subjects
    #                     for s,subj in enumerate(self.subjects):
    #
    #                         P = np.array(df_pupil[subj]) # current subject
    #
    #                         # in seconds
    #                         phase_start = -pupil_step_lim[0] + pupil_time_of_interest[0]
    #                         phase_end = -pupil_step_lim[0] + pupil_time_of_interest[1]
    #                         # in sample rate units
    #                         phase_start = int(phase_start*self.sample_rate)
    #                         phase_end = int(phase_end*self.sample_rate)
    #                         # mean within phasic time window
    #                         this_phasic = np.nanmean(P[phase_start:phase_end])
    #
    #                         save_phasics.append(this_phasic)
    #                         print(subj)
    #                     # save phasics
    #                     df_out['coeff_{}_t{}'.format(time_locked,twi+1)] = np.array(save_phasics)
    #
    #             #######################
    #             df_out['subject'] = self.subjects
    #             df_out.to_csv(os.path.join(self.jasp_folder, '{}_correlation_phasic_{}_{}.csv'.format(self.exp, cond, iv)), float_format='%.16f')
    #
    #         # combine error and correct for 2-way interaction test in JASP
    #         df_error = pd.read_csv(os.path.join(self.jasp_folder, '{}_correlation_phasic_{}_{}.csv'.format(self.exp, 'error', iv)), float_precision='%.16f')
    #         df_correct = pd.read_csv(os.path.join(self.jasp_folder, '{}_correlation_phasic_{}_{}.csv'.format(self.exp, 'correct', iv)), float_precision='%.16f')
    #
    #         df_error.rename(columns={"coeff_feed_locked_t1": "coeff_feed_locked_t1_error", "coeff_feed_locked_t2": "coeff_feed_locked_t2_error"}, inplace=True)
    #         df_correct.rename(columns={"coeff_feed_locked_t1": "coeff_feed_locked_t1_correct", "coeff_feed_locked_t2": "coeff_feed_locked_t2_correct"}, inplace=True)
    #
    #         df_anova = pd.concat([df_error, df_correct], axis=1)
    #         df_anova = df_anova.loc[:, ~df_anova.columns.str.contains('^Unnamed')] # drop all unnamed columns
    #
    #         df_anova.to_csv(os.path.join(self.jasp_folder, '{}_correlation_phasic_{}_{}.csv'.format(self.exp, 'accuracy_anova', iv)), float_format='%.16f')
    #
    #     print('success: information_evoked_get_phasics')
    #
    #
    # def plot_information_phasics(self, ):
    #     """Plot the group level average correlation coefficients in each time window across all trials.
    #
    #     Notes
    #     -----
    #     3 figures, GROUP LEVEL DATA
    #     x-axis time window.
    #     Figure output as PDF in figure folder.
    #     """
    #     dvs = ['model_i', 'model_H', 'model_D']
    #     ylabels = ['r', 'r', 'r']
    #     xlabel = 'Time window'
    #     xticklabels = ['Early','Late']
    #     colors = ['teal', 'orange', 'purple']
    #     bar_width = 0.7
    #     xind = np.arange(len(xticklabels))
    #     ylim = [-0.3, 0.3]
    #
    #     for dvi, model_dv in enumerate(dvs):
    #         # single figure
    #         fig = plt.figure(figsize=(2,2))
    #         ax = fig.add_subplot(111)
    #
    #         DFIN = pd.read_csv(os.path.join(self.jasp_folder, '{}_correlation_phasic_{}_{}.csv'.format(self.exp, 'all_trials', model_dv)), float_precision='%.16f')
    #         DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
    #         DFIN.drop(['subject'], axis=1, inplace=True)
    #
    #         # Group average per BIN WINDOW
    #         GROUP = np.mean(DFIN)
    #         SEM = np.true_divide(np.std(DFIN),np.sqrt(len(self.subjects)))
    #         print(GROUP)
    #
    #         ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
    #
    #         # plot bar graph
    #         ax.bar(xind, GROUP, width=bar_width, yerr=SEM, capsize=3, color=colors[dvi], edgecolor='black', ecolor='black')
    #
    #         # individual points, repeated measures connected with lines
    #         for s in np.arange(DFIN.shape[0]):
    #             ax.plot(xind, DFIN.iloc[s,:], linestyle='-', marker='o', markersize=3, fillstyle='full', color='black', alpha=0.2) # marker, line, black
    #
    #         # set figure parameters
    #         ax.set_ylabel(ylabels[dvi])
    #         ax.set_xlabel(xlabel)
    #         # ax.set_ylim(ylim)
    #         ax.set_xticks(xind)
    #         ax.set_xticklabels(xticklabels)
    #
    #         sns.despine(offset=10, trim=True)
    #         plt.tight_layout()
    #         fig.savefig(os.path.join(self.figure_folder,'{}_correlation_phasic_{}.pdf'.format(self.exp, model_dv)))
    #     print('success: plot_information_phasics')
    #
    #
    # def plot_information_phasics_accuracy_split(self,):
    #     """Plot the average correlation coefficients in each time window split by error vs. correct.
    #
    #     Notes
    #     -----
    #     1 figure, GROUP LEVEL DATA
    #     x-axis time window.
    #     Figure output as PDF in figure folder.
    #     """
    #     dvs = ['model_i', 'model_H', 'model_D']
    #     ylabels = ['r', 'r', 'r']
    #     xlabel = 'Time window'
    #     xticklabels = ['Early','Late']
    #     colors = ['red', 'blue']
    #     bar_width = 0.7
    #     xind = np.arange(len(xticklabels))
    #     ylim = [-0.3, 0.3]
    #
    #     for dvi, model_dv in enumerate(dvs):
    #         # single figure
    #         fig = plt.figure(figsize=(2,2))
    #         ax = fig.add_subplot(111) # 1 subplot per bin windo
    #
    #         for c, cond in enumerate(['error', 'correct']):
    #
    #             df_in = pd.read_csv(os.path.join(self.jasp_folder, '{}_correlation_phasic_{}_{}.csv'.format(self.exp, cond, model_dv)), float_precision='%.16f')
    #             df_in = df_in.loc[:, ~df_in.columns.str.contains('^Unnamed')] # drop all unnamed columns
    #             df_in.drop(['subject'], axis=1, inplace=True)
    #
    #             SEM = np.true_divide(np.std(df_in),np.sqrt(len(self.subjects)))
    #
    #             # plot bar graph
    #             ax.errorbar(xind, np.mean(df_in), yerr=SEM,  marker='o', markersize=3, fmt='-', elinewidth=1, label=cond, capsize=3, color=colors[c], alpha=1)
    #
    #         ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
    #
    #         # set figure parameters
    #         ax.set_ylabel(ylabels[dvi])
    #         ax.set_xlabel(xlabel)
    #         # ax.set_ylim(ylim)
    #         ax.set_xticks(xind)
    #         ax.set_xticklabels(xticklabels)
    #         ax.set_title(model_dv)
    #         # ax.legend()
    #
    #         sns.despine(offset=10, trim=True)
    #         plt.tight_layout()
    #         fig.savefig(os.path.join(self.figure_folder,'{}_correlation_phasic_accuracy_split_{}.pdf'.format(self.exp, model_dv)))
    #     print('success: plot_information_phasics_accuracy_split')




 
    # def dataframe_evoked_correlation(self):
    #     """Partial correlation of theoretic variables with other variables removed.
    #
    #     Notes
    #     -----
    #     Drop omission trials (in subject loop).
    #     Output in dataframe folder.
    #     """
    #     DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
    #     DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns
    #
    #     ivs = ['model_i', 'model_H', 'model_D']
    #
    #     pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas
    #     df_out = pd.DataFrame() # timepoints x subjects
    #
    #     for t,time_locked in enumerate(self.time_locked):
    #
    #         for cond in ['correct', 'error', 'all_trials']:
    #
    #             # Loop through IVs
    #             for i,iv in enumerate(ivs):
    #
    #                 # loop subjects
    #                 for s, subj in enumerate(self.subjects):
    #
    #                     this_subj = int(''.join(filter(str.isdigit, subj)))
    #                     # get current subject's data only
    #
    #                     SBEHAV = DF[DF['subject']==this_subj].reset_index()
    #                     SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked))), float_precision='%.16f')
    #                     SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
    #
    #                     # merge behavioral and evoked dataframes so we can group by conditions
    #                     SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
    #
    #                     #### DROP OMISSIONS HERE ####
    #                     SDATA = SDATA[SDATA['drop_trial'] == 0] # drop outliers based on RT
    #                     #############################
    #
    #                     evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
    #
    #                     save_timepoint_r = []
    #
    #                     # loop timepoints, regress
    #                     for col in evoked_cols:
    #
    #                         # First remove other ivs from current iv with linear regression
    #                         remove_ivs = [i for i in ivs if not i == iv]
    #
    #                         # model: iv1 ~ constant + iv2 + iv3, take residuals into correlation with pupil
    #                         Y = np.array(SDATA[iv]) # current iv
    #                         X = SDATA[remove_ivs]
    #
    #                         # remove all missing cases
    #                         X['Y'] = Y
    #                         X['pupil'] = np.array(SDATA[col]) # pupil
    #                         X.dropna(subset=X.columns.values, inplace=True)
    #                         Y = X['Y']
    #                         y = X['pupil']
    #                         X.drop(columns=['Y', 'pupil'], inplace=True)
    #
    #                         if cond == 'correct':
    #                             mask = SDATA['correct']==True
    #                             X = X[mask] # ivs to partial out
    #                             Y = Y[mask] # current iv
    #                             y = y[mask]
    #                         elif cond == 'error':
    #                             mask = SDATA['correct']==False
    #                             X = X[mask]
    #                             Y = Y[mask]
    #                             y = y[mask]
    #
    #                         X = sm.add_constant(X)
    #
    #                         # partial correlation via ordinary least squares linear regression, get residuals
    #                         model = sm.OLS(Y, X, missing='drop')
    #                         results = model.fit()
    #                         x = results.resid # residuals of theoretic variable regression
    #
    #                         r, pval = sp.stats.pearsonr(x, y)
    #
    #                         save_timepoint_r.append(self.fisher_transform(r))
    #
    #                     # add column for each subject with timepoints as rows
    #                     df_out[subj] = np.array(save_timepoint_r)
    #                     df_out[subj] = df_out[subj].apply(lambda x: '%.16f' % x) # remove scientific notation from df
    #
    #                 # save output file
    #                 df_out.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_{}_{}.csv'.format(self.exp, time_locked, cond, iv)), float_format='%.16f')
    #     print('success: dataframe_evoked_regression')



    # def confound_rt_pupil(self,):
    #     """Compute single-trial correlation between RT and pupil_dvs, subject and group level
    #
    #     Notes
    #     -----
    #     Plots a random subject.
    #     """
    #     dvs = ['pupil_feed_locked_t1', 'pupil_feed_locked_t2', 'pupil_baseline_feed_locked']
    #     DFOUT = pd.DataFrame() # subjects x pupil_dv (fischer z-transformed correlation coefficients)
    #     for sp, pupil_dv in enumerate(dvs):
    #
    #         DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
    #
    #         ############################
    #         # drop outliers and missing trials
    #         DF = DF[DF['drop_trial']==0]
    #         ############################
    #
    #         plot_subject = np.random.randint(0, len(self.subjects)) # plot random subject
    #         save_coeff = []
    #         for s, subj in enumerate(np.unique(DF['subject'])):
    #             this_df = DF[DF['subject']==subj].copy(deep=False)
    #
    #             x = np.array(this_df['RT'])
    #             y = np.array(this_df[pupil_dv])
    #             r,pval = stats.pearsonr(x,y)
    #             save_coeff.append(self.fisher_transform(r))
    #
    #             if s==plot_subject:  # plot one random subject
    #                 fig = plt.figure(figsize=(2,2))
    #                 ax = fig.add_subplot(111)
    #                 ax.plot(x, y, 'o', markersize=3, color='grey') # marker, line, black
    #                 m, b = np.polyfit(x, y, 1)
    #                 ax.plot(x, m*x+b, color='grey',alpha=1)
    #                 # set figure parameters
    #                 ax.set_title('subject={}, r = {}, p = {}'.format(subj, np.round(r,2),np.round(pval,3)))
    #                 ax.set_ylabel(pupil_dv)
    #                 ax.set_xlabel('RT (s)')
    #                 # ax.legend()
    #                 plt.tight_layout()
    #                 fig.savefig(os.path.join(self.figure_folder,'{}_confound_RT_{}.pdf'.format(self.exp, pupil_dv)))
    #         DFOUT[pupil_dv] = np.array(save_coeff)
    #     DFOUT.to_csv(os.path.join(self.jasp_folder, '{}_confound_RT.csv'.format(self.exp)))
    #     print('success: confound_rt_pupil')
    #
    #
    # def confound_baseline_phasic(self,):
    #     """Compute single-trial correlation between feedback_baseline and phasic t1 and t2.
    #
    #     Notes
    #     -----
    #     Plots a random subject.
    #     """
    #     dvs = ['pupil_feed_locked_t1', 'pupil_feed_locked_t2']
    #     DFOUT = pd.DataFrame() # subjects x pupil_dv (fischer z-transformed correlation coefficients)
    #     for sp, pupil_dv in enumerate(dvs):
    #
    #         DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
    #
    #         ############################
    #         # drop outliers and missing trials
    #         DF = DF[DF['drop_trial']==0]
    #         ############################
    #
    #         plot_subject = np.random.randint(0, len(self.subjects)) # plot random subject
    #         save_coeff = []
    #         for s, subj in enumerate(np.unique(DF['subject'])):
    #             this_df = DF[DF['subject']==subj].copy(deep=False)
    #
    #             x = np.array(this_df['pupil_baseline_feed_locked'])
    #             y = np.array(this_df[pupil_dv])
    #             r,pval = stats.pearsonr(x,y)
    #             save_coeff.append(self.fisher_transform(r))
    #
    #             if s==plot_subject:  # plot one random subject
    #                 fig = plt.figure(figsize=(2,2))
    #                 ax = fig.add_subplot(111)
    #                 ax.plot(x, y, 'o', markersize=3, color='grey') # marker, line, black
    #                 m, b = np.polyfit(x, y, 1)
    #                 ax.plot(x, m*x+b, color='grey',alpha=1)
    #                 # set figure parameters
    #                 ax.set_title('subject={}, r = {}, p = {}'.format(subj, np.round(r,2),np.round(pval,3)))
    #                 ax.set_ylabel(pupil_dv)
    #                 ax.set_xlabel('pupil_baseline_feed_locked')
    #                 # ax.legend()
    #                 plt.tight_layout()
    #                 fig.savefig(os.path.join(self.figure_folder,'{}_confound_baseline_phasic_{}.pdf'.format(self.exp, pupil_dv)))
    #         DFOUT[pupil_dv] = np.array(save_coeff)
    #     DFOUT.to_csv(os.path.join(self.jasp_folder, '{}_confound_baseline_phasic.csv'.format(self.exp)))
    #     print('success: confound_baseline_phasic')
    
        