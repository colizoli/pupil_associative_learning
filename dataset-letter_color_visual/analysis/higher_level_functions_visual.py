#!/usr/bin/env python
# encoding: utf-8
"""
Letter-color associations formed through statistical learning in the visual modality
"Letter-color visual 2AFC task" for short
Python code by O.Colizoli 2022
Python 3.6
"""

import os, sys, datetime
import numpy as np
import scipy as sp
import scipy.stats as stats
import statsmodels.formula.api as sm
from scipy.signal import decimate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
import re

#conda install -c conda-forge/label/gcc7 mne
from copy import deepcopy
import itertools
# import pingouin as pg # stats package
# from pingouin import pairwise_ttests

from IPython import embed as shell


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 1, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 1, 
    'ytick.major.width': 1,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

############################################
# Define parameters
############################################

class higherLevel(object):
    def __init__(self, subjects, experiment_name, project_directory, sample_rate, time_locked, pupil_step_lim, baseline_window, pupil_time_of_interest):        
        self.subjects           = subjects
        self.exp                = experiment_name
        self.project_directory  = project_directory
        self.figure_folder      = os.path.join(project_directory, 'figures')
        self.dataframe_folder   = os.path.join(project_directory, 'data_frames')
        self.trial_bin_folder   = os.path.join(self.dataframe_folder,'trial_bins_pupil') # for average pupil in different trial bin windows
        self.jasp_folder        = os.path.join(self.dataframe_folder,'jasp') # for dataframes to input into JASP
        
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
            
        if not os.path.isdir(self.dataframe_folder):
            os.mkdir(self.dataframe_folder)
        
        if not os.path.isdir(self.trial_bin_folder):
            os.mkdir(self.trial_bin_folder)
            
        if not os.path.isdir(self.jasp_folder):
            os.mkdir(self.jasp_folder)
            
        ##############################    
        # Pupil time series information:
        ##############################
        self.sample_rate        = sample_rate
        self.time_locked        = time_locked
        self.pupil_step_lim     = pupil_step_lim                
        self.baseline_window    = baseline_window              
        self.pupil_time_of_interest = pupil_time_of_interest
        self.downsample_rate    = 20 # 20 Hz
        self.downsample_factor  = self.sample_rate / self.downsample_rate
    
    def tsplot(self, ax, data, alpha_fill=0.2,alpha_line=1, **kw):
        # replacing seaborn tsplot
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        cis = self.bootstrap(data)
        ax.fill_between(x,cis[0],cis[1],alpha=alpha_fill,**kw) # debug double label!
        ax.plot(x,est,alpha=alpha_line,**kw)
        ax.margins(x=0)
    
    
    def bootstrap(self, data, n_boot=10000, ci=68):
        # bootstrap confidence interval for new tsplot
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
        # permutation-based cluster correction on time courses, then plots the stats as a bar in yloc
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
        # computes phasic pupil in selected time window per trial
        # adds phasics to behavioral data frame
        # loop through subjects' log files
        
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
        # concatenates all subjects into a single dataframe
        # flags missing trials from concantenated dataframe
        # counts missing trials and outliers
        
        # output all subjects' data
        DF = pd.DataFrame()
        
        files = []
        for s,subj in enumerate(self.subjects):
            this_dir = os.path.join(self.project_directory,subj,'beh')
            for i in os.listdir(this_dir):
                if os.path.isfile(os.path.join(this_dir,i)) and (self.exp in i) and ('beh' in i):
                    files.append(os.path.join(this_dir,i))
                    
        counter = 0    
        for f in files:
            this_data = pd.read_csv(f)
            this_data = this_data.loc[:, ~this_data.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            ###############################
            this_data['missing'] = this_data['button']=='missing'
            this_data['drop_trial'] = np.array(this_data['missing']) #logical or
            
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
        # save whole dataframe with all subjects
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        #####################
        print('success: higherlevel_dataframe')
        
    def average_conditions(self):
        # averages the phasic pupil per subject PER CONDITION 
        # saves separate dataframes for the different combinations of factors
                
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject','trial_num'],inplace=True)
        DF.reset_index()
        
        ############################
        # drop outliers and missing trials
        DF = DF[DF['drop_trial']==0]
        ############################
        
        #interaction accuracy and frequency
        for pupil_dv in ['pupil_feed_locked_t1','pupil_feed_locked_t2','RT']: #interaction accuracy and frequency
            '''
            ######## CORRECT x FREQUENCY ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject','correct','frequency'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct*frequency_{}.csv'.format(self.exp,pupil_dv))) # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['frequency','correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct*frequency_{}_rmanova.csv'.format(self.exp,pupil_dv))) # for stats
        
        '''
        ######## FREQUENCY ########
        '''
        for pupil_dv in ['correct','RT']: # mean accuracy
            DFOUT = DF.groupby(['subject','frequency'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_frequency_{}.csv'.format(self.exp,pupil_dv))) # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['frequency']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp,pupil_dv))) # for stats
        print('success: average_conditions')

    def plot_phasic_pupil_pe(self,):
        # Phasic pupil feed_locked
        # GROUP LEVEL DATA
        # separate lines for correct, x-axis is mapping conditions
        ylim = [ 
            [-1.5,6.5], # t1
            [-3.25,2.25], # t2
        ]
        tick_spacer = [1.5,1]
        
        dvs = ['pupil_feed_locked_t1','pupil_feed_locked_t2',]
        ylabels = ['Pupil response\n(% signal change)','Pupil response\n(% signal change)']
        factor = ['frequency','correct']
        xlabel = 'Letter-color frequency'
        xticklabels = ['20%','40%','80%'] 
        labels = ['Error','Correct']
        colors = ['red','blue'] 
        
        xind = np.arange(len(xticklabels))
        dot_offset = [0.05,-0.05]
        
        fig = plt.figure(figsize=(2,2*len(ylabels)))
        
        subplot_counter = 1
        
        for dvi,pupil_dv in enumerate(dvs):

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_correct*frequency_{}.csv'.format(self.exp,pupil_dv)))
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby(factor)[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
            
            ax = fig.add_subplot(len(ylabels),1,subplot_counter)
            subplot_counter += 1
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
            # # plot line graph
            for x in[0,1]: # split by error, correct
                D = GROUP[GROUP['correct']==x]
                print(D)
                ax.errorbar(xind,np.array(D['mean']),yerr=np.array(D['sem']),fmt='-',elinewidth=1,label=labels[x],capsize=0, color=colors[x], alpha=1)

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
        fig.savefig(os.path.join(self.figure_folder,'{}_correct*frequency_lines.pdf'.format(self.exp)))
        print('success: plot_phasic_pupil_pe')
        
    def plot_behavior(self,):
        # plots the group level means of accuracy and RT per mapping condition
        # whole figure, 2 subplots
        
        #######################
        # Frequency
        #######################
        dvs = ['correct','RT']
        ylabels = ['Accuracy', 'RT (s)']
        factor = 'frequency'
        xlabel = 'Letter-color frequency'
        xticklabels = ['20%','40%','80%'] 
        
        bar_width = 0.7
        xind = np.arange(len(xticklabels))
        
        fig = plt.figure(figsize=(2,2*len(ylabels)))
        subplot_counter = 1
        
        for dvi,pupil_dv in enumerate(dvs):

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_{}_{}.csv'.format(self.exp,factor,pupil_dv)))
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby([factor])[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
            
            ax = fig.add_subplot(int(len(ylabels)),1,int(subplot_counter)) # 1 subplot per bin window

            subplot_counter += 1
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
                       
            # plot bar graph
            for xi,x in enumerate(GROUP[factor]):
                # ax.bar(xind[xi],np.array(GROUP['mean'][xi]), width=bar_width, yerr=np.array(GROUP['sem'][xi]), color='blue', alpha=alphas[xi], edgecolor='white', ecolor='black')
                ax.bar(xind[xi],np.array(GROUP['mean'][xi]), width=bar_width, yerr=np.array(GROUP['sem'][xi]), capsize=3, color=(0,0,0,0), edgecolor='black', ecolor='black')
                
            # individual points, repeated measures connected with lines
            DFIN = DFIN.groupby(['subject',factor])[pupil_dv].mean() # hack for unstacking to work
            DFIN = DFIN.unstack(factor)
            for s in np.array(DFIN):
                ax.plot(xind, s, linestyle='-',marker='o', markersize=3,fillstyle='full',color='black',alpha=0.05) # marker, line, black
                
            # set figure parameters
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
        fig.savefig(os.path.join(self.figure_folder,'{}_{}_behavior.pdf'.format(self.exp,factor)))
    
        print('success: plot_behav')
    
    def dataframe_evoked_pupil_higher(self):
        # Evoked pupil responses, split by factors and save as higher level dataframe
        # Need to combine evoked files with behavioral data frame, looping through subjects
        # DROP OMISSIONS (in subject loop)
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        csv_names = deepcopy(['subject','correct','correct*frequency'])
        factors = [['subject'],['correct'],['correct','frequency']]
        
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
        # plots evoked pupil 2 subplits
        # plots the group level mean for target_locked
        # plots the group level accuracy x mapping interaction for target_locked

        ylim_feed = [-4,15]
        tick_spacer = 3
        
        fig = plt.figure(figsize=(6,2))
        #######################
        # FEEDBACK MEAN RESPONSE
        #######################
        t = 0
        time_locked = 'feed_locked'
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax = fig.add_subplot(131)
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

        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
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
        
        #######################
        # CORRECT
        #######################
        t = 0
        time_locked = 'feed_locked'
        csv_name = 'correct'
        factor = 'correct'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax = fig.add_subplot(132)
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

        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
        
        #######################
        # CORRECT x FREQUENCY
        #######################
        t = 0
        time_locked = 'feed_locked'
        csv_name = 'correct*frequency'
        factor = ['correct','frequency']
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax = fig.add_subplot(133)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,csv_name)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        ########
        # make unique labels for each of the 4 conditions
        conditions = [
            (COND['correct'] == 0) & (COND['frequency'] == 80), # Easy Error 1
            (COND['correct'] == 1) & (COND['frequency'] == 80), # Easy Correct 2
            (COND['correct'] == 0) & (COND['frequency'] == 20), # Hard Error 3
            (COND['correct'] == 1) & (COND['frequency'] == 20), # Hard Correct 4
            (COND['correct'] == 0) & (COND['frequency'] == 40), # Medium Error 5 # coded like this to keep in order with other experiments
            (COND['correct'] == 1) & (COND['frequency'] == 40), # Medium Correct 6
            ]
        values = [1,2,3,4,5,6]
        conditions = np.select(conditions, values) # don't add as column to time series otherwise it gets plotted
        ########
                    
        xticklabels = ['Error 80%','Correct 80%','Error 20%','Correct 20%','Error 40%','Correct 40%']
        colorsts = ['r','b','r','b','r','b']
        alpha_fills = [0.2,0.2,0.1,0.1,0.15,.15] # fill
        alpha_lines = [1,1,0.3,0.3,0.6,0.6]
        save_conds = []
        # plot time series
        
        for i,x in enumerate(values):
            TS = COND[conditions==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        # pe_interaction = (save_conds[0]-save_conds[1]) - (save_conds[2]-save_conds[3])
        # self.cluster_sig_bar_1samp(array=pe_interaction, x=pd.Series(range(pe_interaction.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            
        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        # ax.legend(loc='best')
                
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked.pdf'.format(self.exp)))
        print('success: plot_evoked_pupil')
   