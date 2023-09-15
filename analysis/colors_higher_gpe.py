#!/usr/bin/env python
# encoding: utf-8
"""
Analysis gradient prediction errors - Colors control task, mean responses and higher level analyses
O.Colizoli 2019
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

#conda install -c conda-forge/label/gcc7 mne
from copy import deepcopy
import itertools
# import pingouin as pg # stats package
# from pingouin import pairwise_ttests

from IPython import embed as shell

# import pupil_control # for cluster plotting function

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
    def __init__(self, subjects, experiment_name, project_directory, sample_rate, time_locked, pupil_step_lim, baseline_window, pupil_time_of_interest,colors):        
        self.subjects = subjects
        self.exp = experiment_name
        self.project_directory = project_directory
        self.figure_folder = os.path.join(project_directory, 'figures')
        self.dataframe_folder = os.path.join(project_directory, 'data_frames')
        self.sample_rate = sample_rate
        self.time_locked = time_locked
        self.pupil_step_lim = pupil_step_lim                
        self.baseline_window = baseline_window              
        self.pupil_time_of_interest = pupil_time_of_interest
        self.colors = colors
            
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
            
        if not os.path.isdir(self.dataframe_folder):
            os.mkdir(self.dataframe_folder)
            
        ##############################    
        # Define Conditions of Interest
        ##############################
        self.factors = [
            ['r'], # rgb separate columns, so just use one column for individual colors
        ] 
        self.csv_names = [
            'r',
        ]
        self.anova_dv_type =['mean']

        ##############################    
        # Pupil time series information:
        ##############################
        self.downsample_rate = 20 # 20 Hz
        self.downsample_factor = self.sample_rate / self.downsample_rate 
    
    def higherlevel_dataframe(self,):            
        # output all subjects' data
        # Remove missed trials
        DF = pd.DataFrame()
        
        files = []
        for s,subj in enumerate(self.subjects):
            this_dir = os.path.join(self.project_directory,subj,'log')
            for i in os.listdir(this_dir):
                if os.path.isfile(os.path.join(this_dir,i)) and self.exp in i:
                    files.append(os.path.join(this_dir,i))
    
        counter = 0    
        for f in files:
            this_data = pd.read_csv(f)
            # if this_data.shape[0] == 240: # otherwise it was not completed
            # concatenate all subjects
            DF = pd.concat([DF,this_data],axis=0)
       
        # count missing
        DF['missing'] = DF['RT'].isna()
        missing = DF.groupby(['subject','missing'])['RT'].count()
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_missing.csv'.format(self.exp)))
        
        #####################
        # drop missed trials
        DF = DF[DF['missing']!=True] 
        #####################
        #####################
        # save whole dataframe with all subjects
        DF.drop(['Unnamed: 0'],axis=1,inplace=True)
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        #####################
        print('success: higherlevel_dataframe')
        
    def dataframe_behavior_higher(self):
        # mean accuracy, and RT grouped by frequency, and subject, then also by blocks
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        
        grouped1 = DF.groupby(['subject'])['RT'].mean()
        grouped2 = DF.groupby(['subject','r'])['RT'].mean()
    
        grouped1.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_mean.csv'.format(self.exp)))
        grouped2.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_r.csv'.format(self.exp)))
        
        print('success: dataframe_behavior_higher')
    
    def dataframe_phasic_pupil_higher(self):
        # Phasic pupil, split by self.factors and save as dataframe
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        
        csv_names = deepcopy(self.csv_names)

        for t,time_locked in enumerate(self.time_locked):
            # Loop through conditions                
            for c,cond in enumerate(csv_names):
                # intialize dataframe per condition
                COND = pd.DataFrame()
                g_idx = deepcopy(self.factors)[c]        # need to add subject idx for groupby()
                g_idx.insert(0, 'subject') # get strings not list element
                try:
                    COND = pd.DataFrame(DF.groupby(g_idx)['pupil_'+time_locked,'pupil_baseline_'+time_locked].mean()).reset_index()
                except:
                    shell()
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_phasics_{}.csv'.format(self.exp,time_locked,cond)))
        print('success: dataframe_phasic_pupil_higher')
    
    def dataframe_evoked_pupil_higher(self):
        # Evoked pupil responses, split by self.factors and save as dataframe
        # Need to combine np.array evoked with data frame, looping through subjects
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        
        csv_names = deepcopy(self.csv_names)

        for t,time_locked in enumerate(self.time_locked):
            # Loop through conditions                
            for c,cond in enumerate(csv_names):
                # intialize dataframe per condition
                COND = pd.DataFrame()
                g_idx = deepcopy(self.factors)[c]        # need to add subject idx for groupby()
                g_idx.insert(0, 'subject') # get strings not list element
                for s,subj in enumerate(self.subjects):
                    subj_num = int(re.findall('[0-9]+', subj)[0]) # extract number out of string
                    SBEHAV = DF[DF['subject']==subj_num].reset_index()
                    SPUPIL = pd.DataFrame(np.load(os.path.join(self.project_directory,subj,'processed','{}_{}_{}_pupil_events_basecorr.npy'.format(subj,self.exp,time_locked))))
                    SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
                    SDATA['missing'] = SDATA['RT'].isna()
                    SDATA = SDATA[SDATA['missing']!=True] # drop missed trials
                    df = pd.DataFrame(SDATA.groupby(g_idx)[np.arange(SPUPIL.shape[-1])].mean()).reset_index() # only get kernels out
                    # add to condition dataframe
                    COND = pd.concat([COND,df],join='outer',axis=0) # can also do: this_cond = this_cond.append()  
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,cond)))
            # MEAN RESPONSE for all trials
            df = pd.DataFrame(COND.groupby(['subject'])[np.arange(SPUPIL.shape[-1])].mean()).reset_index()
            df.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_mean_response.csv'.format(self.exp,time_locked)))
        print('success: dataframe_evoked_pupil_higher')
    
    def pairwise_posthoc(self, factors, dv, df, csv_name, exp):
        # Tests all combinations of pairwise posthoc comparisons: paired samples t-tests
        # note pairiwse_ttests not working with list of within subjects factors, so wrote loop
        
        OUT = pd.DataFrame()
        if len(factors) == 2:
            # when multiple, need to select subsets, then run tests
            for i in [0,1]:
                for subf in np.unique(df[factors[i]]): # first factor
                
                    this_df = df[df[factors[i]]==subf]
                    posthocs = pairwise_ttests(dv=dv, within=factors[not i], subject='subject', marginal=True,effsize='cohen',data=this_df)
                    posthocs['comparison'] = np.repeat(subf,len(posthocs))
                    OUT = pd.concat([OUT,posthocs],axis=0)
            OUT.to_csv(os.path.join(self.dataframe_folder, 'anova_output','{}_rm_posthoc_{}_{}.csv'.format(exp, dv, csv_name)))
        elif len(factors) == 1:
            # not working with list of 2 columns
            posthocs = pairwise_ttests(dv=dv, within=factors, subject='subject', marginal=True,effsize='cohen',data=df)
            posthocs.to_csv(os.path.join(self.dataframe_folder, 'anova_output','{}_rm_posthoc_{}_{}.csv'.format(exp,dv, csv_name)))
        print('success: pairwise_posthoc factors = {}, DV = {}'.format(csv_name, dv))
        
    def run_anova_behav(self):
        # Runs a repeated measures ANOVA for all dvs for all conditions 
        # Outputs a CSV file containing statistics 
        # Loads higher level subjects file into dataframe
        # outputs pairwise posthoc comparisons
        # Uses pinguoin package (python >3) 
                
        dvs = ['RT']
        factors = [['r']]
        cond = factors
        # loop over factors, loop through dvs    
        for i,dv in enumerate(dvs):
            # grab the corresponding higher level file
            d = pd.read_csv(os.path.join(self.dataframe_folder,'{}_behavior_{}.csv'.format(self.exp,cond[i][0])))
            aov = d.rm_anova(dv=dv, within=factors[i], subject='subject',detailed=True)
            print(aov)
            aov.to_csv(os.path.join(self.dataframe_folder, 'anova_output','{}_rm_anova_{}_{}.csv'.format(self.exp, dv, cond[i][0])))
            
            # get pairwise post hoc comparisions
            self.pairwise_posthoc(factors[i],dv,d,cond[i][0],self.exp)
        print('success: run_anova_behav')
        
    def run_anova_pupil(self):
        # Runs a repeated measures ANOVA for all dvs for all conditions 
        # Outputs a CSV file containing statistics 
        # Loads higher level subjects file into dataframe
        # Uses pinguoin package (python >3) 
         
        for t,time_locked in enumerate(self.time_locked):
            
            dvs = ['pupil_{}'.format(time_locked),'pupil_baseline_{}'.format(time_locked)]
            FACS = [['r'],['r']]
            cond = FACS
            # loop over factors, loop through dvs    
            for dv in dvs:
                for i,factors in enumerate(FACS):
                    # grab the corresponding higher level file
                    d = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_phasics_{}.csv'.format(self.exp,time_locked,cond[i][0])))
                    aov = d.rm_anova(dv=dv, within=factors, subject='subject',detailed=True)
                    print(aov)
                    aov.to_csv(os.path.join(self.dataframe_folder, 'anova_output','{}_rm_anova_{}_{}.csv'.format(self.exp,dv, cond[i][0])))
            
                    # get pairwise post hoc comparisions
                    self.pairwise_posthoc(factors,dv,d,cond[i][0],self.exp)
        print('success: run_anova_pupil')
        
    def plot_behav(self):
        # plots the group level means of accuracy and RT (self.factors)
        # whole figure, 1 panel
        fig = plt.figure(figsize=(3,3))
        #######################
        # COLOR RT
        #######################
        factor = 'r'
        ax = fig.add_subplot(111)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_behavior_r.csv'.format(self.exp)))
        MEANS = pd.DataFrame(COND.groupby(factor)['RT'].agg(['mean','std']).reset_index())        
        MEANS['sem'] = np.true_divide(MEANS['std'],np.sqrt(len(self.subjects)))
        
        xticklabels = np.unique(COND[factor])
        xind = np.array([0.3,0.35,0.40,0.45,0.5,0.55])
        xlim = [0.25,0.6]
        bar_width = 0.05
        # plot bar graph
        for x in [0,1,2,3,4,5]:
            ax.bar(xind[x],np.array(MEANS['mean'][x]), width=bar_width, yerr=np.array(MEANS['sem'][x]), color=self.colors[x], alpha=1, edgecolor='white', ecolor='black')

        print(MEANS)
        # set figure parameters
        ax.set_xlabel(factor)
        ax.set_ylabel('RT')
        ax.set_xticks(xind)
        ax.set_xticklabels(xticklabels)
        ax.set_ylim([0.25,0.5])
        
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_behavior.pdf'.format(self.exp)))
        print('success: plot_behav')
        
    def plot_phasic_pupil(self):
        # plots the group level means of phasic pupil (self.factors)
        # whole figure, 1 panel
        # for each pupil DV 
        
        for t,time_locked in enumerate(self.time_locked):
            pupil_dvs = ['pupil_'+time_locked,'pupil_baseline_'+time_locked]
        
            for pupil in pupil_dvs:
                fig = plt.figure(figsize=(3,3))
                fig.suptitle(pupil)
                #######################
                # COLOR
                #######################
                factor = 'r'
                ax = fig.add_subplot(111)
                ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
                # Compute means, sems across group
                COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_phasics_{}.csv'.format(self.exp,time_locked,factor)))
                MEANS = pd.DataFrame(COND.groupby(factor)[pupil].agg(['mean','std']).reset_index())
                MEANS['sem'] = np.true_divide(MEANS['std'],np.sqrt(len(self.subjects)))
        
                xticklabels = np.unique(COND[factor])
                xind = np.array([0.3,0.35,0.40,0.45,0.5,0.55])
                xlim = [0.25,0.6]
                bar_width = 0.05
                # plot bar graph
                for x in [0,1,2,3,4,5]:
                    ax.bar(xind[x],np.array(MEANS['mean'][x]), width=bar_width, yerr=np.array(MEANS['sem'][x]), color=self.colors[x], alpha=1, edgecolor='white', ecolor='black')
        
                # set figure parameters
                ax.set_xlabel(factor)
                ax.set_ylabel('Pupil response\n(% signal change)')
                ax.set_xticks(xind)
                ax.set_xticklabels(xticklabels)
                # ax.set_title(time_locked)
        
                # whole figure format
                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder,'{}_{}.pdf'.format(self.exp,pupil)))
        print('success: plot_phasic_pupil')

    def plot_evoked_pupil(self):
        # plots the group level means of evoked pupil (self.factors)
        # whole figure, 6 panels, (top row) feedback mean, frequency, error, (bottom row) each of the 3 frequency x error plots 

        kernel = int((self.pupil_step_lim[1]-self.pupil_step_lim[0])*self.sample_rate) # length of evoked responses
        ci = 68 # confidence intervals bootstrapping
        
        ylim = [-2,9]
        
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[1] - self.pupil_step_lim[0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
        # Shade time of interest in grey, will be different for events
        tw_begin = int(event_onset + (self.pupil_time_of_interest[0]*self.sample_rate))
        tw_end = int(event_onset + (self.pupil_time_of_interest[1]*self.sample_rate))
        
        for t,time_locked in enumerate(self.time_locked):
            fig = plt.figure(figsize=(6,3))
            #######################
            # FEEDBACK MEAN RESPONSE
            #######################
            factor = 'mean_response'
            ax = fig.add_subplot(121)
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
    
            # Compute means, sems across group
            COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,factor)))
            COND.drop(['Unnamed: 0'],axis=1,inplace=True)
            xticklabels = ['mean response']
            colors = ['black']
            alphas = [1]

            # plot time series
            i=0
            TS = np.array(COND.iloc[:,-kernel:]) # index from back to avoid extra unnamed column pandas
            sns.tsplot(TS, condition=xticklabels[i], ci=ci, value='Pupil response\n(% signal change)', color=colors[i], alpha=1, lw=1, ls='-', ax=ax)
            # stats on mean response
            pupil_control.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
        
            # set figure parameters
            ax.axvline(int(abs(self.pupil_step_lim[0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            xticks = [event_onset,mid_point,end_sample]
            ax.set_xticks(xticks)
            ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[1],2),self.pupil_step_lim[1]])
            # ax.set_ylim(ylim)
            ax.set_xlabel('Time from stimulus (s)')
            ax.set_ylabel('Pupil response\n(% signal change)')
            ax.set_title(time_locked)
            ax.legend()
            # compute peak of mean response to center time window around
            m = np.mean(TS,axis=0)
            argm = np.true_divide(np.argmax(m),self.sample_rate) + self.pupil_step_lim[0] # subtract pupil baseline to get timing
            print('mean response = {} peak @ {} seconds'.format(np.max(m),argm))
            # ax.axvline(np.argmax(m), lw=0.25, alpha=0.5, color = 'k')
        
            #######################
            # COLOR
            #######################
            factor = 'r'
            ax = fig.add_subplot(122)
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
    
            # Compute means, sems across group
            COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,factor)))
            COND.drop(['Unnamed: 0'],axis=1,inplace=True)
            xticklabels = np.unique(COND[factor])

            #save_conds = []
            # plot time series
            for i,x in enumerate(np.unique(COND[factor])):
                TS = COND[COND[factor]==x]
                TS = np.array(TS.iloc[:,-kernel:])
                sns.tsplot(TS, condition=xticklabels[i], ci=ci, value='Pupil response\n(% signal change)', color=self.colors[i], alpha=1, lw=1, ls='-', ax=ax)
                #save_conds.append(TS) # for stats
                # stats
                pupil_control.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=i+1, color=self.colors[i], ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

            # set figure parameters
            ax.axvline(int(abs(self.pupil_step_lim[0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            xticks = [event_onset,mid_point,end_sample]
            ax.set_xticks(xticks)
            ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[1],2),self.pupil_step_lim[1]])
            # ax.set_ylim(ylim)
            ax.set_xlabel('Time from stimulus (s)')
            ax.set_ylabel('Pupil response\n(% signal change)')
            ax.set_title(time_locked)
            ax.legend()
    
        
            # whole figure format
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_{}_evoked.pdf'.format(self.exp,time_locked)))
        print('success: plot_evoked_pupil')
    



   