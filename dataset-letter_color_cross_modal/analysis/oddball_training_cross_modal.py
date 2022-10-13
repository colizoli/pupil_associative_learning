#!/usr/bin/env python
# encoding: utf-8
"""
Training (Odd-ball Task) for letter-color associations formed through cross-modal statistical learning
"Training letter-color cross-modal 2AFC task" for short
Python code by O.Colizoli 2022
Python 3.6
"""

import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from IPython import embed as shell # for debugging

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
    def __init__(self, subjects, experiment_name, project_directory):        
        self.subjects           = subjects
        self.exp                = experiment_name
        self.project_directory  = project_directory
        self.figure_folder      = os.path.join(project_directory, 'figures')
        self.dataframe_folder   = os.path.join(project_directory, 'data_frames')
        self.jasp_folder        = os.path.join(self.dataframe_folder,'jasp') # for dataframes to input into JASP
        
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
            
        if not os.path.isdir(self.dataframe_folder):
            os.mkdir(self.dataframe_folder)
            
        if not os.path.isdir(self.jasp_folder):
            os.mkdir(self.jasp_folder)
                    
    def create_subjects_dataframe(self,):
        # creates a single data frame from all subjects
        # flags missed trials
        
        # output all subjects' data
        DF = pd.DataFrame()
        
        files = []
        for s,subj in enumerate(self.subjects):
            this_dir = os.path.join(self.project_directory,subj,'beh') # derivatives folder
            for i in os.listdir(this_dir):
                if os.path.isfile(os.path.join(this_dir,i)) and self.exp in i:
                    files.append(os.path.join(this_dir,i))
    
        counter = 0    
        for f in files:
            this_data = pd.read_csv(f)
            # concatenate all subjects
            DF = pd.concat([DF,this_data],axis=0)
            ###############################
            DF['missing'] = DF['button']=='missing'
            
        # count missing
        missing = DF.groupby(['subject','button'])['button'].count()
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_missing.csv'.format(self.exp)))
        ### print how many outliers
        print('Missing = {}%'.format(np.true_divide(np.sum(DF['missing']),DF.shape[0])*100))
        
        #####################
        # save whole dataframe with all subjects
        DF.drop(['Unnamed: 0'],axis=1,inplace=True)
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        #####################
        print('success: create_subjects_dataframe')
        
    def average_conditions(self):
        # drop missed trials, calculate mean accuracy, and RT grouped by frequency, and subject, then also by blocks
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        
        #####################
        # drop missed trials
        DF = DF[DF['button']!='missing'] 
        #####################
        
        for dv in ['correct','RT']:
            DFOUT = DF.groupby(['subject','oddball'])[dv].mean()
            DFOUT.to_csv(os.path.join(self.dataframe_folder,'{}_oddball_{}.csv'.format(self.exp,dv))) # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['oddball']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_oddball_{}_rmanova.csv'.format(self.exp,dv))) # for stats
        
        print('success: average_conditions')
        
    def plot_behav(self):
        # plots the accuracy and RT during the odd-ball task, 2 subplots
        #######################
        # Oddball
        #######################
        dvs = ['correct','RT']
        ylabels = ['Accuracy', 'RT (s)']
        factor = 'oddball'
        xlabel = 'Odd-ball present'
        xticklabels = ['No','Yes'] 
        
        bar_width = 0.7
        xind = np.arange(len(xticklabels))
        
        fig = plt.figure(figsize=(2,2*len(ylabels)))
        subplot_counter = 1
        
        for dvi,dv in enumerate(dvs):

            DFIN = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_{}.csv'.format(self.exp,factor,dv)))
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby([factor])[dv].agg(['mean','std']).reset_index())
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
            DFIN = DFIN.groupby(['subject',factor])[dv].mean() # hack for unstacking to work
            DFIN = DFIN.unstack(factor)
            for s in np.array(DFIN):
                ax.plot(xind, s, linestyle='-',marker='o', markersize=3,fillstyle='full',color='black',alpha=0.05) # marker, line, black
                
            # set figure parameters
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            if dv == 'correct':
                ax.set_ylim([0.0,1.2])
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.2))
            else:
                ax.set_ylim([0.2,1]) #RT
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.2))

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_{}_behavior.pdf'.format(self.exp,factor)))
        print('success: plot_behav')
        
        
    def calculate_actual_frequencies(self):
        # calculate the actual frequencies of the pairs presented during the oddball task
        # 10 trials x 10 reps = 100 trials PER letter PER frequency condition in the training task
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        DF = DF[DF['oddball']==0] # remove odd-balls
        DF['for_counts'] = np.repeat(1,len(DF)) # to count something
        
        G = DF.groupby(['subject','frequency','letter','r'])['for_counts'].count() # group by letter and R-code of RGB
        G.to_csv(os.path.join(self.dataframe_folder,'{}_actual_frequencies.csv'.format(self.exp)))
        
        print('success: calculate_actual_frequencies')


   