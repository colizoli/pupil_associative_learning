#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Pupil dilation offers a time-window in prediction error

Data set #2 Odd-ball task (independent learning phase) - Higher Level Functions
Python code O.Colizoli 2023 (olympia.colizoli@donders.ru.nl)
Python 3.6

================================================
"""

import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython import embed as shell # for debugging


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
    jasp_folder : str
        Path to the jasp directory for stats
    """
    
    def __init__(self, subjects, experiment_name, project_directory):        
        """Constructor method
        """
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
        """Combine behavior and phasic pupil dataframes of all subjects into a single large dataframe. 
        
        Notes
        -----
        Flag missing trials from concantenated dataframe.
        Output in dataframe folder: task-experiment_name_subjects.csv
        """
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
        """Average the DVs per subject per condition of interest. 

        Notes
        -----
        Save separate dataframes for the different combinations of factors, in jasp folders for statistical testing.
        Drop missed trials.
        """
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
        """Plot the group level means of accuracy and RT per odd-ball condition.

        Notes
        -----
        GROUP LEVEL DATA
        x-axis is odd-ball conditions.
        Figure output as PDF in figure folder.
        """
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
            
            ax = fig.add_subplot(int(len(ylabels)), 1, int(subplot_counter),  ) # 1 subplot per bin window
            ax.set_box_aspect(1)
            
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
                ax.plot(xind, s, linestyle='-', marker='o', markersize=3, fillstyle='full', color='black', alpha=0.1) # marker, line, black
                
            # set figure parameters
            ax.set_title(ylabels[dvi]) # for consistent formatting
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            # if dv == 'correct':
            #     ax.set_ylim([0.0,1.2])
            #     ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.2))
            # else:
            #     ax.set_ylim([0.2,1]) #RT
            #     ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.2))

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_{}_behavior.pdf'.format(self.exp,factor)))
        print('success: plot_behav')
        
        
    def calculate_actual_frequencies(self):
        """Calculate the actual frequencies of the pairs presented during the oddball task.

        Notes
        -----
        10 trials x 10 reps = 100 trials PER letter PER frequency condition in the training task.
        """
        letter_trials = 100 # how many trials per letter in the training task
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)))
        DF = DF[DF['oddball']==0] # remove odd-balls
        DF['for_counts'] = np.repeat(1,len(DF)) # to count something
        
        print(DF.groupby(['subject','frequency','letter'])['for_counts'].count())
        
        G = DF.groupby(['subject','frequency','letter','r'])['for_counts'].count() # group by letter and R-code of RGB
        G = pd.DataFrame(G)
        
        # calculate as percentage per letter
        G['actual_frequency'] = np.true_divide(G['for_counts'],letter_trials)*100
        
        # split data into bins/quartiles based on actual frequencies
        for b in [2,3,4]:
            # group into frequency bins with equal numeric edges, but unequal data partitions
            G['frequency_bin_{}'.format(b)]=pd.cut(G['actual_frequency'], bins=b) # 3 bins of equal width, but number of elements differ
            print(G['frequency_bin_{}'.format(b)].value_counts())
            
            # group into equal-sized partitions of data, but unequal bin widths
            G['frequency_qcut_{}'.format(b)]=pd.qcut(G['actual_frequency'], q=b)
            print(G['frequency_qcut_{}'.format(b)].value_counts())

        G.to_csv(os.path.join(self.dataframe_folder,'{}_actual_frequencies.csv'.format(self.exp)))

        print('success: calculate_actual_frequencies')
        
    
    def information_theory_code_stimuli(self, df_in):
        
        # df_in = pd.read_csv(fn_in)
        
        # make new column to give each letter-color combination a unique identifier (1 - 36)        
        mapping = [
            # KEEP ORIGINAL MAPPINGS TO SEE 'FLIP'
            (df_in['letter'] == 'A') & (df_in['r'] == 76) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'A') & (df_in['r'] == 157) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'A') & (df_in['r'] == 0) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'A') & (df_in['r'] == 3) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'A') & (df_in['r'] == 138) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'A') & (df_in['r'] == 75) & (df_in['oddball'] == 0), 
            #
            (df_in['letter'] == 'D') & (df_in['r'] == 76) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'D') & (df_in['r'] == 157) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'D') & (df_in['r'] == 0) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'D') & (df_in['r'] == 3) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'D') & (df_in['r'] == 138) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'D') & (df_in['r'] == 75) & (df_in['oddball'] == 0), 
            #
            (df_in['letter'] == 'I') & (df_in['r'] == 76) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'I') & (df_in['r'] == 157) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'I') & (df_in['r'] == 0) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'I') & (df_in['r'] == 3) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'I') & (df_in['r'] == 138) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'I') & (df_in['r'] == 75) & (df_in['oddball'] == 0), 
            #
            (df_in['letter'] == 'O') & (df_in['r'] == 76) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'O') & (df_in['r'] == 157) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'O') & (df_in['r'] == 0) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'O') & (df_in['r'] == 3) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'O') & (df_in['r'] == 138) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'O') & (df_in['r'] == 75) & (df_in['oddball'] == 0), 
            #
            (df_in['letter'] == 'R') & (df_in['r'] == 76) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'R') & (df_in['r'] == 157) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'R') & (df_in['r'] == 0) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'R') & (df_in['r'] == 3) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'R') & (df_in['r'] == 138) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'R') & (df_in['r'] == 75) & (df_in['oddball'] == 0), 
            #
            (df_in['letter'] == 'T') & (df_in['r'] == 76) & (df_in['oddball'] == 0),  
            (df_in['letter'] == 'T') & (df_in['r'] == 157) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'T') & (df_in['r'] == 0) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'T') & (df_in['r'] == 3) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'T') & (df_in['r'] == 138) & (df_in['oddball'] == 0), 
            (df_in['letter'] == 'T') & (df_in['r'] == 75) & (df_in['oddball'] == 0), 
            
            ]
        
        elements = np.arange(36)  # also elements is the same as priors (start with 0 so they can be indexed by element)
        df_in['letter_color_pair'] = np.select(mapping, elements)
        print('success: information_theory_code_stimuli')
        
        return df_in
        
        
    def idt_model(self, df, df_data_column, elements):
        
        data = np.array(df[df_data_column])
    
        # initialize output variables for current subject
        model_e = [] # trial sequence
        model_P = [] # probabilities of all elements 
        model_p = [] # probability of current element 
        model_i = [] # surprise
        
        # loop trials
        for t in np.arange(df.shape[0]):
            vector = data[:t+1] #  trial number starts at 0, all the targets that have been seen so far
            
            # if it's the first trial, our expectations are based only on the prior (values)
            if t < 1: 
                alpha1 = np.ones(len(elements)) # np.sum(alpha) == len(elements), flat prior
                
            # Updated estimated probabilities (posterior)
            p = []
            for k in elements:
                # +1 because in the prior there is one element of the same type; +4 because in the prior there are 4 elements
                # The influence of the prior should be sampled by a distribution or
                # set to a certain value based on Kidd et al. (2012, 2014)
                p.append((np.sum(vector == k) + alpha1[k]) / (len(vector) + len(alpha1)))       

            model_e.append(vector[-1])  # element in current trial = last element in the vector
            model_P.append(p)           # probability of all elements in NEXT trial
            model_p.append(p[vector[-1]]) # probability of element in NEXT trial
            
            I = -np.log2(p)     # complexity of every event (each cue_target_pair is a potential event)
            i = I[vector[-1]]   # surprise of the current event (last element in vector)
            model_i.append(i)
         
        return [model_i, model_p, model_P[-1]] # return only the last array
        
        
    def information_theory_estimates(self, ):
        # https://github.com/FrancescPoli/eye_processing/blob/master/ITDmodel.m
        
        fn_in = os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp))
        
        # drop oddball trials
        df_in = pd.read_csv(fn_in)
        df_in = df_in.loc[:, ~df_in.columns.str.contains('^Unnamed')]
        # sort by subjects then trial_counter in ascending order
        df_in.sort_values(by=['subject', 'trial_num'], ascending=True, inplace=True)
        
        df_in = df_in[df_in['oddball']==0]
        
        # df_in.to_csv(fn_out)
        # fn_in = fn_out
        
        df_in = self.information_theory_code_stimuli(df_in) # code stimuli based on predictions and based on targets
        
        df_out = pd.DataFrame() # add probabilities into the oddball dataframe for sanity checks
        df_prob_out = pd.DataFrame() # prior dataframe, last probabilities of all elements saved
        
        elements = np.arange(36)
        
        # loop subjects
        for s,subj in enumerate(self.subjects):
            
            this_subj = int(''.join(filter(str.isdigit, subj))) # get number of subject only
            # get current subjects data only
            this_df = df_in[df_in['subject']==this_subj].copy()
            print(subj)
            
            # the input to the model is the trial sequence = the order of letter-color pair for each participant
            [model_i, model_p, model_P] = self.idt_model(this_df, 'letter_color_pair', elements)
            # add to priors dataframe
            df_prob_out['{}'.format(this_subj)] = np.array(model_P)
            
            print(subj)
                        
            # get probabilities into df_in to check based on frequency condition
            this_df['model_p'] = np.array(model_p)
            this_df['model_i'] = np.array(model_i)
            
            df_out = pd.concat([df_out, this_df])    # add current subject df to larger df

        # save only priors (elements x subjects)
        df_prob_out.to_csv(os.path.join(self.dataframe_folder,'{}_subjects_priors.csv'.format(self.exp)), float_format='%.8f')
        
        # save all trials, except oddballs, with probability per trial
        df_out.to_csv(os.path.join(self.dataframe_folder,'{}_subjects_information_theory.csv'.format(self.exp))) # overwrite subjects dataframe
        print('success: information_theory_estimates')


    def plot_information_frequency(self,):
        """Plot the model parameteres by frequency condition

        Notes
        -----
        GROUP LEVEL DATA
        x-axis is frequency conditions.
        Figure output as PDF in figure folder.
        """
        #######################
        # Frequency
        #######################
        
        priors = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects_priors.csv'.format(self.exp)), float_precision='%.16f') 
        priors.reset_index(inplace=True)
        priors = priors.rename(columns={'index': 'letter_color_pair'})
        
        DFIN = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects_information_theory.csv'.format(self.exp)), float_precision='%.16f') # overwrite subjects dataframe
        DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
        
        # loop subjects and merge priors with frequency conditions
        save_priors = pd.DataFrame()
        for s,subj in enumerate(self.subjects):

            this_subj = int(''.join(filter(str.isdigit, subj)))
            this_df = DFIN[DFIN['subject']==this_subj].copy()
            
            # get frequency conditions for this subject
            this_freq = this_df[['frequency','letter_color_pair']]
            # get priors for this subject
            
            this_priors = priors[[str(this_subj), 'letter_color_pair']]
            
            # merge priors and frequency on letter-color pair
            m = this_freq.merge(this_priors,how='inner',on=['letter_color_pair'])
            
            # Group based on frequency condition
            group_priors = pd.DataFrame(m.groupby(['frequency'])[str(this_subj)].mean())
            save_priors = pd.concat([save_priors, group_priors], axis=1)
                
        # transpose and save data frame
        save_priors = save_priors.T
        save_priors.to_csv(os.path.join(self.jasp_folder,'{}_subjects_priors_by_frequency.csv'.format(self.exp)), float_format='%.16f')
        
        # FIGURE 
        dvs = ['model_p', 'model_i']
        ylabels = ['Probability', 'Surprise']
        factor = 'frequency'
        xlabel = 'Letter-color frequency'
        xticklabels = ['20%','40%','80%'] 
        bar_width = 0.7
        xind = np.arange(len(xticklabels))
    
        colors = ['lightblue', 'teal']
        
        fig = plt.figure(figsize=(2.67,2))
        ax = fig.add_subplot(1, 2, 1) # 1 subplot per bin window
        
        # probability
        GROUP = np.mean(save_priors)
        SEM = np.true_divide(GROUP,np.sqrt(len(self.subjects)))
        print(GROUP)                  
        
        GROUP = np.array(GROUP)
        SEM = np.array(SEM)
                           
        # plot bar graph
        for xi in np.arange(len(GROUP)):
            ax.bar(xind[xi], GROUP[xi], width=bar_width, yerr=SEM[xi], capsize=3, color=colors[0], edgecolor='black', ecolor='black')
        
        # set figure parameters
        ax.set_title('Oddball Task') # repeat for consistent formatting
        ax.set_ylabel(ylabels[0])
        ax.set_xlabel(xlabel)
        ax.set_xticks(xind)
        ax.set_xticklabels(xticklabels)
        
        # surprise
        ax = fig.add_subplot(1, 2, 2) # 1 subplot per bin window
        
        i = -np.log2(save_priors) 
        GROUP = np.mean(i)
        SEM = np.true_divide(GROUP,np.sqrt(len(self.subjects)))
        print(GROUP)
        
        GROUP = np.array(GROUP)
        SEM = np.array(SEM)
        
        # plot bar graph
        for xi in np.arange(len(GROUP)):
            ax.bar(xind[xi], GROUP[xi], width=bar_width, yerr=SEM[xi], capsize=3, color=colors[1], edgecolor='black', ecolor='black')
            
        # set figure parameters
        ax.set_title('Oddball Task') # repeat for consistent formatting
        ax.set_ylabel(ylabels[1])
        ax.set_xlabel(xlabel)
        ax.set_xticks(xind)
        ax.set_xticklabels(xticklabels)

        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_information_frequency.pdf'.format(self.exp)))
        print('success: plot_information_frequency')
   