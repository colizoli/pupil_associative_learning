#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Pupil dilation offers a time-window in prediction error

Data set #1 Cue-target orientation 2AFC task - Preprocessing pupil dilation
Python code O.Colizoli 2023 (olympia.colizoli@donders.ru.nl)
Python 3.6

================================================
"""

import os
import pandas as pd
import numpy as np
import scipy as sp
from scipy.signal import butter, filtfilt
import seaborn as sns
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from fir import FIRDeconvolution
import mne
import glm_functions_orientation as glm_functions
from IPython import embed as shell # used for debugging


""" Plotting Format"""
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

pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas


class pupilPreprocess(object):
    """Define a class for the preprocessing of the pupil data.

    Parameters
    ----------
    subject : string
        Subject number.
    edf : string
        The name of the current subject's EDF file containing pupil data.
    project_directory : str
        Path to the derivatives data directory.
    sample_rate : int
        Sampling rate of pupil data in Hertz.
    tw_blinks : int or float
        How many seconds to interpolate before and after blinks.
    mph : int or float 
        Detect peaks that are greater than minimum peak height.
    mpd : int or float 
        Blinks separated by minimum number of samples.
    threshold : int or float 
        Detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors.
    
    Attributes
    ----------
    subject : string
        Subject number.
    edf : string
        The name of the current subject's EDF file containing pupil data.
    project_directory : str
        Path to the derivatives data directory.
    figure_folder : str
        Path to the figure directory.
    sample_rate : int
        Sampling rate of pupil data in Hertz.
    time_window_blinks : int or float
        How many seconds to interpolate before and after blinks.
    mph : int or float 
        Detect peaks that are greater than minimum peak height.
    mpd : int or float 
        Blinks separated by minimum number of samples.
    threshold : int or float 
        Detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors.
    add_base : boolean
        Refers to add baseline pupil back into time series. Needs to be initialized as True, regress_blinks_saccades() will make False if called.
    self.blink_starts : empty array
        place holder for blink starts.
    self.blink_ends : empty array
        place holder for blink ends.
    
    """
    def __init__(self, subject, edf, project_directory, sample_rate, tw_blinks, mph, mpd, threshold):
        """Constructor method"""
        self.subject = str(subject)
        self.alias = edf
        self.project_directory = os.path.join(project_directory,self.subject,'beh') # single-subject directory
        self.figure_folder = os.path.join(project_directory,'figures','preprocessing') # group-level directory for easy inspection
        self.sample_rate = sample_rate
        self.time_window_blinks = tw_blinks
        self.mph = mph # blink detection
        self.mpd = mpd
        self.threshold = threshold
        # self.base needs to be initialized True, regress blinks function makes false if called
        self.add_base = True 
        self.blink_starts = np.empty((0), int) # placeholders
        self.blink_ends = np.empty((0), int)
        
        if not os.path.exists(self.project_directory):
            os.makedirs(self.project_directory)
            
        if not os.path.exists(self.figure_folder):
            os.makedirs(self.figure_folder)
       
    
    def housekeeping(self, experiment_name):
        """Replace 'prediction' with new experiment_name.
        
        Parameters
        ----------
        experiment_name : string
            New experiment name to use.
        """

        # RENAME
        files = os.listdir(self.project_directory)
        for f in files:
            src = os.path.join(self.project_directory, f)
            dst = src.replace("task-prediction", experiment_name)
            try:
                os.rename(src, dst)
            except:
                print('did not rename src: {}'.format(src))
    
    
    def read_trials(self,):
        """Read in the message, markers, and data from the EDF pupil data file.

        Notes
        -------
        Saves the pupil dilation time series in the project directory.
        Extracts messages and pupil from raw text files:
        Messages are in the 'L Raw X [px]' column.
        Pupil is in the 'R Dia X [px]' column.
        """
        usecols = ['Time','Type','L Raw X [px]','R Dia X [px]'] # only get these columns out
        # in source/sub-xx/sub-xx_task-predictions_eye.txt
        EDF = pd.read_csv(os.path.join(self.project_directory, '{}.txt'.format(self.alias)), skiprows=38,delimiter='\t', usecols=usecols)
        
        # EXTRACT MESSAGES
        # get row number of all messages in file, before removing from samples, add 1 to get following sample row index
        self.msgs_markers = pd.DataFrame(EDF[EDF['Type'] == 'MSG']['L Raw X [px]']).reset_index()
        self.msgs_markers.columns=['index','msg'] # rename columns
        # save messages and their timestamps as csv file
        self.msgs_markers.to_csv(os.path.join(self.project_directory, '{}_phases.csv'.format(self.alias)))
        # EXTRACT PUPIL DATA
        self.TS = pd.DataFrame(EDF[EDF['Type'] == 'SMP']['R Dia X [px]']).reset_index() # index here referes to original rows, very important to keep it together with the pupil data
        self.TS.columns=['index','pupil']   # rename columns        
        # columns =['index', 'pupil']
        np.save(os.path.join(self.project_directory, '{}.npy'.format(self.alias)), np.array(self.TS))
        print('{} messages processed'.format(self.subject))
    
    
    def preprocess_pupil(self,):
        """Carries out pupil preprocessing routine.

        Notes
        -------
        Current pupil time course is always 'self.pupil'.
        Steps include: interpolation around blinks based on markers, then based on peaks, bandpass filtering, nuisance regression on blinks/saccades,
        convert to percent signal change (z-score also saved).
        Saves time series at each stage with labels, global variables  (e.g., self.pupil_interp)
        Calls the preprocessing plot for each subject.
        """
        cols1 = ['index', 'pupil'] # before preprocessing
        cols2 = ['index', 'pupil', 'pupil_interp','pupil_bp','pupil_clean','pupil_psc', 'pupil_zscore']        
                
        try:
            self.TS = pd.DataFrame(np.load(os.path.join(self.project_directory, '{}.npy'.format(self.alias))), columns=cols1)
        except:
            self.TS = pd.DataFrame(np.load(os.path.join(self.project_directory, '{}.npy'.format(self.alias))), columns=cols2)
                
        self.pupil_raw = np.array(self.TS['pupil'])
        
        self.pupil = self.pupil_raw         # self.pupil is always current pupil stage     
        self.interpolate_blinks_markers()   # uses marker (missing data) to detect blinks and missing events
        self.interpolate_blinks_peaks()     # uses derivative to detect blinks still remaining
        self.bandpass_filter()              # third-order Butterworth, 0.01-6Hz
        self.regress_blinks_saccades()      # use deconvolution to remove blink + saccade events
        self.percent_signal_change()        # converts to percent signal change
        self.zscore()                       # zscores pupil
        
        # save all pupil stages
        # 1: raw, 2: blink interpolated, 3: bandpass, 4: deconvolution, 5: percent signal change
        self.TS['pupil_interp']  = self.pupil_interp
        self.TS['pupil_bp']      = self.pupil_bp # band passed
        self.TS['pupil_clean']   = self.pupil_clean
        self.TS['pupil_psc']     = self.pupil_psc
        self.TS['pupil_zscore']  = self.pupil_zscore
        np.save(os.path.join(self.project_directory, '{}.npy'.format(self.alias)), np.array(self.TS))
        
        self.plot_pupil()                   # plots the pupil in all stages
        
        print('Pupil data preprocessed: Subject {}'.format(self.subject) )
        
    
    def detect_peaks(self, x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None):
        """Detect peaks in data based on their amplitude and other features.
        
        Parameters
        ----------
        x : 1D array_like
            Data.
        mph : {None, number}, optional (default = None)
            Detect peaks that are greater than minimum peak height.
        mpd : positive integer, optional (default = 1)
            Detect peaks that are at least separated by minimum peak distance (in number of data).
        threshold : positive number, optional (default = 0)
            Detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors.
        edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
            For a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a
            flat peak (None).
        kpsh : bool, optional (default = False)
            Keep peaks with same height even if they are closer than `mpd`.
        valley : bool, optional (default = False)
            If True (1), detect valleys (local minima) instead of peaks.
        show : bool, optional (default = False)
            If True (1), plot data in matplotlib figure.
        ax : a matplotlib.axes.Axes instance, optional (default = None).
        
        Returns
        -------
        ind : 1D array_like
            Indices of the peaks in `x`.
        
        Notes
        -----
        The detection of valleys instead of peaks is performed internally by simply
        negating the data: `ind_valleys = detect_peaks(-x)`

        The function can handle NaN's 
        See this IPython Notebook [1]_.
        
        References
        ----------
        .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
        
        Examples
        --------
        >>> from detect_peaks import detect_peaks
        >>> x = np.random.randn(100)
        >>> x[60:81] = np.nan
        >>> # detect all peaks and plot data
        >>> ind = detect_peaks(x, show=True)
        >>> print(ind)
        >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
        >>> # set minimum peak height = 0 and minimum peak distance = 20
        >>> detect_peaks(x, mph=0, mpd=20, show=True)
        >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
        >>> # set minimum peak distance = 2
        >>> detect_peaks(x, mpd=2, show=True)
        >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
        >>> # detection of valleys instead of peaks
        >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
        >>> x = [0, 1, 1, 0, 1, 1, 0]
        >>> # detect both edges
        >>> detect_peaks(x, edge='both', show=True)
        >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
        >>> # set threshold = 2
        >>> detect_peaks(x, threshold = 2, show=True)
        """
        x = np.atleast_1d(x).astype('float64')
        if x.size < 3:
            return np.array([], dtype=int)
        if valley:
            x = -x
        # find indices of all peaks
        dx = x[1:] - x[:-1]
        # handle NaN's
        indnan = np.where(np.isnan(x))[0]
        if indnan.size:
            x[indnan] = np.inf
            dx[np.where(np.isnan(dx))[0]] = np.inf
        ine, ire, ife = np.array([[], [], []], dtype=int)
        if not edge:
            ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
        else:
            if edge.lower() in ['rising', 'both']:
                ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
            if edge.lower() in ['falling', 'both']:
                ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
        ind = np.unique(np.hstack((ine, ire, ife)))
        # handle NaN's
        if ind.size and indnan.size:
            # NaN's and values close to NaN's cannot be peaks
            ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
        # first and last values of x cannot be peaks
        if ind.size and ind[0] == 0:
            ind = ind[1:]
        if ind.size and ind[-1] == x.size-1:
            ind = ind[:-1]
        # remove peaks < minimum peak height
        if ind.size and mph is not None:
            ind = ind[x[ind] >= mph]
        # remove peaks - neighbors < threshold
        if ind.size and threshold > 0:
            dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
            ind = np.delete(ind, np.where(dx < threshold)[0])
        # detect small peaks closer than minimum peak distance
        if ind.size and mpd > 1:
            ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
            idel = np.zeros(ind.size, dtype=bool)
            for i in range(ind.size):
                if not idel[i]:
                    # keep peaks with the same height if kpsh is True
                    idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                        & (x[ind[i]] > x[ind] if kpsh else True)
                    idel[i] = 0  # Keep current peak
            # remove the small peaks and sort back the indices by their occurrence
            ind = np.sort(ind[~idel])

        if show:
            if indnan.size:
                x[indnan] = np.nan
            if valley:
                x = -x
            _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

        return ind
        
        
    def detect_nans(self, pupil):
        """Identify start and end of missing data as blink.
            
        Parameters
        ----------
        pupil : 1D array_like
            Pupil data.
            
        Returns
        -------
        nan_start,nan_end : numpy arrays
            Containing start and stop indices of each missing event.
        """
        nan_start = []
        nan_end = []
        for idx,p in enumerate(pupil):
            
            if not idx==len(pupil)-1: # start can't be last point
                if (p == 0) and (pupil[idx-1] != 0): # current point is nan but not the one before
                    nan_start.append(idx)
            
            if not idx==0:     # end can't be first point
                if (p != 0) and (pupil[idx-1] == 0): # current point is not nan, but the one before is
                    nan_end.append(idx-1)
            
        # make sure start and stop equal length
        nan_start = np.sort(nan_start)
        nan_end = np.sort(nan_end)
        nan_start = nan_start[:len(nan_end)]
        
        if not np.sum(nan_start>nan_end)==0:
            print('Error marker stop comes before start!')
            shell()
        return np.array(nan_start), np.array(nan_end)
        
        
    def interpolate_blinks_peaks(self,):
        """Perform linear interpolation around peaks in the rate of change of the pupil size.
        
        Notes
        -----
        The results are stored in self.pupil_interp, self.pupil is also updated.
        Change mpd,mph in peaks_down, peaks_up to make more or less conservative.
        After calling this method, additional interpolation may be performed by calling self.interpolate_blinks_markers().
        """
        time_window = self.time_window_blinks
        lin_interpolation_points = [[-1*self.sample_rate*time_window], [self.sample_rate*time_window]]
        coalesce_period = int(0.75*self.sample_rate)
        
        # we do not want to start or end with a 0:
        self.pupil_interp = deepcopy(self.pupil[:])
                
        interpolated_time_points = np.zeros(len(self.pupil))
        self.pupil_diff = (np.diff(self.pupil_interp) - np.diff(self.pupil_interp).mean()) / np.diff(self.pupil_interp).std() # derivative of time series
        peaks_down = self.detect_peaks(self.pupil_diff, mph=self.mph, mpd=self.mpd, threshold=self.threshold, edge='rising', kpsh=False, valley=False, show=False, ax=False)
        peaks_up = self.detect_peaks(self.pupil_diff*-1, mph=self.mph, mpd=self.mpd, threshold=self.threshold, edge='rising', kpsh=False, valley=False, show=False, ax=False)
        peaks = np.sort(np.concatenate((peaks_down, peaks_up)))
 
        if len(peaks) > 0:
            # prepare:
            # peak_starts = np.sort(np.concatenate((peaks-1, self.blink_starts)))
            # peak_ends = np.sort(np.concatenate((peaks+1, self.blink_ends)))
            peak_starts = np.sort((peaks-1))
            peak_ends = np.sort((peaks+1))
            start_indices = np.ones(peak_starts.shape[0], dtype=bool)
            end_indices = np.ones(peak_ends.shape[0], dtype=bool)
            for i in range(peak_starts.shape[0]):
                try:
                    if peak_starts[i+1] - peak_ends[i] <= coalesce_period:
                        start_indices[i+1] = False
                        end_indices[i] = False
                except IndexError:
                    pass
            peak_starts = peak_starts[start_indices]
            peak_ends = peak_ends[end_indices] 
            
            # interpolate:
            points_for_interpolation = np.array([peak_starts, peak_ends], dtype=int).T + np.array(lin_interpolation_points).T
            for itp in points_for_interpolation:
                itp = [int(x) for x in itp]
                try:
                    self.pupil_interp[itp[0]:itp[-1]] = np.linspace(self.pupil_interp[itp[0]], self.pupil_interp[itp[-1]], itp[-1]-itp[0])
                    interpolated_time_points[itp[0]:itp[-1]] = 1
                except:
                    pass
            # for regressing out
            self.blink_starts = np.sort(np.append(self.blink_starts, np.array(peak_starts), axis=0))
            self.blink_ends = np.sort(np.append(self.blink_ends, np.array(peak_ends), axis=0))
        # interpolated pupil
        self.pupil = self.pupil_interp
        print('pupil blinks interpolated from derivative')         
    
    
    def interpolate_blinks_markers(self, ):
        """Perform linear interpolation around blinks based on blink markers.
        
        Notes
        -----
        Use after self.blink_detection_pupil().
        spline_interpolation_points() is a 2 by X list detailing the data points around the blinks
        (in s offset from blink start and end) that should be used for fitting the interpolation spline.
        The results are stored in self.pupil_interp, self.pupil is also updated.
        After calling this method, additional interpolation may be performed by calling self.interpolate_peaks()
        """
        time_window = self.time_window_blinks
        lin_interpolation_points = [[-1*self.sample_rate*time_window], [self.sample_rate*time_window]]
        coalesce_period = int(0.75*self.sample_rate)
        
        # we do not want to start or end with a 0:
        self.pupil_interp = deepcopy(self.pupil[:])
        
        # set all missing data to zero or other martker:
        self.pupil_interp[self.pupil_interp<1] = 0
        
        interpolated_time_points = np.zeros(len(self.pupil))

        # identify start of missing data as blink 
        blink_starts,blink_ends = self.detect_nans(self.pupil_interp)
        
        # check for neighbouring blinks (coalesce_period, default is 500ms), and string them together:
        start_indices = np.ones(blink_starts.shape[0], dtype=bool)
        end_indices = np.ones(blink_ends.shape[0], dtype=bool)
        for i in range(blink_starts.shape[0]):
            try:
                if blink_starts[i+1] - blink_ends[i] <= coalesce_period:
                    start_indices[i+1] = False
                    end_indices[i] = False
            except IndexError:
                pass
        
        # these are the blink start and end samples to work with:
        if sum(start_indices) > 0:
            blink_starts = blink_starts[start_indices]
            blink_ends = blink_ends[end_indices]
        else:
            blink_starts = None
            blink_ends = None
        
        # do actual interpolation:
        if sum(start_indices) > 0:
            points_for_interpolation = np.array([blink_starts, blink_ends], dtype=int).T + np.array(lin_interpolation_points).T
            points_for_interpolation[points_for_interpolation<0] = 0 # in case coalesce period extends before pupil starts
            for itp in points_for_interpolation:
                if itp[-1]>len(self.pupil_interp): # last point is outside of pupil time series, interpolate till end
                    last_idx = len(self.pupil_interp)-1
                    self.pupil_interp[int(itp[0]):int(last_idx)] = np.linspace(self.pupil_interp[int(itp[0])], self.pupil_interp[int(last_idx)], int(last_idx-itp[0]))                    
                else:
                    self.pupil_interp[int(itp[0]):int(itp[-1])] = np.linspace(self.pupil_interp[int(itp[0])], self.pupil_interp[int(itp[-1])], int(itp[-1]-itp[0]))
                    # if linspace crashes, make sure end points are not before start points in marker index
        # interpolated pupil
        self.pupil = self.pupil_interp
        # for regression
        self.blink_starts = np.sort(np.append(self.blink_starts, np.array(blink_starts), axis=0))
        self.blink_ends = np.sort(np.append(self.blink_ends, np.array(blink_ends), axis=0))
        print('pupil blinks interpolated from markers')
            
            
    def bandpass_filter(self,):
        """Perform bandpass filtering on pupil time series (3rd order butterworth 0.01 to 6 Hz).
        
        Notes
        -----
        This way adds curved artifact to timeseries
        b,a = butter(N, Wn, btype='bandpass')   # define filter
        y = filtfilt(b, a, self.pupil)          # apply filter
        """        
        N = 3 # order
        Nyquist = 0.5*self.sample_rate
        bpass = [0.01,6] # Hz
        Wn = np.true_divide(bpass,Nyquist) # [low,high]
        
        # low pass
        b,a = butter(N,Wn[1],btype='lowpass') # enter high cutoff value
        self.pupil_lp = filtfilt(b, a, self.pupil.astype('float64'))
        
        # high pass
        b,a = butter(N,Wn[0],btype='highpass') # enter low cutoff value
        self.pupil_hp = filtfilt(b, a, self.pupil.astype('float64')) 
        
        # bandpassed
        self.pupil_bp = self.pupil_hp - (self.pupil-self.pupil_lp)
        
        # baseline pupil
        self.pupil_baseline = self.pupil_lp - self.pupil_bp

        self.pupil = self.pupil_bp
        print('pupil bandpass filtered butterworth')
    
    
    def regress_blinks_saccades(self,):
        """Perform linear regression on pupil time series to remove blink and saccade events.
        
        Notes
        -----
        Blink/saccade events are combined into a single nuisance regressor because they were not tagged seperately.
        The nuisance event is estimated based on deconvolution (see output in figure folder), 
        then this response is removed from the time series with linear regression.
        The residuals of this regression are of interest for the pupil analyses as self.pupil_clean. 
        Also, self.pupil is updated.
        """
        plot_IRFs = True # plot for each subject the deconvolved responses in fig folder
        self.add_base = False
        # params:
        downsample_rate = 100
        new_sample_rate = self.sample_rate / downsample_rate
        interval = 5 # seconds to estimate kernels
        
        self.timepoints = np.array(self.TS.reset_index()['level_0'],dtype=int)
        # events:
        blinks = np.array(self.blink_ends) / self.sample_rate # use ends because interpolation starts at start
        blinks = blinks[blinks>25]
        blinks = blinks[blinks<((self.timepoints[-1]-self.timepoints[0])/self.sample_rate)-interval]
        
        if blinks.size == 0:
            blinks = np.array([0.5])

        events = [blinks]
        event_names = ['blinks']
        
        #######################
        #### Deconvolution ####
        #######################
        # first, we initialize the object
        fd = FIRDeconvolution(
                    signal = self.pupil, 
                    events = events, # blinks, saccades
                    event_names = event_names, 
                    sample_frequency = self.sample_rate, # Hz
                    deconvolution_frequency = downsample_rate, # Hz
                    deconvolution_interval = [0, interval] # 0 = stim_onset - 1
                    )

        # we then tell it to create its design matrix
        fd.create_design_matrix()

        # perform the actual regression, in this case with the statsmodels backend
        fd.regress(method = 'lstsq')

        # and partition the resulting betas according to the different event types
        fd.betas_for_events()
        fd.calculate_rsq()
        response = fd.betas_per_event_type.squeeze() # response blinks

        self.blink_response = response
        
        # demean (baseline correct)
        self.blink_response = self.blink_response - self.blink_response[:int(0.2*new_sample_rate)].mean()
        
        if plot_IRFs:    
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111)
            
            # R-squared value
            rsquared = fd.rsq
            # Add error bars
            fd.bootstrap_on_residuals(nr_repetitions=1000)
            # plot subject
            plot_time = response.shape[-1] # for x-axis

            for b in range(fd.betas_per_event_type.shape[0]):
                ax.plot(np.arange(plot_time), fd.betas_per_event_type[b], label=event_names[b])

            for i in range(fd.bootstrap_betas_per_event_type.shape[0]):
                mb = fd.bootstrap_betas_per_event_type[i].mean(axis = -1)
                sb = fd.bootstrap_betas_per_event_type[i].std(axis = -1)

                ax.fill_between(np.arange(plot_time), 
                                mb - sb, 
                                mb + sb,
                                color = 'k',
                                alpha = 0.1)

            ax.set_xticks([0,plot_time])
            ax.set_xticklabels([0,interval]) # removed 1 second from events
            ax.set_xlabel('Time from event (s)')
            ax.set_title('r2={}'.format(round(rsquared[0],2)))
            ax.legend()
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            # Save figure
            fig.savefig(os.path.join(self.figure_folder,'{}_Deconvolution.pdf'.format(self.subject)))

        # fit:
        # ----
        # define objective function: returns the array to be minimized        
        def double_pupil_IRF(params, x):
            s1 = params['s1']
            s2 = params['s2']
            n1 = params['n1']
            n2 = params['n2']
            tmax1 = params['tmax1']
            tmax2 = params['tmax2']
            return s1 * ((x**n1) * (np.e**((-n1*x)/tmax1))) + s2 * ((x**n2) * (np.e**((-n2*x)/tmax2)))
            
        def double_pupil_IRF_ls(params, x, data):
            s1 = params['s1'].value
            s2 = params['s2'].value
            n1 = params['n1'].value
            n2 = params['n2'].value
            tmax1 = params['tmax1'].value
            tmax2 = params['tmax2'].value
            model = s1 * ((x**n1) * (np.e**((-n1*x)/tmax1))) + s2 * ((x**n2) * (np.e**((-n2*x)/tmax2)))
            return model - data
            
        # create data to be fitted
        x = np.linspace(0,interval,len(self.blink_response))
        
        # create a set of Parameters
        params = Parameters()
        params.add('s1', value=-1, min=-np.inf, max=-1e-25)
        params.add('s2', value=1, min=1e-25, max=np.inf)
        params.add('n1', value=10, min=9, max=11)
        params.add('n2', value=10, min=8, max=12)
        params.add('tmax1', value=0.9, min=0.5, max=1.5)
        params.add('tmax2', value=2.5, min=1.5, max=4)

        # do fit, here with powell method:
        data = self.blink_response
        blink_result = minimize(double_pupil_IRF_ls, params, method='powell', args=(x, data))
        self.blink_fit = double_pupil_IRF(blink_result.params, x)

        # upsample:
        x = np.linspace(0,interval,interval*self.sample_rate)
        blink_kernel = double_pupil_IRF(blink_result.params, x)
        
        # regress out from original timeseries with GLM:
        event_1 = np.ones((len(blinks),3))
        event_1[:,0] = blinks
        event_1[:,1] = 0

        GLM = glm_functions.GeneralLinearModel(input_object=self.pupil, event_object=[event_1], sample_dur=1.0/self.sample_rate, new_sample_dur=1.0/self.sample_rate)
        GLM.configure(IRF=[blink_kernel], regressor_types=['stick'],)
        GLM.execute()
        
        self.GLM_measured = GLM.working_data_array
        self.GLM_predicted = GLM.predicted
        self.GLM_r, self.GLM_p = sp.stats.pearsonr(self.GLM_measured, self.GLM_predicted)
        
        # clean data:
        self.pupil_clean = GLM.residuals + self.pupil_baseline.mean() # CLEANED DATA + MEAN added back
        # final timeseries:
        self.pupil = self.pupil_clean 
        print('pupil blinks and saccades removed with linear regression')
           
           
    def percent_signal_change(self,):
        """Convert processed pupil to percent signal change with respect to the temporal mean.
        
        Notes
        -----
        For median use: (timeseries/median*100)-100
        self.pupil is not updated.
        """
        if self.add_base: # did not regress out blinks/saccades
            self.pupil_psc = self.pupil + self.pupil_baseline.mean()
        else:
            self.pupil_psc = deepcopy(self.pupil)
            
        self.pupil_psc = (self.pupil_psc/np.mean(self.pupil_psc)*100)-100 
        print('pupil converted percent signal change')
    
    
    def zscore(self,):
        """Z-score pupil time series.
        
        Notes
        -----
        self.pupil is not updated.
        """
        if self.add_base: # did not regress out blinks/saccades
            self.pupil_zscore = self.pupil + self.pupil_baseline.mean()
        else:
            self.pupil_zscore = deepcopy(self.pupil)
            
        self.pupil_zscore = sp.stats.zscore(self.pupil_zscore)
        print('pupil z-scored')
        
        
    def plot_pupil(self,):               
        """Plot the pupil in all preprocessing stages (1 figure per subject).
        
        Notes
        -----
        subplots are... 1: raw, 2: blink interpolated, 3: temporal filtering, 4: blinks/saccades removed, 5: percent signal change, 6: z-score
        The pupil is downsampled for plotting.
        The figure is saved as PDF in the figure folder.
        """
        from scipy.signal import decimate
        downsample_rate = 2 # Hz, the lower, the faster the plotting goes
        downsample_factor = self.sample_rate / downsample_rate # 50
         
        # Make a figure
        fig = plt.figure(figsize=(20,10))
        
        # RAW
        try:
            ax = fig.add_subplot(511)
            pupil = self.pupil_raw
            pupil = decimate(pupil, int(downsample_factor), ftype='fir')
            df = pd.DataFrame(pupil)
            df.columns = ['raw']
            # pupil_diff = (np.diff(pupil, prepend=pupil[0]) - np.diff(pupil).mean()) / np.diff(pupil).std() # add derivative on which blinks are detected
            # df['diff'] = pupil_diff
            sns.lineplot(data=df,legend='full',ax=ax)
            # ax.axhline(y=self.mph)
            # ax.axhline(y=0)
        except:
            pass
        
        # BLINK INTERPOLATED
        try: 
            ax = fig.add_subplot(512)
            pupil = self.pupil_interp
            pupil = decimate(pupil, int(downsample_factor), ftype='fir')
            df = pd.DataFrame(pupil)
            df.columns = ['blink interpolated']
            sns.lineplot(data=df,legend='full',ax=ax)
        except:
            pass

        # BANDPASS FILTERED
        try: 
            # add columns to df for filtered
            ax = fig.add_subplot(513)
            
            pupil = self.pupil_lp
            pupil = decimate(pupil, int(downsample_factor), ftype='fir')
            df = pd.DataFrame(pupil)
            df.columns = ['low pass']
            
            pupil = self.pupil_hp
            pupil = decimate(pupil, int(downsample_factor), ftype='fir')
            df['high pass'] = pupil
            
            pupil = self.pupil_bp
            pupil = decimate(pupil, int(downsample_factor), ftype='fir')
            df['band pass'] = pupil
            
            pupil = self.pupil_baseline
            pupil = decimate(pupil, int(downsample_factor), ftype='fir')
            df['baseline'] = pupil
            
            sns.lineplot(data=df,legend='full',ax=ax)
            
        except:
            pass
            
        # CLEAN: BLINKS & SACCADES REMOVED
        try: 
            ax = fig.add_subplot(514)
            pupil = self.pupil_clean
            pupil = decimate(pupil, int(downsample_factor), ftype='fir')
            df = pd.DataFrame(pupil)
            df.columns = ['clean deconv.']
            sns.lineplot(data=df,legend='full',ax=ax)
        except:
            pass
            
        # PSC
        try: 
            ax = fig.add_subplot(515)
            pupil = self.pupil_psc
            pupil = decimate(pupil, int(downsample_factor), ftype='fir')
            df = pd.DataFrame(pupil)
            df.columns = ['perc. signal change']
            sns.lineplot(data=df,legend='full',ax=ax)
        except:
            pass
              
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        # Save figure
        fig.savefig(os.path.join(self.figure_folder, '{}_preprocessing.pdf'.format(self.subject)))
    
    
class trials(object):
    """Define a class for the single trial level pupil data.

    Parameters
    ----------
    subject : string
        Subject number.
    edf : string
        The name of the current subject's EDF file containing pupil data.
    project_directory : str
        Path to the derivatives data directory.
    sample_rate : int
        Sampling rate of pupil data in Hertz.
    phases : list
        Message markers for each event of interest in EDF file as a list of strings (e.g., ['cue','target']).
    time_locked : list
        List of strings indiciting the events for time locking that should be analyzed (e.g., ['cue_locked','target_locked']).
    pupil_step_lim : list 
        List of arrays indicating the size of pupil trial kernels in seconds with respect to first event, first element should max = 0! (e.g., [[-baseline_window,3],[-baseline_window,3]] ).
    baseline_window : float
        Number of seconds before each event in self.time_locked that are averaged for baseline correction.
    pupil_time_of_interest : list
        List of arrays indicating the time windows in seconds in which to average evoked responses, per event in self.time_locked, see in higher.plot_evoked_pupil (e.g., [[1.0,2.0],[1.0,2.0]]).
    
    Attributes
    ----------
    subject : string
        Subject number.
    alias : string
        The name of the current subject's EDF file containing pupil data.
    project_directory : str
        Path to the derivatives data directory.
    figure_folder : str
        Path to the figure directory.
    sample_rate : int
        Sampling rate of pupil data in Hertz.
    phases : list
        Message markers for each event of interest in EDF file as a list of strings (e.g., ['cue','target']).
    time_locked : list
        List of strings indiciting the events for time locking that should be analyzed (e.g., ['cue_locked','target_locked']).
    pupil_step_lim : list 
        List of arrays indicating the size of pupil trial kernels in seconds with respect to first event, first element should max = 0! (e.g., [[-baseline_window,3],[-baseline_window,3]] ).
    baseline_window : float
        Number of seconds before each event in self.time_locked that are averaged for baseline correction.
    pupil_time_of_interest : list
        List of arrays indicating the time windows in seconds in which to average evoked responses, per event in self.time_locked, see in higher.plot_evoked_pupil (e.g., [[1.0,2.0],[1.0,2.0]]).

    """
    
    def __init__(self,subject, edf, project_directory, sample_rate, phases, time_locked, pupil_step_lim, baseline_window):
        """Constructor method"""
        self.subject = subject
        self.alias = edf
        self.project_directory = os.path.join(project_directory, self.subject,'beh') # single-subject directory
        self.figure_folder = os.path.join(project_directory, 'figures', 'preprocessing') # group-level directory for easy inspection
        self.sample_rate = sample_rate
        self.phases = phases
        ##############################    
        # Pupil time series information:
        ##############################
        self.time_locked = time_locked
        self.pupil_step_lim = pupil_step_lim # size of pupil trials in seconds with respect to first event, first element should max = 0!
        self.baseline_window = baseline_window # seconds before event of interest
                
                
    def event_related_subjects(self,pupil_dv):
        """Cut out time series of pupil data locked to time points of interest within the given kernel.
            
        Parameters
            ----------
        pupil_dv : string
            The pupil time series to be processed (e.g., 'pupil_psc' or 'pupil_zscore')
            
        Notes
        -----
        Saves events as numpy arrays per subject in dataframe folder/subjects per event of interest.
        Rows = trials x kernel length
        """        
        cols = ['index', 'pupil', 'pupil_interp', 'pupil_bp', 'pupil_clean', 'pupil_psc', 'pupil_zscore']
        
        # loop through each type of event to lock events to...
        for t,time_locked in enumerate(self.time_locked):
            pupil_step_lim = self.pupil_step_lim[t]
            TS = pd.DataFrame(np.load(os.path.join(self.project_directory, '{}.npy'.format(self.alias))), columns=cols)   
            TS = TS.loc[:,['index',pupil_dv]] # don't need all columns            
            # get indices of phases with respect to full time series (add 1 because always one cell before event)
            phases = pd.read_csv(os.path.join(self.project_directory, '{}_phases.csv'.format(self.alias)))
            phase_idx = np.array(phases[phases['msg'].str.contains(self.phases[t])]['index'])+1
            #print('phases[t]:' + str(np.array(phases[phases['msg'].str.contains(self.phases[t])]['index'])))
            
            # loop through trials, cut out events
            r = len(phase_idx) # number of trials
            c = int((pupil_step_lim[1]-pupil_step_lim[0])*self.sample_rate)
            SAVE_TRIALS = np.zeros((r,c))
            SAVE_TRIALS[SAVE_TRIALS==0] = np.nan # better than zeros for missing data
            # phase_idx refers to original index, not absolute row/position of pupil
            for trial,t_idx in enumerate(phase_idx):
                this_row = TS[TS['index'] == t_idx].index.tolist()[0]
                # gets one extra sample, cut off at end
                this_pupil = TS.loc[int(this_row+(pupil_step_lim[0]*self.sample_rate)):int(this_row+(pupil_step_lim[1]*self.sample_rate))-1,pupil_dv]                    
                this_pupil = np.array(this_pupil)
                SAVE_TRIALS[trial,:len(this_pupil)] = this_pupil # sometimes not enough data at the end
            # save as CSV file
            pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas
            SAVE_TRIALS = pd.DataFrame(SAVE_TRIALS)
            
            SAVE_TRIALS.to_csv(os.path.join(self.project_directory, '{}_{}_evoked.csv'.format(self.alias, time_locked)), float_format='%.16f')
            print('subject {}, {} events extracted'.format(self.subject, time_locked))
        print('sucess: event_related_subjects')
    
    
    def event_related_baseline_correction(self):
        """Baseline correction on evoked responses, per trial. 
        
        Notes
        -----
        Saves baseline pupil in behavioral log file.
        """                 
        # loop through each type of event to lock events to...
        for t,time_locked in enumerate(self.time_locked):
            pupil_step_lim = self.pupil_step_lim[t]
            P = pd.read_csv(os.path.join(self.project_directory, '{}_{}_evoked.csv'.format(self.alias, time_locked)))
            P.drop(['Unnamed: 0'],axis=1,inplace=True)
            P = np.array(P)
            baselines_file = os.path.join(self.project_directory, '{}_{}_baselines.csv'.format(self.alias, time_locked))  # save baseline pupils
            SAVE_TRIALS = []
            for trial in range(len(P)):
                event_idx = int(abs(pupil_step_lim[0]*self.sample_rate))
                base_start = int(event_idx - (self.baseline_window*self.sample_rate))
                base_end = int(base_start + (self.baseline_window*self.sample_rate))
                # mean within baseline window
                this_base = np.mean(P[trial,base_start:base_end]) 
                SAVE_TRIALS.append(this_base)
                # remove baseline mean from each time point
                P[trial] = P[trial]-this_base
            # save baseline corrected events and baseline means too!
            P = pd.DataFrame(P)
            
            P.to_csv(os.path.join(self.project_directory, '{}_{}_evoked_basecorr.csv'.format(self.alias, time_locked)), float_format='%.16f')
            
            B = pd.DataFrame()
            B['pupil_baseline_' + time_locked] = np.array(SAVE_TRIALS) #was pupil_b
            B.to_csv(baselines_file, float_format='%.16f')
            print('subject {}, {} events baseline corrected'.format(self.subject, time_locked))
        print('sucess: event_related_baseline_correction')