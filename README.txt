Manuscript Name: Pupil dilation offers a time-window on prediction error
Published: bioRxiv 2024.10.31.621279; doi: https://doi.org/10.1101/2024.10.31.621279

v1, 12 Nov 2024
Authors: Olympia Colizoli, Tessa van Leeuwen, Danaja Rutar, Harold Bekkering
Data curation: Olympia Colizoli

In this collection, you will find two associative learning tasks during which pupil dilation was recorded.
We primarily focused on the feedback interval of each trial in order to investigate prediction errors following 
prediction outcome or feedback on performance accuracy in two different tasks. All stimuli were presented in the visual 
and/or auditory domain. There are three independent data sets included here: 
Data sets #1 and #2 are for the main analysis of the associated preprint and there is one control experiment 
to investigate the response of the pupil to colors and sounds used in data set #2. 
All data was collected at the Radboud University in Nijmegen, the Netherlands. 

-- Data set #1: Cue-Target 2AFC task
In the first data set, participants were instructed to predict the upcoming orientation
(left vs. right) of a visual target based on the probability of visual and auditory cues. 
After the cue interval, participants were instructed to respond by predicting the orientation of the upcoming target grating.
After the response interval, participants were shown the target, followed by an inter-trial interval.
Dataset #1 has been previously published and the raw data and experimental scripts can be obtained here: https://doi.org/10.34973/t41p-hx94

-- Data set #2: Letter-Color 2AFC task
In the second data set, participants were first exposed to letter-color pairs of stimuli in different
frequency conditions during an odd-ball detection task. The letter-color pair contingencies
were irrelevant to the odd-ball task performance. The participants subsequently completed a
decision-making task in which they had to decide which letter was presented together most
often with which color during the previous odd-ball detection task (match vs. no match). 
In the decision task, participants were shown a letter followed by a colored square.
They were instructed to press the corresponding button for either "match" or "no match" as soon 
as the colored square appeared.
After this response interval, feedback on accuracy (i.e., correct or error) was presented to the participants 
in the auditory modality indicated by two different tones followed by an inter-trial interval.

-- Data set #3: Control Experiment --
Different colors and tones could influence the pupil response due to inherent properties of the
stimuli, and thereby confound true feedback-related signals. Therefore, complementary to the
main analysis, we administered two control tasks in one independent sample of participants
to directly assess whether confounding effects on the pupil’s response to the colors and tones
presented in the letter-color 2AFC task should be expected. 
In one part of the control experiment, participants were instructed to push a button whenever they saw a colored square. 
In the other part of the control experiments, participants passively listened to the two tones presented in the main experiment.
Pupil dilation was continuously recorded during both parts of the control experiment. 


Structure of the data collection:
--------------------------------
README.txt: this file
analysis_conda_list_python36.txt: a list of all packages installed in Python environment used for data analysis
dataset-control_exp
- analysis: all Python scripts for data analysis
-- glm_functions.py: general linear model functions for pupil preprocessing
-- higher_level_functions_control_exp.py: all higher level functions for data analysis
-- participants_control_exp.csv: a list of participants. See codebook for explanation of variable names.
-- preprocessing_functions_control_exp.py: functions for preprocessing pupil data
-- run_control_exp_analysis.py: run all analyses from this script
- derivatives: a copy of the main processed data files
-- data_frames
--- task-control_exp_colors_subjects.csv: colors task, all trial information and pupil DV averaged within time-window(s) of interest. See codebook for explanation of variable names.
--- task-control_exp_sounds_subjects.csv: sounds task, all trial information and pupil DV averaged within time-window(s) of interest. See codebook for explanation of variable names.
- experiment:
--- Control_Colors.py: colors task, control experiment (eye-tracking)
--- Control_Sounds.py: sounds task, control experiment (eye-tracking)
--- EyeLinkCoreGraphicsPsychoPy.py: necessary for the EyeLink
--- funcs_pylink.py: necessary for the EyeLink
--- gpe_params.py: define experimental parameters (e.g., window size, colors, timing)
--- README.txt: explains how to run experiment and instructions for participants
--- stimuli: subject-specific reaction times used for colors task
---- sub-xxx_task-decision_meanRT.csv: time that the colored square was shown to the participant based on the mean reaction time (RT) of yoked participant.
- raw
-- sub-xxx
--- beh: all participants raw logfiles and EyeLink files
---- sub-xxx_task-control_exp_colors_beh.csv: logfile from the colors control experiment. See codebook for explanation of variable names.
---- sub-xxx_task-control_exp_sounds_beh.csv: logfile from the sounds control experiment. See codebook for explanation of variable names.
---- sub-xxx_task-control_exp_colors_recording-eyetracking_physio.asc: eye-tracking events and samples extracted to text format.
---- sub-xxx_task-control_exp_colors_recording-eyetracking_physio.EDF: eye-tracking logfile original format.
---- sub-xxx_task-control_exp_colors_recording-eyetracking_physio.gaz: eye-tracking samples only extracted to text format.
---- sub-xxx_task-control_exp_colors_recording-eyetracking_physio.msg: eye-tracking events only extracted to text format.
- README.txt: notes on the dataset-control_exp collection
dataset-cue_target_orientation
- analysis: all Python scripts for data analysis
-- glm_functions_orientation.py: general linear model functions for pupil preprocessing
-- higher_level_functions_orientation.py: all higher level functions for data analysis
-- participants_orientation.csv: a list of participants. See codebook for explanation of variable names.
-- preprocessing_functions_orientation.py: functions for preprocessing pupil data
-- run_analysis_orientation.py: run all analyses from this script
- derivatives: a copy of the main processed data file
-- data_frames
--- task-cue_target_orientation_subjects.csv: all trial information and pupil DV averaged within time-window(s) of interest. See codebook for explanation of variable names.
- README.txt: notes on the dataset-cue_target_orientation collection
dataset-letter_color_visual
- analysis:
-- glm_functions_visual.py: general linear model functions for pupil preprocessing
-- higher_level_functions_visual.py: all higher level functions for data analysis
-- oddball_training_visual.py: analyze the oddball training task data
-- participants_visual.csv: a list of participants. See codebook for explanation of variable names.
-- preprocessing_functions_visual.py: functions for preprocessing pupil data
-- run_analysis_visual: run all analyses from this script
- derivatives: a copy of the main processed data file
-- data_frames
--- task-letter_color_visual_decision_subjects.csv: decision task, all trial information and pupil DV averaged within time-window(s) of interest. See codebook for explanation of variable names.
--- task-letter_color_visual_training_subjects.csv: training task, all trial information. See codebook for explanation of variable names.
- experiment
--- Decision_Task.py: letter-color 2AFC task (eye-tracking)
--- EyeLinkCoreGraphicsPsychoPy.py: necessary for the EyeLink
--- funcs_pylink.py: necessary for the EyeLink
--- gpe_params.py: define experimental parameters (e.g., window size, colors, timing)
--- Practice_Training.py: practice the oddball training task
--- README.txt: explains how to run experiment and instructions for participants
--- stimuli: balancing trials and conditions for decision and training tasks
---- color_permutations.csv: all possible combinations of six colors drawn randomly for each participant. See codebook for explanation of variable names.
---- decision_balancing.csv: decision task, minimum unit of trials to repeat in the experiment to ensure conditions. See codebook for explanation of variable names.
---- letter_permutations.csv: all possible combinations of six letters drawn randomly for each participant. See codebook for explanation of variable names.
---- practice_training_balancing.csv: practice training task, minimum unit of trials to repeat in the experiment to ensure conditions. See codebook for explanation of variable names.
---- training_balancing.csv: training task, minimum unit of trials to repeat in the experiment to ensure conditions. See codebook for explanation of variable names.
--- test_colors.py: display colors on monitor
--- Training_Task.py: oddball training task
- raw
-- sub-xxx
--- beh: all participants raw logfiles and EyeLink files
---- sub-xxx_task-letter_color_visual_colors.csv: the colors used for this participants in experiment. See codebook for explanation of variable names.
---- sub-xxx_task-letter_color_visual_decision_beh.csv: decision task, logfile of experiment. See codebook for explanation of variable names.
---- sub-xxx_task-letter_color_visual_decision_recording-eyetracking_physio.asc: eye-tracking events and samples extracted to text format.
---- sub-xxx_task-letter_color_visual_decision_recording-eyetracking_physio.edf: eye-tracking logfile original format.
---- sub-xxx_task-letter_color_visual_decision_recording-eyetracking_physio.gaz: eye-tracking samples only extracted to text format.
---- sub-xxx_task-letter_color_visual_decision_recording-eyetracking_physio.msg: eye-tracking events only extracted to text format.
---- sub-xxx_task-letter_color_visual_training_beh.csv: training task, logfile of experiment. See codebook for explanation of variable names.
- README.txt
documentation: codebooks for all csv files
- codebook_participants_control_exp.html
- codebook_participants_orientation.html
- codebook_participants_visual.html
- codebook_sub-xxx_task-control_exp_colors_beh.html
- codebook_sub-xxx_task-letter_color_visual_colors.html
- codebook_sub-xxx_task-letter_color_visual_decision_beh.html
- codebook_sub-xxx_task-letter_color_visual_training_beh.html
- codebook_task-control_exp_colors_subjects.html
- codebook_task-control_exp_sounds_subjects.html
- codebook_task-cue_target_orientation_subjects.html
- codebook_task-letter_color_visual_decision_subjects.html
- codebook_task-letter_color_visual_training_subjects.html



