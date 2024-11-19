Manuscript Name: Pupil dilation offers a time-window on prediction error
Published: bioRxiv 2024.10.31.621279; doi: ttps://doi.org/10.1101/2024.10.31.621279

v1, 12 Nov 2024
Authors: Olympia Colizoli, Tessa van Leeuwen, Danaja Rutar, Harold Bekkering
Data curation: Olympia Colizoli

This data collection contains the data and analysis scripts associated with three independent data sets: 
Data sets #1 and #2 are for the main analysis and there is one control experiment. 
All data was collected at the Radboud University in Nijmegen, the Netherlands. 

-- Dataset #1: Cue-Target 2AFC task
In the first data set, participants were instructed to predict the upcoming orientation
(left vs. right) of a visual target based on the probability of visual and auditory cues. 
Dataset #1 has been previously published and the raw data and experimental scripts can be obtained here: https://doi.org/10.34973/t41p-hx94

-- Dataset #2: Letter-Color 2AFC task
In the second data set, participants were first exposed to letter-color pairs of stimuli in different
frequency conditions during an odd-ball detection task. The letter-color pair contingencies
were irrelevant to the odd-ball task performance. The participants subsequently completed a
decision-making task in which they had to decide which letter was presented together most
often with which color during the previous odd-ball detection task (match vs. no match). 

Pupil dilation was recorded during the decision-making tasks in both data sets and the post-feedback
pupil response was the event of interest. We did not formally compare the results
across the two data sets given substantial differences between these two task contexts. We
expected the post-feedback pupil dilation to scale with KL divergence in both tasks in a
relatively early time window, following the results of O’Reilly et. al. We explored whether later
prediction error components in the post-feedback pupil dilation might reflect other information-theoretic
variables, such as Shannon surprise or entropy.

-- Dataset Control Experiment --
Different colors and tones could influence the pupil response due to inherent properties of the
stimuli, and thereby confound true feedback-related signals. Therefore, complementary to the
main analysis, we administered two control tasks in one independent sample of participants
to directly assess whether confounding effects on the pupil’s response to the colors and tones
presented in the letter-color 2AFC task should be expected.


Structure of the data collection:
--------------------------------
README.txt: this file
dataset-control_exp
- analysis: all Python scripts for data analysis
-- glm_functions.py: general linear model functions for pupil preprocessing
-- higher_level_functions_control_exp.py: all higher level functions for data analysis
-- participants_control_exp.csv: a list of participants
-- preprocessing_functions_control_exp.py: functions for preprocessing pupil data
-- run_control_exp_analysis.py: run all analyses from this script
- derivatives: a copy of the main processed data files
-- data_frames
--- task-control_exp_colors_stim_locked_evoked_colors.csv: colors task, average evoked response per subject per color locked to stimulus onset
--- task-control_exp_colors_stim_locked_evoked_subject.csv: colors task, average evoked response per subject locked to stimulus onset
--- task-control_exp_colors_subjects.csv: colors task, all trial information and pupil DV averaged within time-window(s) of interest
--- task-control_exp_sounds_stim_locked_evoked_colors.csv: sounds task, average evoked response per subject per sound locked to stimulus onset
--- task-control_exp_sounds_stim_locked_evoked_subject.csv: sounds task, average evoked response per subject locked to stimulus onset
--- task-control_exp_sounds_subjects.csv: sounds task, all trial information and pupil DV averaged within time-window(s) of interest
- experiment:
--- Control_Colors.py: colors task, control experiment (eye-tracking)
--- Control_Sounds.py: sounds task, control experiment (eye-tracking)
--- EyeLinkCoreGraphicsPsychoPy.py: necessary for the EyeLink
--- funcs_pylink.py: necessary for the EyeLink
--- gpe_params.py: define experimental parameters (e.g., window size, colors, timing)
--- README.txt: explains how to run experiment and instructions for participants
--- stimuli: subject-specific reaction times used for colors task
- raw
-- sub-xxx
--- beh: all participants raw logfiles and EyeLink files
- README.txt: notes on the dataset-control_exp collection
dataset-cue_target_orientation
- analysis: all Python scripts for data analysis
-- glm_functions_orientation.py: general linear model functions for pupil preprocessing
-- higher_level_functions_orientation.py: all higher level functions for data analysis
-- participants_orientation.csv: a list of participants
-- preprocessing_functions_orientation.py: functions for preprocessing pupil data
-- run_analysis_orientation.py: run all analyses from this script
- derivatives: a copy of the main processed data file
-- data_frames
--- task-cue_target_orientation_subjects.csv: all trial information and pupil DV averaged within time-window(s) of interest
- README.txt: notes on the dataset-cue_target_orientation collection
dataset-letter_color_visual
- analysis:
-- glm_functions_visual.py: general linear model functions for pupil preprocessing
-- higher_level_functions_visual.py: all higher level functions for data analysis
-- oddball_training_visual.py: analyze the oddball training task data
-- participants_visual.csv: a list of participants
-- preprocessing_functions_visual.py: functions for preprocessing pupil data
-- run_analysis_visual: run all analyses from this script
- derivatives: a copy of the main processed data file
-- data_frames
--- task-letter_color_visual_decision_subjects.csv: decision task, all trial information and pupil DV averaged within time-window(s) of interest
--- task-letter_color_visual_training_subjects.csv: training task, all trial information
- experiment
--- Decision_Task.py: letter-color 2AFC task (eye-tracking)
--- EyeLinkCoreGraphicsPsychoPy.py: necessary for the EyeLink
--- funcs_pylink.py: necessary for the EyeLink
--- gpe_params.py: define experimental parameters (e.g., window size, colors, timing)
--- Practice_Training.py: practice the oddball training task
--- README.txt: explains how to run experiment and instructions for participants
--- stimuli: balancing trials and conditions for decision and training tasks
--- test_colors.py: display colors on monitor
--- Training_Task.py: oddball training task
- raw
-- sub-xxx
--- beh: all participants raw logfiles and EyeLink files
- README.txt
documentation
- analysis_conda_list_python36.txt: a list of all packages installed in Python environment used for data analysis
- Colizoli_vanLeeuwen_Rutar_Bekkering_2024_preprint.pdf: a copy of the preprint


