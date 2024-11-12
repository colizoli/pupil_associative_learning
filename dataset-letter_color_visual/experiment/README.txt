# Pupil dilation offers a time-window on prediction error
Olympia Colizoli, Tessa van Leeuwen, Danaja Rutar, Harold Bekkering
bioRxiv 2024.10.31.621279; doi: https://doi.org/10.1101/2024.10.31.621279



-- Dataset #2: Letter-color 2AFC task administer experiment to participants --

PsychoPy tasks to be run in a python environment (we used PsychoPy version 1.82).

First, activate the python environment before running any tasks.
Second, make sure the EyeLink is on. 

Training_Task.py should be administered first. Training refers to the Odd-ball detection task. This script will randomly select the letter-color mappings per participant. It saves those mappings in the stimuli folder by subject number. It is NOT necessary to use the eye-tracker with this script.

Decision_Task.py should be administered after Training_Task.py. 
This script runs with the EyeLink. Do a calibration and validation before the task starts.


-- Instructions (general): --

There are 3 parts to the experiment. 

The first part is a detection task, where you need to respond to the identity of a stimulus and detect oddball (meaning strange) stimuli. You will get a short round of practice trials before beginning. You will get feedback on each trial to tell you if your response was correct or not, or you were too slow to respond. 

After the first task, we will measure your eye-movements during the tasks. There is a short calibration procedure before the tasks begin. During these two tasks, always look at the ‘+’ fixation cross whenever it is on screen. Meaning, do not move your eyes around the screen, but do maintain a steady gaze.

The second task is a decision task, you will get the instructions for the 2nd experiment once the 1st one is finished. 


-- Instructions calibration/validation of eye-tracker: --

You will see a small dot appear on the screen. You are to follow the dot with your eyes, so we can ‘tell’ the eye-tracker where you are looking. 

However, do not try to anticipate where the dot will move by searching around the screen for it. Keep fixating on the dot, then only after the dot moves, follow the dot with your eyes. 

Again, keep your fixation/gaze on the dot until it has moved to a new position. 

This calibration procedure will take place 2 or more times.
