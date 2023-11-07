# data set 2: letter-color 2AFC task

PsychoPy tasks to be run in a python environment (python 2.7 and above should work).

First, activate the python environment before running any tasks.
Second, make sure the EyeLink is on. 

How to run a python script from the terminal: <br>
python my_script.py

Training_Task.py should be administered first. This will randomly select the letter-color mappings per participant. 
It saves those mappings in the stimuli folder by subject number. It is NOT necessary to use the eye-tracker with this script.

Decision_Task.py should be administered after Training_Task.py. 
This script runs with the EyeLink. Do a calibration and validation before the task starts.

See 'Instructions researcher.docx' for what to say to the participants.