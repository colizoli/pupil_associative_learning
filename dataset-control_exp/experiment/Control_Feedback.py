"""
Control experiment: Impulse response functions to feedback tones
"""

# Import necessary modules
from psychopy import core, visual, event, data, sound, gui, monitors
import random
import numpy as np
import pandas as pd
import os, time  # for paths and data
from IPython import embed as shell
import gpe_params as p

debug_mode = False
eye_mode = True

### PARAMETERS ###
# Timing
REPS        = 50   # times to repeat each tone
t_baseline  = .5   # baseline pupil
t_ITI       = p.t_ITI

# Get subject number
# subject_ID = int(input('Please Enter Subject ID:  '))

g = gui.Dlg()
g.addField('Subject Number:')
g.show()
subject_ID = int(g.data[0])

if subject_ID:

    ## Create LogFile folder cwd/LogFiles
    cwd = os.getcwd()
    logfile_dir = os.path.join(cwd,'LogFiles','sub-{}'.format(subject_ID)) 
    if not os.path.isdir(logfile_dir):
        os.makedirs(logfile_dir)
        
    ## output file name with time stamp prevents any overwriting of data
    timestr = time.strftime("%Y%m%d-%H%M%S") 
    output_filename = os.path.join(logfile_dir,'sub-{}_task-control_exp_sounds_beh_{}.csv'.format(subject_ID, timestr))
    cols = ['subject','trial_num','tone','ITI']
    DF = pd.DataFrame(columns=cols)
        
    # Set-up window:
    mon = monitors.Monitor('myMac15', width=p.screen_width, distance=p.screen_dist)
    mon.setSizePix((p.scnWidth, p.scnHeight))
    win = visual.Window((p.scnWidth, p.scnHeight),color=p.grey,colorSpace='rgb255',monitor=mon,fullscr=not debug_mode,units='pix',allowStencil=True,autoLog=False)
    win.setMouseVisible(False)
    
    # Set-up stimuli and timing
    welcome_txt = "SOUNDS\
    \nPlease put on the headphones.\
    \nMaintain fixation in the center of the screen on the '+'.\
    \nYou do not have to do anything else.\
    \nBlink as you normally would.\
    \n\n<Press any button to BEGIN>" 
    
    pauze = 'Take a short break now.\
    \n<Press any button to CONTINUE the task>'
       
    stim_instr  = visual.TextStim(win, color='black', pos=(0.0, 0.0), wrapWidth=p.ww)  # can't really center, and TextBox doesn't work, stupid!
    stim_fix    = visual.TextStim(win, text='+',color='black', pos=(0.0, 0.0), height=p.fh)
    feed        = sound.Sound(500, secs=p.t_feed) 
    clock       = core.Clock()
    
    trials = [0,1]*REPS # error or correct
        
    # Set conditions and stimulus list
    if debug_mode:
        trials    = trials[:10]

    random.shuffle(trials) # shuffle order of colors      

    ### CONFIG & CALIBRATION EYE-TRACKER ###
    if eye_mode:
        import funcs_pylink as eye
        task = 'feedback'
        eye.config(subject_ID,task)
        eye.run_calibration(win,p.scnWidth, p.scnHeight)
        eye.start_recording()
        eye.send_message('subject_ID sub-{} task-{} timestamp {}'.format(subject_ID,task,timestr))

    # Welcome instructions
    stim_instr.setText(welcome_txt)
    stim_instr.draw()
    win.flip()
    core.wait(0.25)
    event.waitKeys()
    
    # Wait a few seconds before first trial to stabilize gaze
    stim_fix.draw()
    win.flip()
    core.wait(3) 
    #### TRIAL LOOP ###
    for i,t in enumerate(trials):
        # i = trial number, t = correct or error
            
        # Target stimuli current trial        
        print('Feedback: {}'.format(t))
                    
        # Pupil baseline
        stim_fix.draw() 
        win.flip()
        if eye_mode:
            eye.send_message('trial {} new trial baseline phase 1'.format(i))
        core.wait(t_baseline) #now longer to see

        ## FEEDBACK
        if t: # correct
            feed.setSound('B', octave=4)
        else: # error
            feed.setSound('D', octave=3)    
        feed.play()
        if eye_mode:
            eye.send_message('trial {} tone phase 2'.format(i))
            
        # ITI
        stim_fix.draw() 
        win.flip()
        ITI = np.round(random.uniform(t_ITI[0],t_ITI[1]),2)
        core.wait(ITI)
        
        # For quitting early
        keys = event.getKeys()
        if keys:
            # q quits the experiment
            if keys[0] == 'q':
                if eye_mode:
                    eye.stop_skip_save()
                core.quit()
        
        # output data frame on each trial 
        DF.loc[i] = [
                int(subject_ID),    # subject
                int(i),             # trial number
                int(t),             # tone error or correct
                ITI                 # ITI
            ]
        DF.to_csv(output_filename)
           
    # End screen for participants
    stim_instr.setText('Well done! Data transfering.....')
    stim_instr.draw()
    win.flip()
        
    # Close-up   
    if eye_mode:
        eye.stop_recording(timestr,task)
    win.close() # Close window
    core.quit()














