"""
Control experiment: Impulse response functions to colors
"""
# phase 1 new trial onset
# phase 2 color onset
# phase 3 response
# phase 4 ITI onset

# Import necessary modules
from psychopy import core, visual, event, data, gui, monitors
import random
import numpy as np
import pandas as pd
import os, time  # for paths and data
from IPython import embed as shell
import gpe_params as p

debug_mode = False
eye_mode = True

# Get subject number
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
    
    ## Get meanRT for setting stim duration
    DFRT = pd.read_csv(os.path.join('stimuli','sub-{}_task-decision_meanRT.csv'.format(subject_ID-100)))
        
    ### PARAMETERS ###
    # Timing
    REPS        = 20    # times to repeat all 6 colors
    t_baseline  = 0.5   # baseline pupil
    t_stim      = np.round(np.array(DFRT['meanRT']),2)  # colored square (mean RT of  matched subject from Decision task)
    t_ITI       = p.t_ITI

    print(np.round(np.array(DFRT['meanRT']),2))
    # RGB255 values 'rgb255' color space
    colors = p.colors
        
    ## output file name with time stamp prevents any overwriting of data
    timestr = time.strftime("%Y%m%d-%H%M%S") 
    output_filename = os.path.join(logfile_dir,'sub-{}_task-colors_events_{}.csv'.format(subject_ID,timestr ))
    cols = ['subject','trial_num','r','g','b','RT','ITI']
    DF = pd.DataFrame(columns=cols)
        
    # Set-up window:
    mon = monitors.Monitor('myMac15', width=p.screen_width, distance=p.screen_dist)
    mon.setSizePix((p.scnWidth, p.scnHeight))
    win = visual.Window((p.scnWidth, p.scnHeight),color=p.grey,colorSpace='rgb255',monitor=mon,fullscr=not debug_mode,units='pix',allowStencil=True,autoLog=False)
    win.setMouseVisible(False)
    
    # Set-up stimuli and timing
    welcome_txt = "COLORS\
    \nMaintain fixation in the center of the screen on the '+'.\
    \nWhen you see a color, press the Right-ALT key as fast as possible.\
    \nBlink as you normally would.\
    \n\n<Press any button to BEGIN>"
    
    
    stim_instr  = visual.TextStim(win, color='black', pos=(0.0, 0.0), wrapWidth=p.ww) 
    stim_fix    = visual.TextStim(win, text='+',color='black', pos=(0.0, 0.0), height=p.fh)
    stim_sq     = visual.Rect(win, width=p.sqw, height=p.sqh, autoLog=None, pos=(0.0, 0.0))
    clock       = core.Clock()
    
    # Set conditions and stimulus list
    if debug_mode:
        reps      = colors*1
    else:
        reps      = colors*REPS
    trials = len(reps)
    random.shuffle(reps) # shuffle order of colors      
    
    ### CONFIG & CALIBRATION EYE-TRACKER ###
    if eye_mode:
        import funcs_pylink as eye
        task = 'colors'
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
    for t in range(trials):

        # Target stimuli current trial        
        print('########## Trial {} #########'.format(t+1))
        print('Color of current trial: {}'.format(reps[t]))
                    
        # Pupil baseline
        stim_fix.draw() 
        win.flip()
        if eye_mode:
            eye.send_message('trial {} new trial baseline phase 1'.format(t))
        core.wait(t_baseline) 
        
        #Present colored square
        stim_sq.lineColorSpace = 'rgb255'
        stim_sq.lineColor = [128,128,128] # grey
        stim_sq.fillColorSpace = 'rgb255'
        stim_sq.fillColor = reps[t]
       
        respond = [] # respond, but fixed duration of colored square
        clock.reset() # for latency measurements
        if eye_mode:
            eye.send_message('trial {} color phase 2'.format(t))
        while clock.getTime() < t_stim:
            stim_sq.draw()
            win.flip()
            if not respond:
                respond = event.waitKeys(maxWait=t_stim-clock.getTime(),keyList=p.buttons, timeStamped=clock)
        
        if respond:
            response, latency = respond[0]
        else:
            response, latency = ('missing', np.nan)
        if eye_mode:
            eye.send_message('trial {} response {} phase 3'.format(t,round(latency,2)))    
               
        # ITI
        stim_fix.draw() 
        win.flip()
        ITI = np.round(random.uniform(t_ITI[0],t_ITI[1]),2)
        if eye_mode:
            eye.send_message('trial {} ITI {} phase 4'.format(t,ITI))    
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
        DF.loc[t] = [
                int(subject_ID),      # subject
                int(t),               # trial number
                int(reps[t][0]), # r
                int(reps[t][1]), # g 
                int(reps[t][2]), # b
                round(latency,8), # RT
                ITI               # ITI
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














