"""
Decision task, gradient prediction errors
"""
# phase 1 new trial onset
# phase 2 letter stimulus onset
# phase 3 delay onset
# phase 4 target onset
# phase 5 response made
# phase 6 pupil baseline feedback
# phase 7 feedback onset
# data saved in ~/source/sub-XXX

# Import necessary modules
from psychopy import core, visual, event, data, sound, gui, monitors
import random
import numpy as np
import pandas as pd
import os, time  # for paths and data
import gpe_params as p
# from IPython import embed as shell

debug_mode = False
eye_mode = True

### PARAMETERS ###
# Timing in seconds
REPS        = 4         # how many times to repeat all trials in decision_balancing
t_cue       = 0.2       # new trial cue
t_stim      = p.t_stim  # stimulus duration
t_delay     = 0.1       # between stimulus and target
t_target    = 2.5       # max RT duration 
t_baseline  = p.t_ITI # pupil baseline
t_ITI       = p.t_ITI # ITI

## load CSV file for balancing stimuli
decision_trials = pd.read_csv(os.path.join('stimuli','decision_balancing.csv'))
          
# Get subject number
#subject_ID = int(input('Please Enter Subject ID:  '))
g = gui.Dlg()
g.addField('Subject Number:')
g.show()
subject_ID = int(g.data[0])

## Create LogFile folder cwd/LogFiles
cwd = os.getcwd()
logfile_dir = os.path.join(cwd,'source','sub-{}'.format(subject_ID)) 
if not os.path.isdir(logfile_dir):
    os.makedirs(logfile_dir)

# subject_ID = 1
if subject_ID:    
        
    ## GET SUBJECT-SPECIFIC STIMULI
    DFS = pd.read_csv(os.path.join(logfile_dir,'sub-{}_colors.csv'.format(subject_ID)))
    letters = np.array(DFS['letter'])
    r = np.array(DFS['r'])
    g = np.array(DFS['g'])
    b = np.array(DFS['b'])
    colors = []
    for c in np.arange(0,len(letters)):    
        colors.append([r[c],g[c],b[c]])
    colors = np.array(colors)
            
    # Counterbalance response buttons
    if np.mod(subject_ID,2) == 0:
        buttons = p.buttons # 0,1
        button_names = p.button_names
    else:
        buttons = np.flip(p.buttons,0)
        button_names = np.flip(p.button_names,0)

        
    ## output file name with time stamp prevents any overwriting of data
    timestr = time.strftime("%Y%m%d-%H%M%S") 
    output_filename = os.path.join(logfile_dir,'sub-{}_task-letter_color_visual_decision_beh_{}.csv'.format(subject_ID, timestr))
    # output dataframe
    cols = ['subject','trial_num','letter','frequency','r','g','b','PBASE','ITI','match','button','correct','RT']
    DF = pd.DataFrame(columns=cols)
        
    # Set-up window:
    mon = monitors.Monitor('myMac15', width=p.screen_width, distance=p.screen_dist)
    mon.setSizePix((p.scnWidth, p.scnHeight))
    win = visual.Window((p.scnWidth, p.scnHeight),color=p.grey,colorSpace='rgb255',monitor=mon,fullscr=not debug_mode,units='pix',allowStencil=True,autoLog=False)
    win.setMouseVisible(False)
    
    # Set-up stimuli and timing
    instr1 ='Based on the first part of the experiment, you may have noticed that certain letters are frequently shown with certain colors.\
    \nA letter that was shown often together with a certain color is considered a MATCH.\
    \n\nIn this decision task, you will need to indicate if you think that the letter MATCHES the color shown.\
    \nIf you are unsure, just guess.\
    \n\nYou will get feedback on each trial (correct, wrong or too slow of a response).\
    \nBefore beginning, we will present to you the sound of the feedback tones.\
    \nPlease put on the headphones to hear the feedback.\
    \n\n\n<Press any button to continue>'
    
    instr2a = 'When you are CORRECT you will hear this tone...'
    instr2b = '<Press any button to continue>'
    
    instr3a = 'When you are INCORRECT (or too slow) you will hear this tone...'
    instr3b = '<Press any button to continue>'

    instr4 = 'During the task, the letter is shown first, then quickly afterwards the color is shown.\
    \nIndicate if the letter matches the color or not BEFORE the color disappears from the screen.\
    \n\nIMPORTANT: Response keys are: MATCH = {}, No Match = {}\
    \n\nYou can give your response as soon as you see the color.\
    \nRemember to keep looking at the + whenever it is on screen!\
    \n\nYou will get 3 breaks during the task during which you can move your head.\
    \n\n\n<Press any button to BEGIN the task>'.format(button_names[1],button_names[0]) 
    
    pauze = 'Take a short break now, you may move your head.\
    \n\n Response keys are: MATCH = {}, No Match = {}\
    \n\n\n<Press any button to CONTINUE the task>'.format(button_names[1],button_names[0]) 
    
    stim_instr  = visual.TextStim(win, color='black', pos=(0.0, 0.0), wrapWidth=p.ww )  # can't really center, and TextBox doesn't work, stupid!
    stim_letter = visual.TextStim(win,font=p.font, color='black', pos=p.lp, height=p.lh)
    stim_fix    = visual.TextStim(win,text='+',color='black', pos=(0.0, 0.0), height=p.fh)
    feed_miss   = visual.TextStim(win, text='Too slow!', color='blue', pos=(0.0, 50), wrapWidth=8)  # can't really center, and TextBox doesn't work, stupid!
    stim_sq     = visual.Rect(win, height=p.sqh, width=p.sqw, fillColor='black',pos=(0.0, 0.0))
    feed        = sound.Sound(500, secs=p.t_feed) 
    clock       = core.Clock()
    
    # Set conditions and stimulus list    
    ALL_TRIALS  = pd.concat([decision_trials]*REPS, ignore_index=True)    
    ALL_TRIALS = ALL_TRIALS.sample(frac=1).reset_index(drop=True) ## SHUFFLE ORDER OF TRIALS
    if debug_mode:
        ALL_TRIALS = ALL_TRIALS.iloc[0:10,:] # just get first 12 trials
    
    ## INSTRUCTIONS 1
    stim_instr.setText(instr1)
    stim_instr.draw()
    win.flip()
    core.wait(0.25)
    event.waitKeys()
    
    ## INSTRUCTIONS 2 - PLAY CORRECT TONE
    stim_instr.setText(instr2a)
    stim_instr.draw()
    ## play correct tone a couple times
    win.flip()
    core.wait(2)
    feed.setSound('B', octave=4)    
    feed.play()
    core.wait(2)
    feed.setSound('B', octave=4)    
    feed.play()
    core.wait(2)
    feed.setSound('B', octave=4)    
    feed.play()
    core.wait(3) # force wait
    stim_instr.setText(instr2b)
    stim_instr.draw()
    win.flip()
    event.waitKeys()
    
    
    ## INSTRUCTIONS 3 - PLAY ERROR/MISS TONE
    stim_instr.setText(instr3a)
    stim_instr.draw()
    win.flip()
    core.wait(2)
    feed.setSound('D', octave=3)    
    feed.play()
    core.wait(2)
    feed.setSound('D', octave=3)    
    feed.play()
    core.wait(2)
    feed.setSound('D', octave=3)    
    feed.play()
    core.wait(3) # force wait
    stim_instr.setText(instr3b)
    stim_instr.draw()
    win.flip()
    event.waitKeys()
    
    
    
    ## INSTRUCTIONS 4 - RESPONSE KEYS
    stim_instr.setText(instr4)
    stim_instr.draw()
    win.flip()
    core.wait(1)
    event.waitKeys()
    
    ### CONFIG & CALIBRATION EYE-TRACKER ###
    if eye_mode:
        import funcs_pylink as eye
        task = 'Dec'
        eye.config(subject_ID,task)
        eye.run_calibration(win,p.scnWidth, p.scnHeight)
        eye.start_recording()
        
    ## INSTRUCTIONS 4 - RESPONSE KEYS
    stim_instr.setText(instr4)
    stim_instr.draw()
    win.flip()
    core.wait(1)
    event.waitKeys()
        
    stim_fix.draw() # stablize gaze
    win.flip()
    core.wait(3)
    #### TRIAL LOOP ###
    for t in range(len(ALL_TRIALS)):
        
        if (t==np.floor(len(ALL_TRIALS)/4)) or (t==np.floor(len(ALL_TRIALS)/4*2)) or (t==np.floor(len(ALL_TRIALS)/4*3)):
            if eye_mode:
                eye.pause_stop_recording() # pause recording
            # take a break!
            stim_instr.setText(pauze)
            stim_instr.draw()
            win.flip()
            core.wait(0.5)
            event.waitKeys()
            if eye_mode:
                # Drift correction, when subject moves their head during breaks
                eye.run_drift_correction(win,p.scnWidth, p.scnHeight)
            # Wait a few seconds before first trial to stabilize gaze
            stim_fix.draw()
            win.flip()
            core.wait(3)
            
        # current trial values
        this_color  = int(ALL_TRIALS['color'][t])-1 # index 0-5, not 1-6
        this_letter = int(ALL_TRIALS['letter'][t])-1
        this_freq   = ALL_TRIALS['frequency'][t]
        this_answer = ALL_TRIALS['answer'][t]
        ITI = np.round(random.uniform(t_ITI[0],t_ITI[1]),2) # ITI for this trial
    
        # Target stimuli current trial        
        print('########## Trial {} #########'.format(t+1))
        print('Color of current trial: {}'.format(colors[this_color]))
        print('Letter of current trial: {}'.format(letters[this_letter]))
        print(this_answer)
        
        # New trial cue
        stim_fix.setColor(p.grey,'rgb255') #blank screen
        stim_fix.draw()
        win.flip()
        if eye_mode:
            eye.send_message('trial {} new trial cue phase 1'.format(t))
        core.wait(t_cue)
        
        # Stimulus (letter)
        stim_letter.setText(letters[this_letter])
        stim_letter.draw() 
        win.flip()
        if eye_mode:
            eye.send_message('trial {} letter stimulus phase 2'.format(t))
        core.wait(t_stim)
        
        # Short delay before target
        stim_fix.setColor(p.grey,'rgb255') #blank screen
        stim_fix.draw()
        win.flip()
        if eye_mode:
            eye.send_message('trial {} target delay phase 3'.format(t))
        core.wait(t_delay)

        # Present colored square, wait for response, or go on if too slow
        stim_sq.lineColorSpace = 'rgb255'
        stim_sq.lineColor = p.grey # grey
        stim_sq.fillColorSpace = 'rgb255'
        stim_sq.fillColor = colors[this_color]
        ## stim_sq.setColor(colors[this_color],'rgb255') # Not working version 1.90.2
        stim_sq.draw() 
        win.flip()                
        if eye_mode:
            eye.send_message('trial {} color target phase 4'.format(t))             
        clock.reset() # for latency measurements
        respond = event.waitKeys(maxWait=t_target, keyList=buttons, timeStamped=clock)
        
        if respond:
            response, latency = respond[0]
        else:
            response, latency = ('missing', np.nan)            
        if eye_mode:
            eye.send_message('trial {} response {} phase 5'.format(t,round(latency,2)))    
        ## check whether correct response?
        correct = buttons[this_answer] == response
                
        # PUPIL BASELINE
        stim_fix.setColor('black')
        if response == 'missing': # distinguish between too slow and error!
            feed_miss.draw()
            stim_fix.draw()
        else:
            stim_fix.draw() 
        win.flip()
        if eye_mode:
            eye.send_message('trial {} pupil baseline phase 6'.format(t))
        PBASE = np.round(random.uniform(t_baseline[0],t_baseline[1]),2)
        core.wait(PBASE)
        
        ## FEEDBACK
        if (response == 'missing') or (not correct):
            feed.setSound('D', octave=3)          
            if response == 'missing': # set back to just fixation
                stim_fix.draw()
                win.flip()
        else: # correct
            feed.setSound('B', octave=4)    
        feed.play()

        # ITI
        if eye_mode:
            eye.send_message('trial {} feedback tone {} phase 7'.format(t,correct))
            eye.send_message('trial {} ITI {}'.format(t,ITI))

        core.wait(ITI)
        
        # output data frame on each trial 
        DF.loc[t] = [
            int(subject_ID),             # subject
            int(t),                      # trial_num
            letters[this_letter],        # letter
            int(this_freq),              # frequency
            int(colors[this_color][0]),  # r
            int(colors[this_color][1]),  # g
            int(colors[this_color][2]),  # b
            PBASE,                       # pupil baseline
            ITI,                         # ITI
            int(this_answer),            # correct answer, 1=match,0=no match
            response,                    # button pressed
            correct,                     # correct response or error?
            round(latency,8)             # RT
        ]
        DF.to_csv(output_filename)
        
        # For quitting early
        keys = event.getKeys()
        if keys:
            # q quits the experiment
            if keys[0] == 'q':
                if eye_mode:
                    eye.stop_skip_save()
                core.quit()
                
           
    # End screen for participants
    stim_instr.setText('Well done! Data transfering.....')
    stim_instr.draw()
    win.flip()
        
    # output mean RT for Control IRF task
    output_filename = os.path.join(logfile_dir,'sub-{}_task-letter_color_visual_decision_meanRT_{}.csv'.format(subject_ID,timestr))
    cols = ['subject','meanRT']
    DF2 = pd.DataFrame(columns=cols)
    DF2.loc[0] = [int(subject_ID),np.nanmean(np.array(DF['RT']))]
    DF2.to_csv(output_filename)
    
# Close-up   
if eye_mode:
    eye.stop_recording(timestr)
win.close() # Close window
core.quit()








