"""
PRACTICE Training task, gradient prediction errors
"""
# data saved in ~/LogFiles/sub-XXX
# only colors saved, not responses

# Import necessary modules
from psychopy import core, visual, event, data, sound, gui, monitors
import random
import numpy as np
import pandas as pd
import os, time  # for paths and data
import gpe_params as p
# from IPython import embed as shell

debug_mode = False

### PARAMETERS ###
# Timing in seconds
REPS        = 2         # how many times to repeat all trials in training_balancing
t_stim      = p.t_stim  # stimulus duration
t_RT        = 1.25-t_stim  # response interval
t_ITI       = [0.5,1.0] # ITI

## need to be in correct order per participant
colors          = np.array(p.colors)
letters         = np.array(p.letters)
freqs           = np.array(p.freqs)
oddball_numbers = np.array(p.oddball_numbers)
oddball_colors  = np.array(p.oddball_colors)

## load CSV file for balancing stimuli
training_trials = pd.read_csv(os.path.join('stimuli','practice_training_balancing.csv'))
color_perms     = pd.read_csv(os.path.join('stimuli','color_permutations.csv'))
letter_perms    = pd.read_csv(os.path.join('stimuli','letter_permutations.csv')) # shuffled version of color perms
                  
# Get subject number
#subject_ID = int(input('Please Enter Subject ID:  '))
g = gui.Dlg()
g.addField('Subject Number:')
g.show()
subject_ID = int(g.data[0])

## Create LogFile folder cwd/LogFiles
cwd = os.getcwd()
logfile_dir = os.path.join(cwd,'LogFiles','sub-{}'.format(subject_ID)) 
if not os.path.isdir(logfile_dir):
    os.makedirs(logfile_dir)

# subject_ID = 1
if subject_ID:
    ### Define stimuli for this subject & save file
    output_stimuli  = os.path.join(logfile_dir,'sub-{}_colors.csv'.format(subject_ID ))
    if not os.path.exists(output_stimuli): # only create stimuli if file does not exist for this subject
        idx = int(subject_ID-1)
        subj_colors     = colors[np.array(color_perms.loc[idx])-1]
        subj_letters    = letters[np.array(letter_perms.loc[idx])-1]
        subj_oddcolor   = oddball_colors[random.randint(1,4)-1]
        DFS = pd.DataFrame()
        DFS['subject']      = np.repeat(subject_ID,len(subj_letters))
        DFS['letter']       = np.array(subj_letters)
        DFS['frequency']    = np.array(freqs) # MAKE SURE THIS MATCHES THE OUTPUT FILE
        DFS['r']            = [c[0] for c in subj_colors]
        DFS['g']            = [c[1] for c in subj_colors]
        DFS['b']            = [c[2] for c in subj_colors]
        DFS['oddcolor_r']   = np.repeat(subj_oddcolor[0],len(subj_letters))
        DFS['oddcolor_g']   = np.repeat(subj_oddcolor[1],len(subj_letters))
        DFS['oddcolor_b']   = np.repeat(subj_oddcolor[2],len(subj_letters))
        DFS.to_csv(output_stimuli)
    else: # load existing file
        DFS = pd.read_csv(output_stimuli)
        subj_colors = [[DFS['r'][i],DFS['g'][i],DFS['b'][i]] for i in range(len(DFS))]
        subj_letters = list(DFS['letter'])
        subj_oddcolor = [DFS['oddcolor_r'][0],DFS['oddcolor_g'][0],DFS['oddcolor_b'][0]]
        
    print('odd ball color = {}'.format(subj_oddcolor))
    
    # Counterbalance response buttons
    if np.mod(subject_ID,2) == 0:
        buttons = p.buttons # 0,1
        button_names = p.button_names
    else:
        buttons = np.flip(p.buttons,0)
        button_names = np.flip(p.button_names,0)
  
    # Set-up window:
    mon = monitors.Monitor('myMac15', width=p.screen_width, distance=p.screen_dist)
    mon.setSizePix((p.scnWidth, p.scnHeight))
    win = visual.Window((p.scnWidth, p.scnHeight),color=p.grey,colorSpace='rgb255',monitor=mon,fullscr=not debug_mode,units='pix',allowStencil=True,autoLog=False)
    win.setMouseVisible(False)
    
    # Set-up stimuli and timing
    welcome_txt = 'PRACTICE! Look for oddballs! \
    \n\nMost of the time you will see a letter. \
    \nHowever, on some trials, you will see a number AND/OR a color that \'does not belong\'. \
    \nWhen that happens, then you must indicate it is an ODDBALL!\
    \n\n\n<Press any button to continue>'
    
    oddball_txt = 'ODDBALL NUMBERS are: 1,2,3,4,5,6,7,8 and 9 (but not 0!). \
    \nThe ODDBALL COLOR is shown here in the square: \
    \n\nWhen you see i) ANY of these numbers OR ii) this color OR iii) BOTH together, then you must indicate ODDBALL!\
    \n\nResponse keys are: ODDBALL = {}, Regular = {} \
    \n\nPlease put on the headphones to hear the feedback.\
    \n\n\n<Press any button to begin>'.format(button_names[1],button_names[0])

    stim_instr  = visual.TextStim(win, color='black', pos=(0.0, 0.0), wrapWidth=p.ww)  # can't really center, and TextBox doesn't work, stupid!
    odd_sq      = visual.Rect(win, width=p.sqw,height=p.sqh,autoLog=None,pos=(210, 150))
    stim_letter = visual.TextStim(win,font=p.font, color='black', pos=p.lp, height=p.lh)
    stim_fix    = visual.TextStim(win, text='+', color='black', pos=(0.0, 0.0), height=p.fh)
    stim_sq     = visual.Rect(win, width=p.sqw, height=p.sqh, autoLog=None, pos=(0.0, 0.0))
    feed        = sound.Sound(500, secs=p.t_feed) 
    feed_miss   = visual.TextStim(win, text='Too slow!', color='blue', pos=(0.0, 50), wrapWidth=8)  # can't really center, and TextBox doesn't work, stupid!
    feed_error  = visual.TextStim(win, text='Error!', color='red', pos=(0.0, 50), wrapWidth=8)
    clock       = core.Clock()
    
    # Set conditqions and stimulus list    
    ALL_TRIALS  = pd.concat([training_trials]*REPS, ignore_index=True)    
    ALL_TRIALS = ALL_TRIALS.sample(frac=1).reset_index(drop=True) ## SHUFFLE ORDER OF TRIALS

    # Welcome instructions
    stim_instr.setText(welcome_txt,)
    stim_instr.draw()
    win.flip()
    core.wait(0.25)
    event.waitKeys()
   
    # Odd ball instructions
    stim_instr.setText(oddball_txt)
    stim_instr.draw()
    odd_sq.lineColorSpace = 'rgb255'
    odd_sq.lineColor = p.grey # grey
    odd_sq.fillColorSpace = 'rgb255'
    odd_sq.fillColor = subj_oddcolor
    ## odd_sq.setColor(subj_oddcolor,colorSpace='rgb255')
    odd_sq.draw()
    win.flip()
    core.wait(0.25)
    event.waitKeys()
    
    # Wait a few seconds before first trial
    stim_fix.draw()
    win.flip()
    core.wait(3)
    
    #### TRIAL LOOP ###
    for t in range(len(ALL_TRIALS)):
        
        stim_fix.setColor('black')
        # current trial values
        this_color_idx  = int(ALL_TRIALS['color'][t]) 
        this_letter_idx = int(ALL_TRIALS['letter'][t])
        this_freq = ALL_TRIALS['frequency'][t]
        this_answer = ALL_TRIALS['oddball'][t]
        
        # Check if oddball LETTER or NUMBER?
        this_letter = []
        if this_letter_idx == 7: # oddball number
            this_letter = oddball_numbers[random.randint(0, 8)]  # choose random 1-9
        elif this_letter_idx == 100: # oddball color + letter
            this_letter = subj_letters[random.randint(0, 5)] # choose random letter 1-6
        else:
            this_letter = subj_letters[this_letter_idx-1]
            
        # Check if oddball COLOR?
        this_color = []
        if this_color_idx == 7: # odd ball color!!
            this_color = subj_oddcolor
        elif this_color_idx == 100: # random normal color, always paired with a number
            this_color_idx = random.randint(1, 6)
            this_color = subj_colors[this_color_idx-1] # index 0-5, not 1-6
        else:
            this_color = subj_colors[this_color_idx-1]
            
                
        # Target stimuli current trial        
        print('########## Trial {} #########'.format(t+1))
        print('Color of current trial: {}'.format(this_color))
        print('Letter of current trial: {}'.format(this_letter))
        print(this_answer)
        
        # STIMULUS (letter) & wait for response
        stim_letter.setText(this_letter)
        stim_sq.lineColorSpace = 'rgb255'
        stim_sq.lineColor = p.grey # grey
        stim_sq.fillColorSpace = 'rgb255'
        stim_sq.fillColor = this_color
        ## stim_sq.setColor(this_color,'rgb255')

        respond1 = [] # during stim
        respond2 = [] # after stim
        clock.reset() # for latency measurements
        while clock.getTime() < t_stim:
            stim_sq.draw()
            stim_letter.draw() 
            win.flip()
            if not respond1:
                respond1 = event.waitKeys(maxWait=t_stim-clock.getTime(),keyList=buttons, timeStamped=clock)
                print('stim {}'.format(respond1))
        # RESPONSE INTERVAL
        clock.reset() # for latency measurements
        while clock.getTime() < t_RT:    
            stim_fix.draw() 
            win.flip()
            if (not respond1) and (not respond2): # did not respond during stimulus or fixation!
                respond2 = event.waitKeys(maxWait=t_RT-clock.getTime(),keyList=buttons, timeStamped=clock)
                print('fix {}'.format(respond2))
        
        ## get RT and ACCURACY
        if respond2: # responded after stimulus
            response, latency = respond2[0]
            RT = round(latency,8)+t_stim
        elif respond1: # responded during stimulus
            response, latency = respond1[0]
            RT = round(latency,8)
        else: # missed
            response, RT = ('missing', np.nan)
        ## check whether correct response?
        correct = buttons[this_answer] == response
                
        ## FEEDBACK
        if (response == 'missing') or (not correct):
            feed.setSound('D', octave=3)
            if response == 'missing':
                feed_miss.draw()
                stim_fix.setColor('blue')
                stim_fix.draw() 
            else:
                feed_error.draw()
                stim_fix.setColor('red')
                stim_fix.draw() 
            feed.play()
        else:
            stim_fix.setColor('green')
            stim_fix.draw() 
        win.flip()
        ITI = np.round(random.uniform(t_ITI[0],t_ITI[1]),2)
        core.wait(ITI)
        
        # For quitting early
        keys = event.getKeys()
        if keys:
            # q quits the experiment
            if keys[0] == 'q':
                core.quit()
           
    # End screen for participants
    stim_instr.setText('Well done!')
    stim_instr.draw()
    win.flip()
    core.wait(2)
        
# Close-up   
win.close() # Close window
core.quit()














