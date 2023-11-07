"""
Test colors on monitor
"""
# Import necessary modules
from psychopy import core, visual, event, data, sound, monitors
import random
import numpy as np
import pandas as pd
import os, time  # for paths and data
import gpe_params as p
from IPython import embed as shell

debug_mode = True

### PARAMETERS ###
colors          = np.array(p.colors)
letters         = np.array(p.letters)
freqs           = np.array(p.freqs)
oddball_numbers = np.array(p.oddball_numbers)
oddball_colors  = np.array(p.oddball_colors)

# Set-up window:
mon = monitors.Monitor('myMac15', width=p.screen_width, distance=p.screen_dist)
mon.setSizePix((p.scnWidth, p.scnHeight))
win = visual.Window((p.scnWidth, p.scnHeight),color=p.grey,colorSpace='rgb255',monitor=mon,fullscr=not debug_mode,units='pix',allowStencil=True,autoLog=False)
win.setMouseVisible(False)

# regular colors
c1      = 'regular colors, push a button'
c2      = 'odd ball colors, push a button'
instr   = visual.TextStim(win, color='black', pos=(0.0, 0.0), wrapWidth=p.ww)  
stim_sq = visual.Rect(win, width=p.sqw, height=p.sqh, autoLog=None, pos=(0.0, 0.0))

# Regular colors
instr.setText(c1)
instr.draw()
win.flip()
event.waitKeys()

for c in colors:
    stim_sq.lineColorSpace = 'rgb255'
    stim_sq.lineColor = p.grey # grey
    stim_sq.fillColorSpace = 'rgb255'
    stim_sq.fillColor = c
    ##stim_sq.setColor(c,colorSpace='rgb255')
    stim_sq.draw()
    win.flip()
    print(c)
    event.waitKeys()

# Oddball colors
instr.setText(c2)
instr.draw()
win.flip()
event.waitKeys()

for c in oddball_colors:
    stim_sq.lineColorSpace = 'rgb255'
    stim_sq.lineColor = p.grey # grey
    stim_sq.fillColorSpace = 'rgb255'
    stim_sq.fillColor = c
        #stim_sq.setColor(c,colorSpace='rgb255')
    stim_sq.draw()
    win.flip()
    print(c)
    event.waitKeys()

win.close() # Close window
core.quit()














