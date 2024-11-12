"""
PsychoPy EyeLink functions using pylink
"""

# IMPORTANT: If you use a screen units other than 'pix', please bear in mind
# that the drift-correction target position needs to be specified in screen
# pixel coordinates, and the gaze coordinates returned by pylink are also in
# screen pixels, with the top-left corner of the screen as the origin (0,0).

# import libraries
import pylink, numpy, os, time
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy # this file needs to be in experiment directory
from psychopy import visual, core, event, monitors
from PIL import Image
from IPython import embed as shell

global tk,dataFileName,dataFolder
tk = None
dataFileName = None
dataFolder = None
    
def config(subject_ID,task):
    # established a link to the tracker
    global tk, dataFileName,dataFolder
    tk = pylink.EyeLink('100.1.1.1')
    # Open an EDF data file EARLY
    # Note that the file name cannot exceeds 8 characters
    # please open eyelink data files early to record as much info as possible
    cwd = os.getcwd()
    dataFolder = os.path.join(cwd,'LogFiles','sub-{}'.format(subject_ID)) 
    if not os.path.exists(dataFolder): os.makedirs(dataFolder)
    print subject_ID
    print task
    dataFileName = str(subject_ID)+".EDF"; # can't be longer than 8 letters
    tk.openDataFile(dataFileName)
        


def run_calibration(win,scnWidth,scnHeight):
    # Initialize custom graphics for camera setup & drift correction
    # you MUST specify the physical properties of your monitor first, otherwise you won't be able to properly use
    # different screen "units" in psychopy. 
    global tk, dataFileName
    
    # call the custom calibration routine "EyeLinkCoreGraphicsPsychopy.py", instead of the default routines that were implemented in SDL
    genv = EyeLinkCoreGraphicsPsychoPy(tk, win)
    pylink.openGraphicsEx(genv)

    # need to put the tracker in offline mode before we change its configrations
    tk.setOfflineMode()

    # sampling rate, 250, 500, 1000, or 2000; this command won't work for EyeLInk II/I
    tk.sendCommand('sample_rate 1000')

    # inform the tracker the resolution of the subject display
    tk.sendCommand("screen_pixel_coords = 0 0 %d %d" % (scnWidth-1, scnHeight-1))

    # save display resolution in EDF data file for Data Viewer integration purposes
    tk.sendMessage("DISPLAY_COORDS = 0 0 %d %d" % (scnWidth-1, scnHeight-1))

    # specify the calibration type, H3, HV3, HV5, HV13 (HV = horiztonal/vertical), 
    tk.sendCommand("calibration_type = HV5") # tk.setCalibrationType('HV9') also works, see the Pylink manual

    # the model of the tracker, 1-EyeLink I, 2-EyeLink II, 3-Newer models (100/1000Plus/DUO)
    eyelinkVer = tk.getTrackerVersion()

    #turn off scenelink camera stuff (EyeLink II/I only)
    if eyelinkVer == 2: tk.sendCommand("scene_camera_gazemap = NO")

    # Set the tracker to parse Events using "GAZE" (or "HREF") data
    tk.sendCommand("recording_parse_type = GAZE")

    # Online parser configuration: 0-> standard/coginitve, 1-> sensitive/psychophysiological
    # the Parser for EyeLink I is more conservative, see below
    if eyelinkVer>=2: tk.sendCommand('select_parser_configuration 0')

    # get Host tracking software version
    hostVer = 0
    if eyelinkVer == 3:
        tvstr  = tk.getTrackerVersionString()
        vindex = tvstr.find("EYELINK CL")
        hostVer = int(float(tvstr[(vindex + len("EYELINK CL")):].strip()))

    # specify the EVENT and SAMPLE data that are stored in EDF or retrievable from the Link
    tk.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT")
    tk.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,FIXUPDATE,SACCADE,BLINK,BUTTON,INPUT")
    if hostVer>=4: 
        tk.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,HTARGET,INPUT")
        tk.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,HTARGET,INPUT")
    else:          
        tk.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS,INPUT")
        tk.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT")
        
    # show some instructions here.
    txt = visual.TextStim(win, text = '(For Researcher) Press ENTER TWICE to see calibration options')
    txt.draw()
    win.flip()
    event.waitKeys()

    # set up the camera and calibrate the tracker at the beginning of each block
    tk.doTrackerSetup()
    # take the tracker offline
    tk.setOfflineMode()
    pylink.pumpDelay(50)


def start_recording():
    # start recording EDF file
    global tk, dataFileName
      
    error = tk.startRecording(1,1,1,1)
    pylink.pumpDelay(100) # wait for 100 ms to make sure data of interest is recorded
    tk.sendMessage('start recording')
    
    #determine which eye(s) are available
    eyeTracked = tk.eyeAvailable() 
    if eyeTracked==2: eyeTracked = 1
        
        
def stop_recording(timestr):
    # stop recording, transfer EDF file to stimulus computer, rename and timestamp
    global tk, dataFileName, dataFolder
    tk.sendMessage('stop recording')
    tk.stopRecording() # stop recording
    tk.setOfflineMode()
    tk.closeDataFile()
    pylink.pumpDelay(50)
    # rename file upon transfer
    original_name = os.path.splitext(dataFileName)
    new_name = 'sub-'+ original_name[0] +'_task-decision_eye_' + timestr + original_name[1] # sub-xxx_task-decision_eye_timestring.EDF 
    tk.receiveDataFile(dataFileName, os.path.join(dataFolder, new_name))
    print('EDF file successfully transferred {}'.format(new_name))
    #close the link to the tracker, graphics
    tk.close()
    pylink.closeGraphics()


def stop_skip_save():
    # exiting early from task, don't save EDF
    global tk
    tk.stopRecording() # stop recording
    # close the EDF data file
    tk.setOfflineMode()
    tk.closeDataFile()
    pylink.pumpDelay(50)
    #close the link to the tracker, graphics
    tk.close()
    pylink.closeGraphics()
    print('Aborted EDF file abandoned')
    
    
def send_message(msg):
    # send msg to EDF file   
    global tk
    tk.sendMessage(msg)
    pylink.pumpDelay(100)
   
 
def pause_stop_recording():
    # stop recording during the break
    global tk
    tk.stopRecording() # stop recording
    # close the EDF data file
    tk.setOfflineMode()
    tk.sendMessage('break pause recording')
    pylink.pumpDelay(50)
    
    
def run_drift_correction(win,scnWidth, scnHeight):
    # drift correction after pauses
    global tk
    txt = visual.TextStim(win, text = 'Drift correction: maintain fixation on the target...')
    txt.draw()
    win.flip()
    core.wait(3)
    try:
        tk.sendMessage('drift correction')
        err = tk.doDriftCorrect(scnWidth/2, scnHeight/2,1,1)
        error = tk.startRecording(1,1,1,1)
        pylink.pumpDelay(100) # wait for 100 ms to make sure data of interest is recorded
    except:
        tk.sendMessage('drift correction failed doing full setup')
        tk.doTrackerSetup()
        error = tk.startRecording(1,1,1,1)
        pylink.pumpDelay(100) # wait for 100 ms to make sure data of interest is recorded
    