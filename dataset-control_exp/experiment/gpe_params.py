"""
PARAMETERS Pupil dilation offers a time-window on prediction erros
"""

# Screen-specific parameters lab B.00.80A
scnWidth, scnHeight = (1920, 1080)
screen_width        = 53.5 # centimeters
screen_dist         = 58.0

# response buttons
buttons = ['lalt','ralt'] # 0,1
button_names = ['Left ALT', 'Right ALT']


# shades of GREEN
colors = [[76,154,68],
          [157,193,131],
          [0,168,107],
          [3,121,112],
          [138,154,91],
          [75,124,89]]

letters         = ['A','D','I','O','R','T']
freqs           = [80,80,40,40,20,20]
oddball_numbers = ['1','2','3','4','5','6','7','8','9']

# to be used with shades of green (orange, baby blue, pink, purple)
oddball_colors  = [[245,127,32],[159,201,235],[115,82,162],[229,181,212]]

grey = [128,128,128]

# Timing seconds
t_stim = 0.75 # stimulus presentation duration (colored square, colored square + letter)
t_feed = 0.3  # feedback tone duration
t_ITI = [3.5,5.5] # pupil ITI period to draw from

# Size  screen 1920 x 1080, units in pixels
sqw = 120 # colored square width
sqh = 120 # colored square height
lh  = 100 # letter size
lp = (0.0,6.0) # letter position, push up to center
fh  = 50  # fixation cross height
ww = 1000 # wrap width of instructions text

# font of letters (not instructions)
font = u'Bookman Old Style'
