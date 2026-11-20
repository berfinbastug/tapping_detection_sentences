"""
Staircase pilot (2IFC) - ACOUSTIC (naive) group.

Before each trial the trial number is shown on the screen for PRE_SOUND_S; during the sound the screen shows
'1' (presentation 1) and '2' (presentation 2); after the sound the key reminder (LEFT = 1st, RIGHT = 2nd).
All trial logic is in staircase_functions.run_session (shared with the speech group).
"""
import staircase_functions as sf

SETUP = 'laptop'          # 'lab' or 'laptop'
GROUP = 'acoustic'        # do not change: this is the acoustic-group script

params = sf.load_params(SETUP)
sf.run_session(params, SETUP, GROUP)