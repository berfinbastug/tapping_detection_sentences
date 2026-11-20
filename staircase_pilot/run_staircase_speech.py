"""
Staircase pilot (2IFC) - SPEECH (informed) group.

Before each trial the sentence is shown on the screen for PRE_SOUND_S; during the sound the screen shows
'1' (presentation 1) and '2' (presentation 2); after the sound the key reminder (LEFT = 1st, RIGHT = 2nd).
All trial logic is in staircase_functions.run_session.
"""
import staircase_functions as sf

SETUP = 'laptop'          # 'lab' or 'laptop'
GROUP = 'speech'          # do not change: this is the speech-group script

params = sf.load_params(SETUP)
sf.run_session(params, SETUP, GROUP)