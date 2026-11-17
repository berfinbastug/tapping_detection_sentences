"""
Detection experiment - ACOUSTIC (naive) group.

Same timeline as the speech group, but the pre-sound screen shows a fixation cross instead of the
sentence (nothing on screen may suggest speech). When the sound starts, the screen switches to the
key reminder. All trial logic is in detection_functions.run_session (shared with the speech group).
"""
import detection_functions as ef

SETUP = 'laptop'          # 'lab' or 'laptop'
GROUP = 'acoustic'        # do not change: this is the acoustic-group script

params = ef.load_params(SETUP)
ef.run_session(params, SETUP, GROUP)