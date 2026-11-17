"""
Tapping experiment - ACOUSTIC (naive) group.

Same timeline as the speech group, but the trial screen shows 'Durchgang n' instead of the
number + sentence (nothing on screen may suggest speech). All trial logic is in
tapping_functions.run_session (shared with the speech group).
"""
import tapping_functions as tf

SETUP = 'laptop'          # 'lab' or 'laptop'
GROUP = 'acoustic'        # do not change: this is the acoustic-group script

params = tf.load_params(SETUP)
tf.run_session(params, SETUP, GROUP)