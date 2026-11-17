"""
Tapping experiment - SPEECH (informed) group.

The trial screen shows the trial number on top and the sentence; PRE_SOUND_S later the sound starts
(and is recorded) while the screen stays on; the screen goes blank when the sound ends. All trial logic
is in tapping_functions.run_session (shared with the acoustic group).
"""
import tapping_functions as tf

SETUP = 'laptop'          # 'lab' or 'laptop'
GROUP = 'speech'          # do not change: this is the speech-group script

params = tf.load_params(SETUP)
tf.run_session(params, SETUP, GROUP)