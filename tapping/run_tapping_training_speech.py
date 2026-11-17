"""
Tapping TRAINING - SPEECH (informed) group.

Explanation screens, listening example, tapping example (with a figure of the taps), stumble example and a
6-sequence test run. All logic is in tapping_training_functions.run_training (shared with the acoustic group).
Afterwards start the main experiment: run_tapping_speech.py.
"""
import tapping_training_functions as tf

SETUP = 'laptop'          # 'lab' or 'laptop'
GROUP = 'speech'          # do not change: this is the speech-group script

params = tf.load_params(SETUP)
tf.run_training(params, SETUP, GROUP)