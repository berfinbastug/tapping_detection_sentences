"""
Tapping TRAINING - ACOUSTIC (naive) group.

Same training as the speech group, but the screens show 'Durchgang n' instead of the words and the texts never
mention speech. All logic is in tapping_training_functions.run_training (shared with the speech group).
Afterwards start the main experiment: run_tapping_acoustic.py.
"""
import tapping_training_functions as tf

SETUP = 'laptop'          # 'lab' or 'laptop'
GROUP = 'acoustic'        # do not change: this is the acoustic-group script

params = tf.load_params(SETUP)
tf.run_training(params, SETUP, GROUP)