"""
Switch experiment - TAPPING PHASE 2 (blocks 4-9), for BOTH modes.

mode 'switch'  -> SPEECH: number + the words on the trial screen, speech texts (read the words), and after every
                  trial the intelligibility question (keys 1-4). Run it AFTER the reveal / familiarisation.
mode 'control' -> ACOUSTIC as in phase 1: 'Durchgang n', acoustic texts, focus question.
The mode is read from the participant's design table and shown in a check dialog before the window opens.

Before: run_switch_tapping_phase1.py, run_switch_detection_phase1.py, break, (switch: reveal + familiarisation)
Trial logic, audio, recording and tap check: tapping/tapping_functions.py (unchanged).
Settings: tapping/tapping_params.py (lab) or tapping_params_laptop.py + switch/switch_tapping_params.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'tapping')]     # switch/ and tapping/ modules

import tapping_functions as tf
import switch_tapping_params as sp
import switch_tapping_functions as stf

SETUP = 'laptop'          # 'lab' or 'laptop'
PHASE = 2                 # do not change: this is the phase-2 script (blocks 4-9)

params = tf.load_params(SETUP)
stf.run_phase(params, sp, SETUP, PHASE)