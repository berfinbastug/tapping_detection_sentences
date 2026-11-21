"""
Switch experiment - TAPPING PHASE 1 (blocks 1-3), for BOTH modes.

Phase 1 is acoustic for everyone: 'Durchgang n' on the trial screen, acoustic texts, and after every trial the
focus question (keys 1-4). The mode (switch / control) is read from the participant's design table and shown in a
check dialog before the window opens. The session ends after block 3.

Before: switch/switch_design_table.py (ID + mode) -> tapping/build_tapping_stimuli.py (same ID, group acoustic)
        -> tapping/run_tapping_training_acoustic.py
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
PHASE = 1                 # do not change: this is the phase-1 script (blocks 1-3)

params = tf.load_params(SETUP)
stf.run_phase(params, sp, SETUP, PHASE)