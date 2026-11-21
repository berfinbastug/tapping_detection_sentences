"""
Switch experiment - DETECTION PHASE 1 (training + block 1), for BOTH modes.

Phase 1 is acoustic for everyone: training as in detection/, then block 1 ('Durchgang n' before each sound),
then the end screen. You choose the key mapping (counterbalance) here; phase 2 reads it back from this output.
The mode (switch / control) is read from the participant's tapping design and shown in a check dialog.

Before: detection/generate_detection_table.py (same ID, group acoustic, sizes [35, 55, 75]; laptop, Python >= 3.9)
        -> detection/build_detection_stimuli.py (same ID, group acoustic)
Trial logic, training, audio and keys: detection/detection_functions.py (unchanged).
Settings: detection/detection_params.py (lab) or detection_params_laptop.py + switch/switch_detection_params.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..', 'detection')]   # switch/ and detection/ modules

import detection_functions as ef
import switch_detection_params as sp
import switch_detection_functions as sdf

SETUP = 'laptop'          # 'lab' or 'laptop'
PHASE = 1                 # do not change: this is the phase-1 script (training + block 1)

params = ef.load_params(SETUP)
sdf.run_phase(params, sp, SETUP, PHASE)