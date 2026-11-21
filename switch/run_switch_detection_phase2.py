"""
Switch experiment - DETECTION PHASE 2 (blocks 2-3, no training), for BOTH modes.

mode 'switch'  -> SPEECH: the sentence is shown before each sound; the start screen says so.
mode 'control' -> ACOUSTIC as in phase 1: 'Durchgang n' before each sound.
The key mapping (counterbalance) is NOT asked: it is read from the phase-1 output of the same participant, so the
keys cannot flip between phases. The mode is read from the participant's tapping design. Both are shown in a check
dialog before the window opens. start_block = 3 only to resume after a crash.

Before: run_switch_tapping_phase2.py (and, of course, run_switch_detection_phase1.py)
Trial logic, audio and keys: detection/detection_functions.py (unchanged).
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
PHASE = 2                 # do not change: this is the phase-2 script (blocks 2-3)

params = ef.load_params(SETUP)
sdf.run_phase(params, sp, SETUP, PHASE)