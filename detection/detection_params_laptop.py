"""
Parameters of the detection experiment - LAPTOP TEST version.
Everything comes from detection_params.py; only the values below are changed.
"""
import os
from detection_params import *          # noqa: F401,F403  (all lab values)

#=====================
# PATHS (local)
#=====================
MAIN_DIR   = '/Users/bastugb/Desktop/tapping_detection_sentences'
STIM_DIR   = os.path.join(MAIN_DIR, 'detection', 'stimuli')
DATA_DIR   = os.path.join(MAIN_DIR, 'detection', 'data_test')
TRAINING_STIM_DIR = os.path.join(MAIN_DIR, 'detection', 'training_stimuli')
TRAINING_DESIGN   = os.path.join(MAIN_DIR, 'detection', 'design_tables_stim', 'training_design.csv')

#=====================
# AUDIO
#=====================
DEVICE_NAME = 'MacBook Pro Speakers'

#=====================
# SCREEN (small window)
#=====================
WIN_SIZE    = [1000, 650]
FULLSCREEN  = False
SCREEN      = 0
TEXT_HEIGHT = 0.55
WRAP_WIDTH  = 24.0
SENTENCE_HEIGHT = 1.2

#=====================
# DESIGN
#=====================
MAX_TRIALS_PER_BLOCK = 4             # e.g. 5 for a quick run through all blocks