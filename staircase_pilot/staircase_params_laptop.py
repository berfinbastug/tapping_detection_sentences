"""
Parameters of the staircase pilot - LAPTOP TEST version.
Everything comes from staircase_params.py; only the values below are changed.
"""
import os
from staircase_params import *          # noqa: F401,F403  (all lab values)

#=====================
# PATHS (local)
#=====================
MAIN_DIR   = '/Users/bastugb/Desktop/tapping_detection_sentences'
REPO_DIR   = MAIN_DIR
DESIGN_DIR = os.path.join(MAIN_DIR, 'staircase_pilot', 'design_tables_staircase')
STIM_DIR   = os.path.join(MAIN_DIR, 'staircase_pilot', 'stimuli')
DATA_DIR   = os.path.join(MAIN_DIR, 'staircase_pilot', 'data_test')
ITEMS_CSV    = os.path.join(MAIN_DIR, 'sws_words', 'items.csv')
SWS_INFO_CSV = os.path.join(MAIN_DIR, 'sws_words', 'sws_info.csv')

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

#=====================
# DESIGN
#=====================
MAX_TRIALS_PER_BLOCK = 5   # e.g. 5 for a quick test run