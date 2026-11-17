"""
Parameters of the tapping experiment - LAPTOP TEST version (EarPods + TASCAM US-2x2HR + desk microphone).
Everything comes from tapping_params.py; only the values below are changed.
"""
import os
from tapping_params import *          # noqa: F401,F403  (all lab values)

#=====================
# PATHS (local)
#=====================
MAIN_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
STIM_DIR = os.path.join(MAIN_DIR, 'tapping', 'stimuli_tapping')
DATA_DIR = os.path.join(MAIN_DIR, 'tapping', 'data_test')

#=====================
# AUDIO: sound through the EarPods, recording through the TASCAM (two devices, no loopback)
#=====================
INPUT_DEVICE_NAME  = 'US-2x2HR'
OUTPUT_DEVICE_NAME = 'EarPods'
N_IN          = 2
LATENCY_CLASS = 3                     # worked in tapping_hardware_test.py
MIC_CHANNEL      = 1                  # input 2
LOOPBACK_CHANNEL = None               # no loopback cable here
REC_CHANNELS      = [MIC_CHANNEL]     # must be set again here (the lab file computed it from its own values)
REC_CHANNEL_NAMES = ['mic']

#=====================
# SCREEN (small window)
#=====================
WIN_SIZE        = [1000, 650]
FULLSCREEN      = False
SCREEN          = 0
TEXT_HEIGHT     = 0.55
WRAP_WIDTH      = 24.0
SENTENCE_HEIGHT = 1.2
NUMBER_HEIGHT   = 0.55
NUMBER_POS_Y    = 2.0

#=====================
# DESIGN
#=====================
MAX_TRIALS_PER_BLOCK = 2              # quick run through all blocks; None = full run