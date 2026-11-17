"""
Parameters of the tapping TRAINING - LAPTOP TEST version (EarPods + TASCAM US-2x2HR + desk microphone).
Everything comes from tapping_training_params.py; only the values below are changed.
"""
import os
from tapping_training_params import *          # noqa: F401,F403  (all lab values)

#=====================
# PATHS (local)
#=====================
MAIN_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
STIM_DIR = os.path.join(MAIN_DIR, 'tapping', 'training_stimuli_tapping')
DATA_DIR = os.path.join(MAIN_DIR, 'tapping', 'data_test')

#=====================
# AUDIO: sound through the EarPods, recording through the TASCAM (two devices, no loopback)
#=====================
INPUT_DEVICE_NAME  = 'US-2x2HR'
OUTPUT_DEVICE_NAME = 'EarPods'
N_IN          = 2
LATENCY_CLASS = 3
MIC_CHANNEL      = 1
LOOPBACK_CHANNEL = None
REC_CHANNELS      = [MIC_CHANNEL]     # must be set again here
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
FIG_WIDTH       = 22.0
FIG_Y           = 4.0
FIG_MARK_SIZE   = 0.4
FIG_TAP_Y       = 3.2
FIG_TAP_LEN     = 0.5
FIG_TEXT_Y      = -2.5