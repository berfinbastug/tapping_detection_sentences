"""
Parameters of the staircase pilot (2IFC, speech group only) - LAB version (booth, server, US-4x4HR).
For test runs on the laptop use staircase_params_laptop.py (it imports this file and changes a few values).

Goal: one detection threshold T for a phase shift -> JITTER_SIZES_MS of detection AND tapping.

Trial (2IFC):
    the same item plays twice; ONE presentation has a phase shift at interval Ik
    (word k+1 and every later word move by shift_ms; + = later / delay, - = earlier / advance).
    ISI_MS of silence from the audible offset of word 6 (presentation 1) to the onset of word 1 (presentation 2).
    Question: which presentation was irregular, the 1st or the 2nd?

Sizes: the staircases move on a fixed grid (SIZE_LEVELS_MS). Because every possible size is known in advance,
build_staircase_stimuli.py checks every trial at every size BEFORE the session. The sound of a trial is
built at run time (the size is only known then) with the same function, from the participant's word bank.
"""
import os
import numpy as np

#=====================
# PATHS
#=====================
MAIN_DIR   = '/Users/bastugb/Desktop/tapping_detection_sentences'
REPO_DIR   = MAIN_DIR                                                   # folder that contains repo/
DESIGN_DIR = os.path.join(MAIN_DIR, 'staircase_pilot', 'design_tables_staircase')   # staircase_<pid>_<group>.csv
STIM_DIR   = os.path.join(MAIN_DIR, 'staircase_pilot', 'stimuli')       # stimuli/<pid>_<group>/
DATA_DIR   = os.path.join(MAIN_DIR, 'staircase_pilot', 'data')
PARTICIPANT_FOLDER = '{pid}_{group}'

# word material (read only by build_staircase_stimuli.py)
ITEMS_CSV    = os.path.join(MAIN_DIR, 'sws_words', 'items.csv')         # from sentence_lists_to_tts.py
SWS_INFO_CSV = os.path.join(MAIN_DIR, 'sws_words', 'sws_info.csv')      # base -> sws_file (relative to MAIN_DIR)
LIST_TO_TYPE = {'items_2_4': 'S24', 'items_3_3': 'S33'}                 # sentence-list file name -> sentence code
WL_FROM      = 'S33'          # PLACEHOLDER: WL_k uses the words of S33_k until the word list exists

# files inside a participant folder (written by build_staircase_stimuli.py)
WORD_NPY    = 'staircase_words.npy'          # float32 (n_words, max_samples): every word this participant hears
WORD_INDEX  = 'staircase_words.csv'          # one row per word: bank_row, word, sws_file, n_samples, on, off
STIM_DESIGN = 'design_stim.csv'              # the design table + bank rows of the 6 words + words text
STIM_META   = 'staircase_stim_meta.json'     # settings, size grid, check results, build time

#=====================
# AUDIO
#=====================
DEVICE_NAME    = 'US-4x4HR'   # looked up by name at startup (must match exactly one device)
FS             = 44100        # must equal the word files' rate
N_OUT_CHANNELS = 2            # stimuli are mono -> copied to both ears
STIM_GAIN      = 1.0          # multiplies every stimulus
LATENCY_CLASS  = 1            # PsychPortAudio reqlatencyclass

#=====================
# STIMULUS (identical to the detection experiment)
#=====================
N_WORDS      = 6
IOI_MS       = 450            # onset-to-onset grid
LEAD_MS      = 50             # silence before word 1's onset (each presentation)
TAIL_MS      = 50             # silence after word 6's file end (each presentation)
EDGE_FADE_MS = 2              # file-edge fade (falls in the silence)
ONSET_DB     = -40            # onset/offset rule: same as the SWS trimming (trim_sws)
ENV_WIN_MS   = 5
MIN_GAP_MS   = 5              # smallest allowed silence between a word's offset and the next onset
ISI_MS       = 800            # audible offset of presentation 1 -> onset of presentation 2

#=====================
# STAIRCASE (PsychoPy's psychopy.data.StairHandler, see staircase_functions.Staircase)
#=====================
STAIRCASES   = ['delay_1', 'delay_2', 'advance_1', 'advance_2']   # 4 interleaved, independent counters
DIRECTION_SIGN = {'delay': +1, 'advance': -1}                      # delay = word comes late
START_MS     = 90.0           # every staircase starts here (= the largest size on the grid, i.e. the ceiling)
GRID_FACTOR  = 1.2            # neighbouring sizes on the grid differ by this factor
MIN_MS       = 3.0            # smallest size on the grid (floor)
SIZE_LEVELS_MS = [round(START_MS / GRID_FACTOR ** k, 2)            # 90, 75, 62.5, ... down to >= MIN_MS
                  for k in range(int(np.floor(np.log(START_MS / MIN_MS) / np.log(GRID_FACTOR))) + 1)]
N_DOWN       = 3              # 3 correct in a row -> smaller (3-down-1-up -> 79.4 % correct)
N_UP         = 1              # 1 wrong -> larger
BIG_STEP     = 2              # grid steps per move before the SWITCH_AFTER-th reversal (2 steps = x/÷ 1.44)
SMALL_STEP   = 1              # grid steps per move from that reversal on (x/÷ 1.2); StairHandler stepSizes
SWITCH_AFTER = 2              # reversals before switching from BIG_STEP to SMALL_STEP
N_DISCARD    = 3              # threshold = geometric mean of the reversals after the first N_DISCARD

#=====================
# DESIGN
#=====================
N_BLOCKS              = 4
TRIALS_PER_BLOCK      = 45
MAX_TRIALS_PER_BLOCK  = None  # None = all; a number = only the first n trials of each block (test runs)

#=====================
# TIMING (seconds)
#=====================
FIRST_TRIAL_DELAY_S = 0.5     # block start -> first pre-sound screen
PRE_SOUND_S         = 2.5     # pre-sound screen (the written sentence) -> sound onset
RESPONSE_WINDOW_S   = 4.0     # from the end of the 2nd presentation
ITI_S               = 0.8     # key press (or end of window) -> next trial's pre-sound screen
NO_RESPONSE_CORRECT = 0       # no answer within the window counts as WRONG for the staircase

#=====================
# KEYS
#=====================
RESPONSE_KEYS = ['2', '3']                 # 2 = LEFT, 3 = RIGHT
KEY_TO_PRESENTATION = {'2': 1, '3': 2}     # LEFT = 1st presentation was irregular, RIGHT = 2nd
CONTINUE_KEYS = ['1', '2', '3', '4']
QUIT_KEY      = 'escape'

#=====================
# SCREEN
#=====================
WIN_SIZE    = [1920, 1080]
FULLSCREEN  = True
SCREEN      = 1               # 1 = booth, 0 = control room
MONITOR     = 'testMonitor'
UNITS       = 'cm'
TEXT_HEIGHT = 1.0
WRAP_WIDTH  = 40.0
SENTENCE_HEIGHT = 2.0         # sentence on the pre-sound screen
LABEL_HEIGHT    = 3.0         # '1' / '2' during the two presentations

# pre-sound screen by group: 'sentence' = the words, 'trial_number' = TRIAL_NUMBER_TEXT, 'fixation' = FIXATION_TEXT
PRE_SOUND_SCREEN  = {'speech': 'sentence'}
TRIAL_NUMBER_TEXT = 'Durchgang {n}'        # n counts within each block (and within the practice)
FIXATION_TEXT     = '+'
CAPITALIZE_FIRST_WORD = True               # 'im Moor ...' -> 'Im Moor ...' (not for word lists)
INTERVAL_LABELS   = ('1', '2')             # on screen from the start of presentation 1 / presentation 2

#=====================
# PRACTICE (before block 1; NOT part of the staircases)
#=====================
N_PRACTICE        = 6         # practice trials (0 = no practice); taken from the first rows of design_stim.csv
PRACTICE_SIZE_MS  = 90.0      # must be a size on SIZE_LEVELS_MS (it was checked there)
FEEDBACK_S        = 1.5

#=====================
# SESSION DIALOG
#=====================
# the group is fixed by the run script (run_staircase_speech.py; speech group only)
DIALOG_FIELDS = {'participant_id': 'P01',
                 'start_block': [1, 2, 3, 4]}             # >1 only to resume after a crash (staircases are
                                                          # restored from the earlier output file)

#=====================
# OUTPUT
#=====================
OUTPUT_FILE_PATTERN     = 'staircase_{pid}_{group}_{time}.tsv'    # one row appended per trial
PRACTICE_OUTPUT_PATTERN = 'practice_{pid}_{group}_{time}.tsv'
THRESHOLD_FILE_PATTERN  = 'thresholds_{pid}_{group}_{time}.csv'  # one row per staircase, written at the end

DESIGN_COLUMNS = ['participant', 'group', 'seed', 'trial', 'block', 'staircase', 'direction', 'copy',
                  'staircase_trial', 'sentence_type', 'sentence_id', 'exposure', 'interval', 'moved_word',
                  'shifted_presentation', 'words']
SESSION_COLUMNS  = ['session_time', 'params_file', 'pre_sound_screen', 'screen_text']
STIM_COLUMNS     = ['size_ms', 'shift_ms', 'stim_id', 'sound_dur_s', 'pres2_start_s']
RESPONSE_COLUMNS = ['key_name', 'response', 'correct', 'rt_from_offset_s', 'key_tDown', 'button_press_duration']
STAIRCASE_COLUMNS = ['sc_size_ms', 'sc_move', 'sc_reversal', 'sc_n_reversals', 'sc_next_size_ms']
TIMING_COLUMNS   = ['t_pre_sound_on', 't_onset_scheduled', 't_onset', 't_label2_on', 't_offset', 't_keys_on']
OUTPUT_COLUMNS = (DESIGN_COLUMNS + SESSION_COLUMNS + STIM_COLUMNS + RESPONSE_COLUMNS + STAIRCASE_COLUMNS
                  + TIMING_COLUMNS)
PRACTICE_COLUMNS = (['participant', 'group', 'session_time', 'practice_trial', 'sentence_type', 'sentence_id',
                     'interval', 'shifted_presentation', 'words', 'screen_text'] + STIM_COLUMNS
                    + RESPONSE_COLUMNS + TIMING_COLUMNS)

#=====================
# TEXTS (English) - speech group only
#=====================
# placeholders: {n_blocks}, {n_practice}, {block}, {window}, {answer}
TEXTS = {
    'speech': {
        'welcome':  ('Welcome!\n\n'
                     'Press any button to continue.'),
        'reading':  ('In this experiment you will hear computer-generated sounds.\n\n'
                     'Before you hear the sounds, you will see the written version of these unusual '
                     'computer-generated sounds on the screen. Please read it before you hear the sounds.\n\n'
                     'Press any button to continue.'),
        'task':     ('When the text disappears, you will hear the sounds twice. In one of the two, the timing '
                     'is perfectly regular. In the other, one interval is perturbed.\n\n'
                     'Your task is to find the irregular one:\n'
                     '   press 2 if you think the FIRST one was irregular\n'
                     '   press 3 if you think the SECOND one was irregular\n\n'
                     'If you are ready, press any button to start.'),
        'practice_intro': ('We start with {n_practice} practice trials. After each one you will '
                           'see whether your answer was correct.\n\nPress any button to begin.'),
        'fb_correct':     'Correct! Sound {answer} was the irregular one.',
        'fb_wrong':       'Wrong. Sound {answer} was the irregular one.',
        'fb_slow':        'Too slow. Please answer within {window} seconds after the sounds end.',
        'start':          ('Now the experiment starts: {n_blocks} blocks, no feedback.\n\n'
                           'Press any button to begin.'),
        'keys_reminder':  '2 = first          3 = second',
        'break':          ('Block {block} of {n_blocks} done.\n\nTake a short break.\n\n'
                           'Press any button when you are ready to continue.'),
        'end':            'Done! Thank you very much.',
    },
}