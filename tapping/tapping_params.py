"""
Parameters of the tapping experiment - LAB version (booth, loopback cable).
For test runs on the laptop use tapping_params_laptop.py (it imports this file and changes a few values).

Trial timeline (automatic, no key presses between trials):
    trial screen appears: speech group = trial number on top + the sentence; acoustic group = 'Durchgang n'
    -> PRE_SOUND_S later the sound starts (30 sounds: lead-in, S1, A, S2, B); the trial screen stays on
    -> the screen goes blank when the sound ends
    -> recording continues REC_POST_S after the sound ends (late taps)
    -> SAVE_MARGIN_S (+ ITI_S) later the next trial screen (fixed pacing: saving the wav happens here)

Recording: every trial is saved as a RAW wav (exactly the samples the audio interface delivered: no filtering,
no normalization). The recording starts REC_PRE_S before the sound. Where the sound starts in the recording
is logged per trial (sound_start_in_rec_s), so every tap can be put on the stimulus time line later.
"""
import os

#=====================
# PATHS
#=====================
MAIN_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'   # CHECK: server path in the lab
STIM_DIR = os.path.join(MAIN_DIR, 'tapping', 'stimuli_tapping')
PARTICIPANT_FOLDER = '{pid}_{group}'                           #   stimuli_tapping/P01_speech/ (build_tapping_stimuli.py)
DATA_DIR = os.path.join(MAIN_DIR, 'tapping', 'data')




# files inside a participant folder
STIM_NPY    = 'tapping_stimuli.npy'
STIM_META   = 'tapping_stim_meta.json'
STIM_DESIGN = 'design_stim.csv'             # design table + stim_row / n_samples / words / tokens
STIM_EVENTS = 'events_stim.csv'             # 30 rows per trial: actual onset of every sound (ms from file start)

# output (one set per session, nothing is ever overwritten)
OUTPUT_FILE_PATTERN = 'tapping_{pid}_{group}_{time}.tsv'           # one row per trial
REC_DIR_PATTERN     = 'tapping_rec_{pid}_{group}_{time}'           # folder with one wav per trial
REC_FILE_PATTERN    = 'trial{trial:03d}.wav'
STIM_INFO_PATTERN   = 'tapping_stiminfo_{pid}_{group}_{time}.csv'  # snapshot: the sounds that were loaded
STIM_META_PATTERN   = 'tapping_stimmeta_{pid}_{group}_{time}.json' # snapshot: settings / files used
EVENTS_COPY_PATTERN = 'tapping_events_{pid}_{group}_{time}.csv'    # copy of events_stim.csv (onsets for analysis)

#=====================
# AUDIO
#=====================
# INPUT = the interface that records. OUTPUT = where the sound plays:
#   OUTPUT_DEVICE_NAME = None and OUTPUT_DEVICE_INDEX = None -> the input device plays too: ONE full-duplex stream,
#   playback and recording on the same clock (needed for the loopback channel). Use this in the lab.
INPUT_DEVICE_NAME   = 'US-4x4HR'    # CHECK in the booth (part of the device name; must match exactly one device)
INPUT_DEVICE_INDEX  = None          # set the DeviceIndex if several devices match (printed at startup)
OUTPUT_DEVICE_NAME  = None
OUTPUT_DEVICE_INDEX = None
FS            = 44100               # must equal fs in STIM_META
N_OUT         = 2                   # stimuli are mono -> copied to both outputs
N_IN          = 2                   # input channels opened (inputs 1 and 2)
LATENCY_CLASS = 1                   # PsychPortAudio reqlatencyclass (CHECK in the booth; 3 worked on the laptop)
STIM_GAIN     = 1.0                 # multiplies every stimulus
CAPTURE_SPARE_S = 30                # capture buffer = longest trial + this

# 0-based input channels
MIC_CHANNEL      = 1                # input 2: desk microphone (taps)
LOOPBACK_CHANNEL = 0                # input 1: cable from an output (the sound as it actually came out)
REC_CHANNELS     = [LOOPBACK_CHANNEL, MIC_CHANNEL]   # saved in this order: wav channel 1 = sound, 2 = taps
REC_CHANNEL_NAMES = ['loopback', 'mic']

#=====================
# RECORDING
#=====================
REC_PRE_S   = 0.5                   # recording starts this long before the sound (keeps the start-up click out)
REC_POST_S  = 1.0                   # recording continues this long after the sound (late taps)
REC_SUBTYPE = 'FLOAT'               # wav sample format: 'FLOAT' = 32-bit float, exactly as recorded

#=====================
# ONLINE TAP CHECK -> console + break-screen feedback (works on a COPY, the saved wav is never changed)
#=====================
TAP_HIGHPASS_HZ   = 1000            # tap detection: same values as tapping_hardware_test.py
TAP_THRESHOLD_MAD = 20
TAP_REFRACTORY_MS = 150
TAP_SMOOTH_MS     = 1
# every sound gets a window from halfway to the previous sound to halfway to the next one; taps are counted per window
RECORDED_WELL_MIN_SOUNDS = 25       # a sequence is 'recorded well' if at least this many of its 30 sounds got a tap
DOUBLE_TAP_MIN_SOUNDS    = 2        # a sequence has 'double taps' if at least this many sounds got 2+ taps
ITI_CAP_MS               = 100      # inter-tap intervals outside IOI_MS +- this are 'off'
                                    # (only between neighbouring sounds that got exactly one tap each)
ITI_OFF_MAX_FRAC         = 0.2      # a sequence 'did not follow' if more than this fraction of its intervals are off
BLOCK_WARN_MIN_TRIALS    = 3        # a warning appears on the break screen if at least this many sequences of the
                                    # block had that problem

#=====================
# DESIGN
#=====================
N_BLOCKS             = 9
TRIALS_PER_BLOCK     = 24
N_EVENTS             = 30
IOI_MS               = 450          # must equal ioi_ms in STIM_META
MAX_TRIALS_PER_BLOCK = None         # None = all; a number = only the first n trials of each block (test runs)

#=====================
# TIMING (seconds)
#=====================
FIRST_TRIAL_DELAY_S = 1.0           # block start -> first trial screen
PRE_SOUND_S         = 1.5           # trial screen alone -> sound onset (the screen stays until the sound ends)
SAVE_MARGIN_S       = 0.3           # end of recording -> next trial screen: time to save the wav (fixed pacing)
ITI_S               = 0.0           # extra blank time on top of SAVE_MARGIN_S
QUIT_POLL_S         = 0.02          # how often the quit key is checked while a sound plays

#=====================
# KEYS
#=====================
CONTINUE_KEYS = ['1', '2', '3', '4']
QUIT_KEY      = 'escape'            # works on every screen and during the sound

#=====================
# SCREEN
#=====================
WIN_SIZE        = [1920, 1080]
FULLSCREEN      = True
SCREEN          = 1                 # 1 = booth, 0 = control room
MONITOR         = 'testMonitor'
UNITS           = 'cm'
TEXT_HEIGHT     = 1.0
WRAP_WIDTH      = 40.0
SENTENCE_HEIGHT = 2.0
NUMBER_HEIGHT   = 1.0               # trial number above the sentence (speech group)
NUMBER_POS_Y    = 4.0               # its vertical position (UNITS above the centre; the sentence is at the centre)

#=====================
# TRIAL SCREEN (by group) - TEMPORARY, final choice later
#=====================
# 'sentence' = trial number (NUMBER_ABOVE_TEXT) on top + the words in the centre
# 'trial_number' = TRIAL_NUMBER_TEXT in the centre
# n counts within each block (as in detection), the same n in both groups
PRE_SOUND_SCREEN      = {'speech': 'sentence', 'acoustic': 'trial_number'}
NUMBER_ABOVE_TEXT     = '{n}'
TRIAL_NUMBER_TEXT     = 'Durchgang {n}'
CAPITALIZE_FIRST_WORD = True                # 'im Moor ...' -> 'Im Moor ...' (word lists stay lowercase)

#=====================
# SESSION DIALOG
#=====================
# the group is fixed by the run script (run_tapping_speech.py / run_tapping_acoustic.py)
DIALOG_FIELDS = {'participant_id': 'P01',                       # upper-cased, as in the stimulus builder
                 'start_block': list(range(1, N_BLOCKS + 1))}   # > 1 only to resume after a crash

#=====================
# OUTPUT COLUMNS (one row per trial)
#=====================
DESIGN_COLUMNS = (['participant', 'group', 'seed', 'trial', 'block', 'sentence_type', 'sentence_id', 'exposure',
                   'role', 'n_jitters', 'jitter_pattern']
                  + [f'{p}_{c}' for p in ('A', 'S2', 'B')
                     for c in ('jitter', 'interval', 'event', 'size_ms', 'direction', 'shift_ms')]
                  + ['S2_critical', 'S2_with_noise', 'critical_with_noise', 'jitter_distance',
                     'clean_events_between', 'S2_boundary_status', 'stim_row', 'n_samples', 'words', 'tokens'])
SESSION_COLUMNS = ['session_time', 'setup', 'params_file', 'trial_in_block', 'pre_sound_screen', 'screen_text']
TIMING_COLUMNS = ['t_screen_on', 't_screen_off', 't_rec_scheduled', 't_rec_start', 't_capture_start',
                  't_sound_onset', 't_sound_offset', 'sound_start_in_rec_s']           # psychtoolbox GetSecs clock
REC_COLUMNS = ['rec_file', 'rec_channels', 'rec_n_samples', 'rec_s', 'overflow', 'peak_per_channel']
CHECK_COLUMNS = ['online_n_taps', 'online_n_taps_during', 'online_n_sounds_tapped', 'online_n_sounds_double',
                 'online_iti_median_ms', 'online_iti_off_frac', 'online_recorded_well', 'online_flag_double',
                 'online_flag_timing', 'late_start_next_s']
OUTPUT_COLUMNS = DESIGN_COLUMNS + SESSION_COLUMNS + TIMING_COLUMNS + REC_COLUMNS + CHECK_COLUMNS

#=====================
# TEXTS (German)
#=====================
# Order: main_welcome -> main_task -> main_blocks -> main_breaks -> start -> block 1 -> break -> ... -> end
# (resuming at a later block shows only 'start').
# placeholders: {n_blocks}, {n_trials} (TRIALS_PER_BLOCK), {block_min} (minutes per block, rounded up),
#               {block}, {n_good} / {n_played} (sequences recorded well / played in the block),
#               {feedback} (the fb_ texts that apply, or nothing)
_CONTINUE = 'Drücken Sie eine beliebige Taste, um fortzufahren.'

TEXTS = {
    'speech': {
        'main_welcome': ('Gut gemacht! Sie haben das Training abgeschlossen.\n\n'
                         'Jetzt beginnt das eigentliche Experiment.\n\n' + _CONTINUE),

        'main_task': ('Ihre Aufgabe ist dieselbe wie im Training.\n\n'
                      'Sie hören Klangfolgen aus Rauschklängen und Wörtern. Lesen Sie die Wörter auf dem Bildschirm. '
                      'Tippen Sie bei jedem Klang einmal. Tippen Sie genau dann, wenn der Klang beginnt. '
                      'Tippen Sie bei den Rauschklängen und bei den Wörtern. Beginnen Sie beim ersten Klang. '
                      'Tippen Sie weiter bis zum letzten Klang.\n\n' + _CONTINUE),

        'main_blocks': ('Das Experiment besteht aus {n_blocks} Blöcken mit jeweils {n_trials} Klangfolgen.\n\n'
                        'Innerhalb eines Blocks beginnt die nächste Klangfolge automatisch einige Sekunden nach dem '
                        'Ende der vorherigen.\n\n' + _CONTINUE),

        'main_breaks': ('Am Ende jedes Blocks sehen Sie, wie gut Ihr Tippen aufgezeichnet wurde.\n\n'
                        'Danach können Sie so lange Pause machen, wie Sie möchten. Der nächste Block beginnt erst, '
                        'wenn Sie eine Taste drücken.\n\n' + _CONTINUE),

        'start': ('Jeder Block dauert etwa {block_min} Minuten.\n\n'
                  'Bereit? Drücken Sie eine beliebige Taste, um zu beginnen.'),

        'break': ('Block {block} von {n_blocks} geschafft!\n\n'
                  'Ihr Tippen wurde in {n_good} von {n_played} Klangfolgen gut aufgezeichnet.\n\n'
                  '{feedback}'
                  'Machen Sie eine kurze Pause.\n\n'
                  'Drücken Sie eine beliebige Taste, wenn Sie bereit sind weiterzumachen.'),

        # break-screen feedback: each one is added to {feedback} when it applies (see ONLINE TAP CHECK)
        'fb_missing': ('Einige Ihrer Tipper wurden nicht aufgezeichnet. '
                       'Bitte tippen Sie etwas fester, neben dem Mikrofon.'),     
        'fb_double':  'Bei einigen Klängen haben Sie zweimal getippt. Bitte tippen Sie nur einmal pro Klang.',
        'fb_timing':  ('Ihr Tippen ist den Klängen nicht genau gefolgt. Versuchen Sie, genau in dem Moment zu tippen, '
                       'in dem ein Klang beginnt.'),

        'end': ('Geschafft! Sie haben diesen Teil des Experiments abgeschlossen. Vielen Dank!\n\n'
                'Bitte geben Sie der Versuchsleitung Bescheid, dass Sie fertig sind.'),
    },
}

# acoustic group: same screens; only the task description differs (nothing that suggests speech)
TEXTS['acoustic'] = dict(TEXTS['speech'])
TEXTS['acoustic'].update({
    'main_task': ('Ihre Aufgabe ist dieselbe wie im Training.\n\n'
                  'Sie hören Klangfolgen aus Rauschklängen und Tönen. Sie sehen die Nummer des Durchgangs auf dem '
                  'Bildschirm. Tippen Sie bei jedem Klang einmal. Tippen Sie genau dann, wenn der Klang beginnt. '
                  'Tippen Sie bei den Rauschklängen und bei den Tönen. Beginnen Sie beim ersten Klang. '
                  'Tippen Sie weiter bis zum letzten Klang.\n\n' + _CONTINUE),
})
INTRO_SCREENS = ['main_welcome', 'main_task', 'main_blocks', 'main_breaks']   # shown before 'start' (block 1 only)