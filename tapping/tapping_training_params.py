"""
Parameters of the tapping TRAINING - LAB version (booth, loopback cable).
For test runs on the laptop use tapping_training_params_laptop.py (it imports this file and changes a few values).
The training is its own script; afterwards the experimenter starts the main experiment (run_tapping_*.py).
Hardware, recording and tap-check values are copies of tapping_params.py: change them in both files.

Flow (TEXTS keys in brackets):
    welcome -> hear -> example_intro -> LISTEN example (no tapping)
    -> structure -> task -> TAP example -> see_taps -> figure + feedback
          passed: LEFT = again, RIGHT = continue | failed: feedback + fb_retry -> again (MAX_EXAMPLE_ATTEMPTS)
    -> shifts -> stumble_intro -> STUMBLE example -> written feedback (same rules)
    -> test_intro -> TEST RUN (6 sequences, automatic like the main trials) -> test_pass / test_fail
          failed: run again (up to MAX_TEST_RUNS runs in total)
    -> end
If the last allowed attempt fails, the window closes (the experimenter decides what to do).

Trial screen and timing as in the main experiment: speech = trial number on top + the words; acoustic =
'Durchgang n'; the screen appears PRE_SOUND_S before the sound and disappears when the sound ends.
"""
import os

#=====================
# PATHS
#=====================
MAIN_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'   # CHECK: server path in the lab
STIM_DIR = os.path.join(MAIN_DIR, 'tapping', 'training_stimuli_tapping')
DATA_DIR = os.path.join(MAIN_DIR, 'tapping', 'data')


STIM_NPY    = 'training_stimuli.npy'
STIM_META   = 'training_stim_meta.json'
STIM_DESIGN = 'design_stim.csv'
STIM_EVENTS = 'events_stim.csv'

OUTPUT_FILE_PATTERN = 'tapping_training_{pid}_{group}_{time}.tsv'          # one row per training trial
REC_DIR_PATTERN     = 'tapping_training_rec_{pid}_{group}_{time}'          # one wav per training trial
REC_FILE_PATTERN    = '{phase}_r{rnd}_t{trial}.wav'                        # rnd = attempt / test run
STIM_META_PATTERN   = 'tapping_training_stimmeta_{pid}_{group}_{time}.json'
EVENTS_COPY_PATTERN = 'tapping_training_events_{pid}_{group}_{time}.csv'

#=====================
# AUDIO (same as tapping_params.py)
#=====================
INPUT_DEVICE_NAME   = 'US-4x4HR'    # CHECK in the booth
INPUT_DEVICE_INDEX  = None
OUTPUT_DEVICE_NAME  = None          # None = full duplex on the input device (needed for the loopback)
OUTPUT_DEVICE_INDEX = None
FS            = 44100
N_OUT         = 2
N_IN          = 2
LATENCY_CLASS = 1                   # CHECK in the booth
STIM_GAIN     = 1.0
CAPTURE_SPARE_S = 30

MIC_CHANNEL      = 1
LOOPBACK_CHANNEL = 0
REC_CHANNELS     = [LOOPBACK_CHANNEL, MIC_CHANNEL]
REC_CHANNEL_NAMES = ['loopback', 'mic']

#=====================
# RECORDING (same as tapping_params.py)
#=====================
REC_PRE_S   = 0.5
REC_POST_S  = 1.0
REC_SUBTYPE = 'FLOAT'

#=====================
# TAP CHECK (same rules as the break screen of the main experiment; works on a copy of the recording)
#=====================
TAP_HIGHPASS_HZ   = 1000
TAP_THRESHOLD_MAD = 20
TAP_REFRACTORY_MS = 150
TAP_SMOOTH_MS     = 1
RECORDED_WELL_MIN_SOUNDS = 25
DOUBLE_TAP_MIN_SOUNDS    = 2
ITI_CAP_MS               = 100
ITI_OFF_MAX_FRAC         = 0.2
# training only: in a shifted sequence the interval with the shift and the next EXCLUDE_AFTER_SHIFT intervals
# (people are still correcting) are left out of the timing check
EXCLUDE_AFTER_SHIFT      = 2
# a sequence PASSES if it was recorded well, has no double taps and its timing was not off

#=====================
# ATTEMPTS
#=====================
MAX_EXAMPLE_ATTEMPTS = 3            # tap example and stumble example: failed attempts allowed before the window closes
MAX_TEST_RUNS        = 4            # test run: first run + up to 3 repetitions
TEST_PASS_MIN        = 4            # the test run passes if at least this many of its 6 sequences pass

#=====================
# DESIGN
#=====================
N_EVENTS = 30
IOI_MS   = 450

#=====================
# TIMING (seconds)
#=====================
FIRST_TRIAL_DELAY_S = 1.0           # key press -> trial screen (examples and the start of a test run)
PRE_SOUND_S         = 1.5           # trial screen alone -> sound onset
SAVE_MARGIN_S       = 0.3           # test run: end of recording -> next trial screen
ITI_S               = 0.0
QUIT_POLL_S         = 0.02

#=====================
# KEYS
#=====================
CONTINUE_KEYS = ['1', '2', '3', '4']
RESPONSE_KEYS = ['2', '3']          # 2 = LEFT, 3 = RIGHT (as in detection)
QUIT_KEY      = 'escape'

#=====================
# SCREEN
#=====================
WIN_SIZE        = [1920, 1080]
FULLSCREEN      = True
SCREEN          = 1
MONITOR         = 'testMonitor'
UNITS           = 'cm'
TEXT_HEIGHT     = 1.0
WRAP_WIDTH      = 40.0
SENTENCE_HEIGHT = 2.0
NUMBER_HEIGHT   = 1.0
NUMBER_POS_Y    = 4.0

PRE_SOUND_SCREEN      = {'speech': 'sentence', 'acoustic': 'trial_number'}
NUMBER_ABOVE_TEXT     = '{n}'
TRIAL_NUMBER_TEXT     = 'Durchgang {n}'
CAPITALIZE_FIRST_WORD = True

#=====================
# FEEDBACK FIGURE (tap example): ■ = noise sound, ● = word / tone sound, taps as ticks underneath
#=====================
FIG_WIDTH     = 36.0                # width of the time line (UNITS)
FIG_Y         = 5.0                 # height of the row of sounds (UNITS above the centre)
FIG_MARK_SIZE = 0.6                 # size of ■ / ●
FIG_TAP_Y     = 3.8                 # height of the tap ticks
FIG_TAP_LEN   = 0.8
FIG_TAP_WIDTH = 3                   # line width (pixels)
FIG_COLOR     = 'white'
FIG_TEXT_Y    = -3.0                # the feedback text under the figure

#=====================
# SESSION DIALOG
#=====================
DIALOG_FIELDS = {'participant_id': 'P01'}

#=====================
# OUTPUT COLUMNS (one row per training trial)
#=====================
TRIAL_COLUMNS = ['participant', 'group', 'session_time', 'setup', 'params_file', 'phase', 'round', 'trial',
                 'item_id', 'sentence_type', 'words', 'stim_id', 'jittered', 'jitter_part', 'jitter_interval',
                 'jitter_event', 'shift_ms', 'stim_row', 'n_samples', 'tokens', 'screen_text']
TIMING_COLUMNS = ['t_screen_on', 't_screen_off', 't_rec_scheduled', 't_rec_start', 't_capture_start',
                  't_sound_onset', 't_sound_offset', 'sound_start_in_rec_s']
REC_COLUMNS = ['rec_file', 'rec_channels', 'rec_n_samples', 'rec_s', 'overflow', 'peak_per_channel']
CHECK_COLUMNS = ['online_n_taps', 'online_n_taps_during', 'online_n_sounds_tapped', 'online_n_sounds_double',
                 'online_iti_median_ms', 'online_iti_off_frac', 'online_iti_excluded', 'online_recorded_well',
                 'online_flag_double', 'online_flag_timing', 'online_passed', 'checked']
OUTPUT_COLUMNS = TRIAL_COLUMNS + TIMING_COLUMNS + REC_COLUMNS + CHECK_COLUMNS

#=====================
# TEXTS (German)
#=====================
# Bold (**...**) in the drafts is not shown: PsychoPy TextStim has no inline bold.
# placeholders: {words} (example word list), {n_pass} / {n_test} (test run), {feedback} (fb_ lines that apply)
_CONTINUE = 'Drücken Sie eine beliebige Taste, um fortzufahren.'
_CHOICE = ('Drücken Sie LINKS, um es noch einmal zu versuchen.\n\n'
           'Drücken Sie RECHTS, um fortzufahren.')

TEXTS = {
    'speech': {
        'welcome': ('Willkommen zum Experiment!\n\n'
                    'In diesem Teil sollen Sie zu Klängen mittippen. Zuerst erklären wir Ihnen die Aufgabe '
                    'Schritt für Schritt.\n\n' + _CONTINUE),

        'hear': ('Sie hören Folgen von kurzen Klängen. Jede Klangfolge besteht aus 30 Klängen und dauert etwa '
                 '14 Sekunden. Sie ist immer gleich aufgebaut:\n\n'
                 '1. 6 Rauschklänge\n'
                 '2. 6 computergenerierte Wörter\n'
                 '3. 6 Rauschklänge\n'
                 '4. dieselben 6 Wörter noch einmal\n'
                 '5. 6 Rauschklänge\n\n'
                 'Manchmal bilden die Wörter einen Satz, manchmal sind sie eine Liste von Wörtern.\n\n'
                 'Das klingt vielleicht kompliziert, deshalb hören wir uns ein Beispiel an.\n\n' + _CONTINUE),

        'example_intro': ('Vor jeder Klangfolge erscheinen die Wörter auf dem Bildschirm. Sie bleiben dort, während '
                          'die Klangfolge abgespielt wird, damit Sie sie lesen können.\n\n'
                          'Hören Sie jetzt erst einmal nur zu. Sie müssen noch nicht tippen.\n\n'
                          'Drücken Sie eine beliebige Taste, um ein Beispiel zu hören.'),

        'structure': ('Das haben Sie gerade gehört:\n\n'
                      '■ ■ ■ ■ ■ ■\n'
                      '{words}\n'
                      '■ ■ ■ ■ ■ ■\n'
                      '{words}\n'
                      '■ ■ ■ ■ ■ ■\n\n'
                      '■ = Rauschklang\n\n' + _CONTINUE),

        'task': ('Ihre Aufgabe ist es, zu den Klängen mitzutippen.\n\n'
                 'Auf dem Tisch sehen Sie ein kleines Mikrofon. Tippen Sie mit Ihrem Zeigefinger neben dem '
                 'Mikrofon auf den Tisch.\n\n'
                 'Tippen Sie bei jedem Klang einmal, genau in dem Moment, in dem er beginnt. Das gilt für die '
                 'Rauschklänge und für die Wörter, vom ersten bis zum letzten Klang.\n\n'
                 'Wenn Sie einen Klang verpassen, machen Sie einfach beim nächsten weiter. Versuchen Sie aber, '
                 'keinen zu verpassen.\n\n'
                 'Probieren wir es aus. Drücken Sie eine beliebige Taste, wenn Sie bereit sind.'),

        'see_taps': 'Drücken Sie eine beliebige Taste, um zu sehen, wie Sie getippt haben.',

        'fb_pass': 'Gut gemacht! Ihr Tippen wurde gut aufgezeichnet.\n\n' + _CHOICE,
        'fb_missing': ('Einige Ihrer Tipper wurden nicht aufgezeichnet. '
                       'Bitte tippen Sie etwas fester, neben dem Mikrofon.'),
        'fb_double': 'Bei einigen Klängen haben Sie zweimal getippt. Bitte tippen Sie nur einmal pro Klang.',
        'fb_timing': ('Ihr Tippen ist den Klängen nicht genau gefolgt. Versuchen Sie, genau in dem Moment zu '
                      'tippen, in dem ein Klang beginnt.'),
        'fb_retry': ('Versuchen wir es noch einmal. Drücken Sie eine beliebige Taste, um das Beispiel zu '
                     'wiederholen.'),

        'shifts': ('In manchen Klangfolgen gerät der Rhythmus ins Stolpern: Ein Klang kommt etwas zu früh oder '
                   'etwas zu spät. Das kann bei den Rauschklängen oder bei den Wörtern passieren.\n\n'
                   'Hören Sie genau hin und tippen Sie weiter genau in dem Moment, in dem ein Klang beginnt, auch '
                   'wenn der Rhythmus stolpert.\n\n' + _CONTINUE),

        'stumble_intro': ('Probieren wir ein Beispiel aus. Sie hören dieselbe Klangfolge wie vorhin, aber diesmal '
                          'gerät der Rhythmus einmal ins Stolpern. In diesem Beispiel ist das Stolpern extra groß, '
                          'damit man es leicht hört.\n\n'
                          'Wie vorhin sehen Sie zuerst die Wörter, dann wird die Klangfolge abgespielt. Tippen Sie '
                          'bei jedem Klang mit.\n\n'
                          'Drücken Sie eine beliebige Taste, wenn Sie bereit sind.'),

        'test_intro': ('Jetzt machen wir einen kurzen Probedurchlauf mit 6 Klangfolgen.\n\n'
                       'Er läuft genauso ab wie das eigentliche Experiment: Die nächste Klangfolge beginnt '
                       'automatisch einige Sekunden nach dem Ende der vorherigen. Tippen Sie bei jedem Klang mit, '
                       'vom ersten bis zum letzten.\n\n'
                       'In manchen Klangfolgen gerät der Rhythmus ins Stolpern, aber das Stolpern wird viel kleiner '
                       'sein als im Beispiel.\n\n'
                       'Drücken Sie eine beliebige Taste, um zu beginnen.'),

        # DRAFT - to be revised
        'test_pass': ('Probedurchlauf beendet!\n\n'
                      'Ihr Tippen wurde in {n_pass} von {n_test} Klangfolgen gut aufgezeichnet. Gut gemacht!\n\n'
                      + _CONTINUE),
        'test_fail': ('Probedurchlauf beendet!\n\n'
                      'Ihr Tippen wurde in {n_pass} von {n_test} Klangfolgen gut aufgezeichnet.\n\n'
                      '{feedback}'
                      'Wir machen den Probedurchlauf noch einmal.\n\n'
                      'Drücken Sie eine beliebige Taste, um zu beginnen.'),

        'end': 'Das Training ist beendet.\n\nDrücken Sie eine beliebige Taste, um den Bildschirm zu schließen.',
    },
}

# acoustic group: the screens that mention words or sentences (nothing on screen may suggest speech)
TEXTS['acoustic'] = dict(TEXTS['speech'])
TEXTS['acoustic'].update({
    'hear': ('Sie hören Folgen von kurzen Klängen. Jede Klangfolge besteht aus 30 Klängen und dauert etwa '
             '14 Sekunden. Sie ist immer gleich aufgebaut:\n\n'
             '1. 6 Rauschklänge\n'
             '2. 6 Töne\n'
             '3. 6 Rauschklänge\n'
             '4. 6 Töne\n'
             '5. 6 Rauschklänge\n\n'
             'Das klingt vielleicht kompliziert, deshalb hören wir uns ein Beispiel an.\n\n' + _CONTINUE),

    'example_intro': ('Vor jeder Klangfolge erscheint die Nummer des Durchgangs auf dem Bildschirm. Sie bleibt dort, '
                      'während die Klangfolge abgespielt wird.\n\n'
                      'Hören Sie jetzt erst einmal nur zu. Sie müssen noch nicht tippen.\n\n'
                      'Drücken Sie eine beliebige Taste, um ein Beispiel zu hören.'),

    'structure': ('Das haben Sie gerade gehört:\n\n'
                  '■ ■ ■ ■ ■ ■\n'
                  '● ● ● ● ● ●\n'
                  '■ ■ ■ ■ ■ ■\n'
                  '● ● ● ● ● ●\n'
                  '■ ■ ■ ■ ■ ■\n\n'
                  '■ = Rauschklang\n'
                  '● = Ton\n\n' + _CONTINUE),

    'task': ('Ihre Aufgabe ist es, zu den Klängen mitzutippen.\n\n'
             'Auf dem Tisch sehen Sie ein kleines Mikrofon. Tippen Sie mit Ihrem Zeigefinger neben dem '
             'Mikrofon auf den Tisch.\n\n'
             'Tippen Sie bei jedem Klang einmal, genau in dem Moment, in dem er beginnt. Das gilt für die '
             'Rauschklänge und für die Töne, vom ersten bis zum letzten Klang.\n\n'
             'Wenn Sie einen Klang verpassen, machen Sie einfach beim nächsten weiter. Versuchen Sie aber, '
             'keinen zu verpassen.\n\n'
             'Probieren wir es aus. Drücken Sie eine beliebige Taste, wenn Sie bereit sind.'),

    'shifts': ('In manchen Klangfolgen gerät der Rhythmus ins Stolpern: Ein Klang kommt etwas zu früh oder '
               'etwas zu spät. Das kann bei den Rauschklängen oder bei den Tönen passieren.\n\n'
               'Hören Sie genau hin und tippen Sie weiter genau in dem Moment, in dem ein Klang beginnt, auch '
               'wenn der Rhythmus stolpert.\n\n' + _CONTINUE),

    'stumble_intro': ('Probieren wir ein Beispiel aus. Sie hören dieselbe Klangfolge wie vorhin, aber diesmal '
                      'gerät der Rhythmus einmal ins Stolpern. In diesem Beispiel ist das Stolpern extra groß, '
                      'damit man es leicht hört.\n\n'
                      'Wie vorhin sehen Sie zuerst die Nummer des Durchgangs, dann wird die Klangfolge abgespielt. '
                      'Tippen Sie bei jedem Klang mit.\n\n'
                      'Drücken Sie eine beliebige Taste, wenn Sie bereit sind.'),
})