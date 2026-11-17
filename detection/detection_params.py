"""
Parameters of the detection experiment - LAB version (booth, server, US-4x4HR).
For test runs on the laptop use detection_params_laptop.py (it imports this file and changes a few values).

Trial timeline:
    blank (ITI_S after the key press)
    -> pre-sound screen for PRE_SOUND_S: the sentence (speech group) or the trial number (acoustic group),
       nothing else on the screen
    -> sound starts; at the same moment the screen shows the key reminder (LEFT: ... RIGHT: ...)
    -> sound plays to the end (presses before its end are ignored)
    -> response window: RESPONSE_WINDOW_S from the END of the sound
    -> blank; next trial ITI_S after the press (or after the window ends if there is no response)
"""
import os

#=====================
# PATHS
#=====================
MAIN_DIR   = '/Users/bastugb/Desktop/tapping_detection_sentences'  
STIM_DIR   = os.path.join(MAIN_DIR, "detection", 'stimuli')               
PARTICIPANT_FOLDER = '{pid}_{group}'                        
DATA_DIR   = os.path.join(MAIN_DIR, "detection", 'data')

# files inside a participant folder
STIM_NPY    = 'detection_stimuli.npy'
STIM_META   = 'detection_stim_meta.json'
STIM_INDEX  = 'detection_stim_index.csv'                    # per trial: words, word onsets/offsets
STIM_DESIGN = 'design_stim.csv'                             # the design table + stim_row / n_samples

# in the OUTPUT section, after OUTPUT_FILE_PATTERN:
STIM_INFO_PATTERN   = 'stiminfo_{pid}_{group}_{time}.csv'    # snapshot of this participant's sounds (per trial)
STIM_META_PATTERN   = 'stimmeta_{pid}_{group}_{time}.json'   # snapshot of the stimulus settings / files used

# in the TRAINING section, after TRAINING_NPY:
TRAINING_INDEX      = 'training_stim_index.csv'
TRAINING_META       = 'training_stim_meta.json'


#=====================
# AUDIO
#=====================
DEVICE_NAME    = 'US-4x4HR'   # looked up by name at startup (must match exactly one device)
FS             = 44100        # must equal fs in STIM_META
N_OUT_CHANNELS = 2            # stimuli are mono -> copied to both ears
STIM_GAIN      = 1.0          # multiplies every stimulus (replaces the old 'stimulus_rms')
LATENCY_CLASS  = 1            # PsychPortAudio reqlatencyclass

#=====================
# DESIGN
#=====================
N_BLOCKS              = 3
TRIALS_PER_BLOCK      = 54
MAX_TRIALS_PER_BLOCK  = None  # None = all; a number = only the first n trials of each block (test runs)

#=====================
# TIMING (seconds)
#=====================
FIRST_TRIAL_DELAY_S    = 0.5    # block start -> first pre-sound screen
PRE_SOUND_S            = 2.5    # pre-sound screen (sentence / fixation) -> sound onset
RESPOND_DURING_SOUND   = False  # presses before the sound ends are ignored
RESPONSE_WINDOW_S      = 4.0    # from the end of the sound
ITI_S                  = 1.0    # key press (or end of window) -> next trial's pre-sound screen
STOP_SOUND_ON_RESPONSE = False  # the sentence always plays to the end

#=====================
# KEYS
#=====================
RESPONSE_KEYS = ['2', '3']                 # 2 = LEFT, 3 = RIGHT
CONTINUE_KEYS = ['1', '2', '3', '4']
QUIT_KEY      = 'escape'
# counterbalance condition -> what each key means
KEY_MAP = {1: {'2': 'regular',   '3': 'irregular'},    # LEFT regular,   RIGHT irregular
           2: {'2': 'irregular', '3': 'regular'}}      # LEFT irregular, RIGHT regular

#=====================
# SCREEN
#=====================
WIN_SIZE    = [1920, 1080]
FULLSCREEN  = True
SCREEN      = 1               # 1 = booth, 0 = control room
MONITOR     = 'testMonitor'
UNITS       = 'cm'
TEXT_HEIGHT = 1.0           # instruction screens (long German texts: check they fit)
WRAP_WIDTH  = 40.0
SENTENCE_HEIGHT = 2.0         # sentence on the pre-sound screen

#=====================
# PRE-SOUND SCREEN (by group)
#=====================
# 'sentence' = the words, 'trial_number' = TRIAL_NUMBER_TEXT, 'fixation' = FIXATION_TEXT
PRE_SOUND_SCREEN  = {'speech': 'sentence', 'acoustic': 'trial_number'}
TRIAL_NUMBER_TEXT = 'Durchgang {n}'   # n counts within each block (and within each practice round)
FIXATION_TEXT     = '+'               # also the listening screen of the examples
CAPITALIZE_FIRST_WORD = True  # 'im Moor ...' -> 'Im Moor ...' on screen

#=====================
# SESSION DIALOG
#=====================
# the group is fixed by the run script (run_detection_speech.py / run_detection_acoustic.py)
DIALOG_FIELDS = {'participant_id': 'P01',              # must match the design table names
                 'counterbalance': [1, 2],
                 'start_block': [1, 2, 3]}             # >1 only to resume after a crash

#=====================
# OUTPUT
#=====================
OUTPUT_FILE_PATTERN = 'detection_{pid}_{group}_{time}.tsv'   # one row appended per trial

DESIGN_COLUMNS = ['participant', 'group', 'seed', 'trial', 'block', 'sentence_type', 'sentence_id',
                  'exposure', 'role', 'jittered', 'interval', 'moved_word', 'size_ms', 'direction',
                  'shift_ms', 'boundary_status', 'stim_id', 'stim_row', 'n_samples', 'correct_response']
SESSION_COLUMNS = ['session_time', 'counterbalance', 'params_file', 'pre_sound_screen', 'screen_text']
RESPONSE_COLUMNS = ['key_name', 'response', 'correct',               # response: regular / irregular / none
                    'rt_from_offset_s', 'rt_from_onset_s',           # offset = end of the sound
                    'key_tDown', 'button_press_duration']
TIMING_COLUMNS = ['t_pre_sound_on', 't_onset_scheduled', 't_onset', 't_keys_on',   # psychtoolbox GetSecs clock
                  't_offset', 'sound_dur_s']
OUTPUT_COLUMNS = DESIGN_COLUMNS + SESSION_COLUMNS + RESPONSE_COLUMNS + TIMING_COLUMNS

#=====================
# TRAINING (demo + practice)
#=====================
TRAINING_STIM_DIR   = os.path.join(MAIN_DIR, "detection", 'training_stimuli')            # from build_training_stimuli.py
TRAINING_NPY        = 'training_stimuli.npy'
TRAINING_DESIGN     = os.path.join(MAIN_DIR, "detection", 'design_tables_stim', 'training_design.csv')
TRAINING_OUTPUT_PATTERN = 'training_{pid}_{group}_{time}.tsv'

DEMO_PREVIEW_S      = 2.5     # written words / dots before an example
DEMO_FIX_TO_SOUND_S = 0.5     # '+' -> example sound
DEMO_AFTER_SOUND_S  = 1.0     # end of example sound -> next screen
DEMO_GAP            = '      '  # extra space shown at the shift point of the irregular example
DOT                 = '●'  # acoustic group: one dot per sound

PRACTICE_PASS       = 4       # correct out of the practice trials needed to go on
MAX_AUTO_ROUNDS     = 3       # below PRACTICE_PASS practice repeats automatically up to this many rounds;
                              # after that the participant chooses (LEFT again / RIGHT start)
RESHUFFLE_REPEATS   = True    # repeated practice rounds in a new random order
FEEDBACK_S          = 1.5     # practice feedback screen

TRAINING_COLUMNS = ['participant', 'group', 'session_time', 'counterbalance', 'phase', 'round', 'trial_in_round',
                    'item_id', 'sentence_type', 'role', 'interval', 'shift_ms', 'stim_id', 'correct_response',
                    'screen_text', 'key_name', 'response', 'correct', 'rt_from_offset_s',
                    't_pre_sound_on', 't_onset', 't_offset']

#=====================
# BREAK-SCREEN SCORE
#=====================
SCORE_METHOD = 'balanced'     # 'balanced' = mean of % correct on irregular and on regular trials
                              #   (always pressing the same button -> 50 %); 'plain' = % of all trials
SCORE_HIGH, SCORE_MID = 80, 60   # >= HIGH: 'great', >= MID: 'well', else: 'low'

#=====================
# TEXTS (German)
#=====================
# placeholders: {left} {right} (answer on each button), {window} (response window, s), {n_blocks},
# {n_practice}, {block}, {score}, {n_correct}, {answer}
RESPONSE_LABELS = {'regular': 'regelmäßig', 'irregular': 'unregelmäßig'}

_CONTINUE = 'Drücken Sie eine beliebige Taste, um fortzufahren.'
_BREAK_END = ('Machen Sie eine kurze Pause.\n\n'
              'Drücken Sie eine beliebige Taste, wenn Sie bereit sind weiterzumachen.')

TEXTS = {
    'speech': {
        'welcome': ('Willkommen zum letzten Teil des Experiments!\n\n'
                    'Sie hören dieselbe computergenerierte Sprache wie vorhin. Manchmal bilden die Wörter einen Satz, '
                    'manchmal sind sie nur eine Liste von Wörtern ohne Zusammenhang.\n\n'
                    'Diesmal sollen Sie nicht mittippen. Stattdessen ist Ihre Aufgabe, genau auf das Timing der Wörter '
                    'zu achten.\n\n' + _CONTINUE),


        'regular': ('In manchen Wortfolgen kommen die Wörter in vollkommen gleichmäßigen Abständen, '
                    'wie das Ticken einer Uhr.\n\n'
                    'Diese Wortfolgen nennen wir REGELMÄSSIG.\n\n'
                    'Drücken Sie eine beliebige Taste, um ein Beispiel für eine regelmäßige Wortfolge zu hören.'),


        'irregular': ('In anderen Wortfolgen gerät der Rhythmus einmal ins Stolpern: An einer Stelle kommt ein Wort '
                      'etwas zu früh oder etwas zu spät. Danach kommen die Wörter wieder gleichmäßig.\n\n'
                      'Diese Wortfolgen nennen wir UNREGELMÄSSIG.\n\n'
                      'Drücken Sie eine beliebige Taste, um ein Beispiel für eine unregelmäßige Wortfolge zu hören. '
                      'In diesem Beispiel ist das Stolpern extra groß, damit man es leicht erkennt.'),


        'check': ('Haben Sie den Unterschied gehört?\n\n'
                  'Drücken Sie LINKS, um die Beispiele noch einmal zu hören.\n\n'
                  'Drücken Sie RECHTS, um fortzufahren.'),


        'task': ('Entscheiden Sie nach jeder Wortfolge, ob sie regelmäßig oder unregelmäßig war:\n\n'
                 '• Unregelmäßig: Der Rhythmus ist ins Stolpern geraten, weil ein Wort zu früh oder zu spät kam.\n'
                 '• Regelmäßig: Alle Wörter kamen in gleichmäßigen Abständen.\n\n'
                 'Im Experiment wird das Stolpern viel kleiner sein als im Beispiel, deshalb werden manche Wortfolgen '
                 'schwer zu beurteilen sein. Das ist normal. Wenn Sie unsicher sind, verlassen Sie sich auf Ihren '
                 'ersten Eindruck.\n\n' + _CONTINUE),


        'how': ('Jeder Durchgang läuft so ab:\n\n'
                '1. Sie sehen die Wörter auf dem Bildschirm.\n'
                '2. Die Wortfolge wird abgespielt.\n'
                '3. Wenn die Wortfolge zu Ende ist, drücken Sie eine Taste:\n'
                '      LINKS = {left}        RECHTS = {right}\n\n'
                'Bitte warten Sie, bis die Wortfolge vollständig zu Ende ist. Tastendrücke während der Wortfolge '
                'werden nicht gezählt. Nach dem Ende der Wortfolge haben Sie {window} Sekunden Zeit zum Antworten.\n\n'
                + _CONTINUE),


        'practice_intro': ('Wir üben jetzt mit {n_practice} Beispielen. Nach jedem Beispiel sehen Sie, '
                           'ob Ihre Antwort richtig war.\n\n'
                           'Drücken Sie eine beliebige Taste, um zu beginnen.'),


        'fb_correct': 'Richtig!\n\nDiese Wortfolge war {answer}.',

        'fb_wrong':   'Leider falsch.\n\nDiese Wortfolge war {answer}.',

        'fb_slow':    'Zu langsam.\n\nBitte antworten Sie innerhalb von {window} Sekunden nach dem Ende der Wortfolge.',

        'practice_result': ('Sie hatten {n_correct} von {n_practice} richtig.\n\n'
                            'Drücken Sie LINKS, um noch einmal zu üben.\n\n'
                            'Drücken Sie RECHTS, um das Experiment zu starten.'),


        'practice_repeat': ('Sie hatten {n_correct} von {n_practice} richtig.\n\n'          # NEW (not in your file)
                            'Wir üben noch einmal.\n\n'
                            'Drücken Sie eine beliebige Taste, um zu beginnen.'),


        'start': ('Das Experiment besteht aus {n_blocks} Blöcken von jeweils etwa 5 Minuten. Zwischen den Blöcken '
                  'können Sie so lange Pause machen, wie Sie möchten.\n\n'
                  'Ab jetzt sehen Sie nicht mehr nach jeder Antwort, ob sie richtig war. Am Ende jedes Blocks sehen '
                  'Sie, wie viele Antworten richtig waren.\n\n'
                  'Bereit? Drücken Sie eine beliebige Taste, um zu beginnen.'),


        'keys_reminder': 'LINKS: {left}          RECHTS: {right}',

        'break_high': ('Block {block} von {n_blocks} geschafft!\n\n'
                       'Sie haben {score} % richtig beantwortet. Das machen Sie großartig!\n\n' + _BREAK_END),

        'break_mid': ('Block {block} von {n_blocks} geschafft!\n\n'
                      'Sie haben {score} % richtig beantwortet. Das machen Sie gut!\n\n' + _BREAK_END),

        'break_low': ('Block {block} von {n_blocks} geschafft!\n\n'
                      'Sie haben {score} % richtig beantwortet. Die Unterschiede sind sehr klein, deshalb ist die '
                      'Aufgabe schwierig. Versuchen Sie, im nächsten Block genau auf den Rhythmus zu hören.\n\n'
                      + _BREAK_END),


        'end': ('Geschafft! Im letzten Block haben Sie {score} % richtig beantwortet.\n\n'
                'Vielen Dank! Bitte geben Sie der Versuchsleitung Bescheid, dass Sie fertig sind.'),
    },
}

# acoustic group: same screens, nothing that suggests speech
TEXTS['acoustic'] = dict(TEXTS['speech'])
TEXTS['acoustic'].update({
    'welcome': ('Willkommen zum letzten Teil des Experiments!\n\n'
                'Sie hören dieselben computergenerierten Klänge wie vorhin: Folgen von kurzen Klängen, '
                'die nacheinander abgespielt werden.\n\n'
                'Diesmal sollen Sie nicht mittippen. Stattdessen ist Ihre Aufgabe, genau auf das Timing der Klänge '
                'zu achten.\n\n' + _CONTINUE),


    'regular': ('In manchen Klangfolgen kommen die Klänge in vollkommen gleichmäßigen Abständen, '
                'wie das Ticken einer Uhr.\n\n'
                'Diese Klangfolgen nennen wir REGELMÄSSIG.\n\n'
                'Drücken Sie eine beliebige Taste, um ein Beispiel für eine regelmäßige Klangfolge zu hören.'),


    'irregular': ('In anderen Klangfolgen gerät der Rhythmus einmal ins Stolpern: An einer Stelle kommt ein Klang '
                  'etwas zu früh oder etwas zu spät. Danach kommen die Klänge wieder gleichmäßig.\n\n'
                  'Diese Klangfolgen nennen wir UNREGELMÄSSIG.\n\n'
                  'Drücken Sie eine beliebige Taste, um ein Beispiel für eine unregelmäßige Klangfolge zu hören. '
                  'In diesem Beispiel ist das Stolpern extra groß, damit man es leicht erkennt.'),


    'task': ('Entscheiden Sie nach jeder Klangfolge, ob sie regelmäßig oder unregelmäßig war:\n\n'
             '• Unregelmäßig: Der Rhythmus ist ins Stolpern geraten, weil ein Klang zu früh oder zu spät kam.\n'
             '• Regelmäßig: Alle Klänge kamen in gleichmäßigen Abständen.\n\n'
             'Im Experiment wird das Stolpern viel kleiner sein als im Beispiel, deshalb werden manche Klangfolgen '
             'schwer zu beurteilen sein. Das ist normal. Wenn Sie unsicher sind, verlassen Sie sich auf Ihren '
             'ersten Eindruck.\n\n' + _CONTINUE),

             
    'how': ('Jeder Durchgang läuft so ab:\n\n'
            '1. Sie sehen die Nummer des Durchgangs auf dem Bildschirm.\n'
            '2. Die Klangfolge wird abgespielt.\n'
            '3. Wenn die Klangfolge zu Ende ist, drücken Sie eine Taste:\n'
            '      LINKS = {left}        RECHTS = {right}\n\n'
            'Bitte warten Sie, bis die Klangfolge vollständig zu Ende ist. Tastendrücke während der Klangfolge '
            'werden nicht gezählt. Nach dem Ende der Klangfolge haben Sie {window} Sekunden Zeit zum Antworten.\n\n'
            + _CONTINUE),
    'fb_correct': 'Richtig!\n\nDiese Klangfolge war {answer}.',
    'fb_wrong':   'Leider falsch.\n\nDiese Klangfolge war {answer}.',
    'fb_slow':    'Zu langsam.\n\nBitte antworten Sie innerhalb von {window} Sekunden nach dem Ende der Klangfolge.',
})