"""
Switch experiment - DETECTION: the settings that are NEW compared with detection/detection_params.py.
Everything else (paths, audio, timing, keys, screen, training, base texts) comes from detection_params.py /
detection_params_laptop.py, chosen by SETUP in the run script. Nothing in detection/ is changed.

Phases (blocks of the detection design table; every block is balanced on its own, so the table is used as it is):
    phase 1 = block 1      : after tapping phase 1; training + 1 block, acoustic for everyone
    phase 2 = blocks 2-3   : after tapping phase 2; no training; mode 'switch' -> speech, 'control' -> acoustic
The mode is read from the participant's TAPPING design (made by switch_design_table.py), so it cannot be typed wrong.
The key mapping (counterbalance) is chosen in phase 1 and read back from the phase-1 output in phase 2.
"""

#=====================
# PHASES AND CONDITIONS
#=====================
PHASE_BLOCKS = {1: [1], 2: [2, 3]}
STIM_GROUP   = 'acoustic'                    # detection tables/stimuli are always made as 'acoustic' (both modes)
CONDITION    = {'switch':  {1: 'acoustic', 2: 'speech'},
                'control': {1: 'acoustic', 2: 'acoustic'}}
PRE_SOUND_SCREEN = {'acoustic': 'trial_number', 'speech': 'sentence'}   # pre-sound screen per condition

# where the mode is read from (relative to params.MAIN_DIR; the first file that exists is used)
MODE_SOURCES = [('tapping', 'stimuli_tapping', '{pid}_acoustic', 'design_stim.csv'),
                ('tapping', 'design_tables', 'tapping_{pid}_acoustic.csv')]

#=====================
# OUTPUT (in params.DATA_DIR, never overwritten)
#=====================
OUTPUT_FILE_PATTERN     = 'switch_detection_{pid}_{mode}_phase{phase}_{time}.tsv'
TRAINING_OUTPUT_PATTERN = 'switch_detection_training_{pid}_{mode}_phase{phase}_{time}.tsv'
PHASE1_GLOB             = 'switch_detection*_{pid}_{mode}_phase1_*.tsv'   # phase 2 reads the counterbalance here
SNAPSHOT_TAG            = '{mode}_phase{phase}'   # stimulus snapshot: stiminfo_<pid>_<tag>_<time>.csv ...
SWITCH_COLUMNS = ['mode', 'phase', 'phase_condition', 'block_in_phase']

RESPONSE_EST_S = 1.0                         # assumed answer time, only for the 'minutes per block' estimate

#=====================
# TEXTS (German) - changes to detection_params.TEXTS[condition]
#=====================
# detection is no longer the last part of the session: (old, new) text replacements in these screens
TEXT_REPLACE = {'welcome': ('Willkommen zum letzten Teil des Experiments!', 'Willkommen zum nächsten Teil des Experiments!')}

# the 'start' screen per phase and condition
# placeholders as in detection_params: {left} {right}, {window}, {n_blocks} (= blocks in THIS phase),
# {block_min} (minutes per block, rounded up), {block}, {score}
START_TEXT = {
    1: {'acoustic': ('Dieser Teil besteht aus einem Block von etwa {block_min} Minuten.\n\n'
                     'Ab jetzt sehen Sie nicht mehr nach jeder Antwort, ob sie richtig war. Am Ende des Blocks '
                     'sehen Sie, wie viele Antworten richtig waren.\n\n'
                     'Bereit? Drücken Sie eine beliebige Taste, um zu beginnen.')},
    2: {'acoustic': ('Willkommen zurück!\n\n'
                     'Ihre Aufgabe ist dieselbe wie vorhin: Ist der Rhythmus regelmäßig oder unregelmäßig?\n'
                     'LINKS: {left}          RECHTS: {right}\n\n'
                     'Dieser Teil besteht aus {n_blocks} Blöcken von jeweils etwa {block_min} Minuten. Zwischen den '
                     'Blöcken können Sie so lange Pause machen, wie Sie möchten.\n\n'
                     'Bereit? Drücken Sie eine beliebige Taste, um zu beginnen.'),
        'speech':   ('Willkommen zurück!\n\n'
                     'Ihre Aufgabe ist dieselbe wie vorhin: Ist der Rhythmus regelmäßig oder unregelmäßig?\n'
                     'LINKS: {left}          RECHTS: {right}\n\n'
                     'Neu ist: Vor jeder Klangfolge sehen Sie jetzt den Satz auf dem Bildschirm. Lesen Sie ihn.\n\n'
                     'Dieser Teil besteht aus {n_blocks} Blöcken von jeweils etwa {block_min} Minuten. Zwischen den '
                     'Blöcken können Sie so lange Pause machen, wie Sie möchten.\n\n'
                     'Bereit? Drücken Sie eine beliebige Taste, um zu beginnen.')},
}