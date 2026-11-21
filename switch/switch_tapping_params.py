"""
Switch experiment - TAPPING: the settings that are NEW compared with tapping/tapping_params.py.
Everything else (paths, audio, recording, tap check, timing, screen, base texts) comes from tapping_params.py /
tapping_params_laptop.py, chosen by SETUP in the run script. Nothing in tapping/ is changed.

Phases (blocks of the design table made by switch_design_table.py):
    phase 1 = blocks 1-3 : acoustic for everyone ('Durchgang n', acoustic texts, focus question)
    phase 2 = blocks 4-9 : mode 'switch'  -> speech   (number + sentence, speech texts, intelligibility question)
                           mode 'control' -> acoustic (as in phase 1)
The mode is read from the participant's design table (column 'mode'), so it cannot be typed wrong.

Per trial: trial screen -> sound + recording (exactly as in tapping/) -> wav saved -> RATING SCREEN (keys 1-4,
no time limit) -> blank AFTER_RATING_S -> next trial.
"""

#=====================
# PHASES AND CONDITIONS
#=====================
PHASE_BLOCKS = {1: [1, 2, 3], 2: [4, 5, 6, 7, 8, 9]}
STIM_GROUP   = 'acoustic'                    # stimuli/design are always built as 'acoustic' (both modes)
CONDITION    = {'switch':  {1: 'acoustic', 2: 'speech'},
                'control': {1: 'acoustic', 2: 'acoustic'}}
PRE_SOUND_SCREEN = {'acoustic': 'trial_number', 'speech': 'sentence'}   # trial screen per condition

#=====================
# OUTPUT (in params.DATA_DIR, never overwritten)
#=====================
OUTPUT_FILE_PATTERN = 'switch_tapping_{pid}_{mode}_phase{phase}_{time}.tsv'
REC_DIR_PATTERN     = 'switch_tapping_rec_{pid}_{mode}_phase{phase}_{time}'
SNAPSHOT_TAG        = '{mode}_phase{phase}'  # stimulus snapshot files: tapping_stiminfo_<pid>_<tag>_<time>.csv ...
SWITCH_COLUMNS = ['mode', 'phase', 'phase_condition', 'block_in_phase',
                  'rating_question', 'rating', 'rating_rt', 't_rating_on', 't_rating_off']

#=====================
# RATING AFTER EVERY TRIAL
#=====================
RATING_KEYS    = ['1', '2', '3', '4']        # number row; no time limit
AFTER_RATING_S = 0.5                         # blank screen between the answer and the next trial screen
RATING_EST_S   = 2.0                         # assumed answer time, only for the 'minutes per block' estimate

RATING_QUESTION = {   # short name stored in the data file
    'acoustic': 'focus',
    'speech':   'intelligibility',
}
_SCALE = '1 = {low}          2          3          4 = {high}'
RATING_SCREEN = {
    'acoustic': ('Wie konzentriert waren Sie während dieser Klangfolge auf die Klänge?\n\n'
                 + _SCALE.format(low='gar nicht', high='voll und ganz')
                 + '\n\nDrücken Sie eine Taste von 1 bis 4.'),
    'speech':   ('Wie gut konnten Sie die Wörter verstehen?\n\n'
                 + _SCALE.format(low='gar nicht', high='sehr gut')
                 + '\n\nDrücken Sie eine Taste von 1 bis 4.'),
}

#=====================
# TEXTS (German) - added to / replacing the texts of tapping_params.TEXTS[condition]
#=====================
# placeholders as in tapping_params: {n_blocks} (= blocks in THIS phase), {n_trials}, {block_min}, {block}
# (= block within this phase), {n_good}, {n_played}, {feedback}
_CONTINUE = 'Drücken Sie eine beliebige Taste, um fortzufahren.'
_QUESTION_INTRO = ('Nach jeder Klangfolge erscheint eine Frage:\n\n{question}\n\n'
                   'Antworten Sie mit den Tasten 1 bis 4 ({scale}). Es gibt keine richtigen oder falschen '
                   'Antworten. Nach Ihrer Antwort beginnt kurz darauf die nächste Klangfolge.\n\n' + _CONTINUE)
_BLOCKS = ('Dieser Teil besteht aus {n_blocks} Blöcken mit jeweils {n_trials} Klangfolgen.\n\n'
           'Nach jeder Klangfolge beantworten Sie eine kurze Frage. Danach beginnt die nächste Klangfolge '
           'automatisch.\n\n' + _CONTINUE)
_P2_WELCOME = ('Willkommen zurück!\n\nJetzt geht es mit dem Tippen weiter.\n\n' + _CONTINUE)

EXTRA_TEXTS = {
    'acoustic': {
        'main_blocks':    _BLOCKS,
        'question_intro': _QUESTION_INTRO.format(
            question='Wie konzentriert waren Sie während dieser Klangfolge auf die Klänge?',
            scale='1 = gar nicht, 4 = voll und ganz'),
        'p2_welcome':     _P2_WELCOME,
    },
    'speech': {
        'main_blocks':    _BLOCKS,
        'question_intro': _QUESTION_INTRO.format(
            question='Wie gut konnten Sie die Wörter verstehen?',
            scale='1 = gar nicht, 4 = sehr gut'),
        'p2_welcome':     _P2_WELCOME,
    },
}
# screens before 'start' (only when the phase starts at its first block; a resumed phase shows only 'start')
# main_welcome / main_task / main_breaks are the texts of tapping_params.TEXTS[condition]
INTRO_SCREENS = {1: ['main_welcome', 'main_task', 'main_blocks', 'question_intro', 'main_breaks'],
                 2: ['p2_welcome', 'main_task', 'main_blocks', 'question_intro']}