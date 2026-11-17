"""
Detection experiment - SPEECH (informed) group.

Before each sound the sentence is shown on the screen (only the sentence) for PRE_SOUND_S;
when the sound starts, the screen switches to the key reminder.
All trial logic is in detection_functions.run_session (shared with the acoustic group).
"""
import detection_functions as ef

SETUP = 'laptop'          # 'lab' or 'laptop'
GROUP = 'speech'          # do not change: this is the speech-group script

params = ef.load_params(SETUP)
ef.run_session(params, SETUP, GROUP)