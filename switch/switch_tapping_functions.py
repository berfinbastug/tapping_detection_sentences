"""Session for the switch experiment (tapping): runs ONE phase (blocks 1-3 or 4-9).

All trial logic is imported unchanged from tapping/tapping_functions.py (loading, run_trial, recording, online tap
check, break feedback, screens). New here: phases, condition per phase from the design table's 'mode' column, the
rating screen after every trial, and phase/mode columns in the output.
Python 3.6 compatible (lab PC).
"""
import os

from psychtoolbox import GetSecs
from psychopy import core, gui, visual
from psychopy.hardware import keyboard

import tapping_functions as tf


def read_mode(design, sp):
    """The participant's mode ('switch' / 'control') from the design table made by switch_design_table.py."""
    if 'mode' not in design.columns:
        raise ValueError('the design table has no "mode" column: make it with switch/switch_design_table.py, '
                         'then rebuild the stimuli (build_tapping_stimuli.py, group acoustic)')
    modes = design['mode'].unique()
    if len(modes) != 1 or modes[0] not in sp.CONDITION:
        raise ValueError(f'design table: mode must be one of {list(sp.CONDITION)}, found {list(modes)}')
    return str(modes[0])


def texts(params, sp, condition):
    """tapping_params texts for this condition + the switch texts (which replace / add screens)."""
    return {**params.TEXTS[condition], **sp.EXTRA_TEXTS[condition]}


def confirm(pid, mode, phase, condition, blocks, start_block, setup):
    """Second dialog: the experimenter sees what will run before the window opens."""
    msg = (f'{pid}: mode = {mode}, phase {phase} = {condition.upper()}, blocks {blocks[0]}-{blocks[-1]}, '
           f'start at block {start_block} ({setup})')
    print(msg)
    dlg = gui.Dlg(title='Check before starting')
    dlg.addText(msg)
    dlg.show()
    if not dlg.OK:
        core.quit()


def ask_rating(win, kb, params, sp, condition):
    """Rating screen after a trial: waits (no time limit) for one of RATING_KEYS; the quit key stops."""
    kb.clearEvents()
    t_on = tf.display_text(sp.RATING_SCREEN[condition], win, params)
    kb.clock.reset()
    keys = kb.waitKeys(keyList=sp.RATING_KEYS + [params.QUIT_KEY], waitRelease=False)
    t_off = tf.display_text('', win, params)
    if keys[0].name == params.QUIT_KEY:
        raise tf.QuitExperiment
    return dict(rating_question=sp.RATING_QUESTION[condition], rating=int(keys[0].name),
                rating_rt=round(float(keys[0].rt), 4), t_rating_on=t_on, t_rating_off=t_off)


def show_and_wait(text, win, kb, params):
    """As tf.show_and_wait, but old key presses (e.g. a second rating key) cannot skip the screen."""
    kb.clearEvents()
    tf.show_and_wait(text, win, kb, params)


def run_phase(params, sp, setup, phase):
    """Dialog -> checks -> intro screens -> the blocks of this phase (trial + rating) with break screens -> end."""
    tf.check_params(params)
    blocks = sp.PHASE_BLOCKS[phase]
    group = sp.STIM_GROUP

    # ---------- dialog
    info = {'participant_id': 'S01', 'start_block': [str(b) for b in blocks]}   # start_block > first: resume
    dlg = gui.DlgFromDict(dictionary=info, title=f'Switch tapping - phase {phase} ({setup})')
    if not dlg.OK:
        core.quit()
    pid = str(info['participant_id']).strip().upper()          # same rule as the stimulus builder
    start_block = int(info['start_block'])

    # ---------- design, mode, sounds
    folder = tf.participant_dir(params, pid, group)
    tf.load_stim_meta(params, folder, pid, group)
    design = tf.load_design(params, folder, pid, group)
    onsets = tf.load_events(params, folder, design)
    mode = read_mode(design, sp)
    condition = sp.CONDITION[mode][phase]
    screen_kind = sp.PRE_SOUND_SCREEN[condition]
    confirm(pid, mode, phase, condition, blocks, start_block, setup)

    phase_design = design[design['block'].isin(blocks)].reset_index(drop=True)
    print(f'{pid} ({mode}, phase {phase} = {condition}): {len(phase_design)} trials - loading sounds from {folder} ...')
    sounds = tf.load_participant_sounds(params, folder, phase_design)
    max_sound_s = max(len(y) for y in sounds.values()) / params.FS

    # ---------- output files (new for every session)
    os.makedirs(params.DATA_DIR, exist_ok=True)
    session_time = tf.get_datetime_string()
    fmt = dict(pid=pid, mode=mode, phase=phase, time=session_time)
    out_path = os.path.join(params.DATA_DIR, sp.OUTPUT_FILE_PATTERN.format(**fmt))
    rec_dir = os.path.join(params.DATA_DIR, sp.REC_DIR_PATTERN.format(**fmt))
    if os.path.exists(out_path) or os.path.exists(rec_dir):
        raise FileExistsError(f'{out_path} or {rec_dir} exists - not overwriting')
    os.makedirs(rec_dir)
    print('Output:', out_path, '| recordings:', rec_dir)
    tf.save_stim_snapshot(params, folder, phase_design, sounds, pid, sp.SNAPSHOT_TAG.format(**fmt), setup,
                          session_time)
    columns = params.OUTPUT_COLUMNS + sp.SWITCH_COLUMNS

    trial_s = (max_sound_s + params.REC_POST_S + params.SAVE_MARGIN_S + params.ITI_S + params.PRE_SOUND_S
               + sp.RATING_EST_S + sp.AFTER_RATING_S)
    n_per_block = params.TRIALS_PER_BLOCK if params.MAX_TRIALS_PER_BLOCK is None else \
        min(params.MAX_TRIALS_PER_BLOCK, params.TRIALS_PER_BLOCK)
    n_run = n_per_block * len([b for b in blocks if b >= start_block])
    print(f'about {trial_s:.1f} s per trial (incl. rating) -> {n_run} trials = {n_run * trial_s / 60:.0f} min '
          f'without breaks')

    aud = win = None
    try:
        # ---------- audio, window, keyboard
        aud = tf.open_audio(params, max_sound_s)
        win = visual.Window(params.WIN_SIZE, fullscr=params.FULLSCREEN, monitor=params.MONITOR,
                            units=params.UNITS, screen=params.SCREEN)
        win.mouseVisible = False
        kb = keyboard.Keyboard()
        T = texts(params, sp, condition)
        values = dict(n_blocks=len(blocks), n_trials=params.TRIALS_PER_BLOCK,
                      block_min=int(-(-params.TRIALS_PER_BLOCK * trial_s // 60)))   # minutes, rounded up

        # ---------- instructions (only when the phase starts at its first block)
        if start_block == blocks[0]:
            for key in sp.INTRO_SCREENS[(phase, condition)]:
                show_and_wait(tf.fill(T[key], values), win, kb, params)
        show_and_wait(tf.fill(T['start'], values), win, kb, params)

        # ---------- blocks of this phase
        for block in [b for b in blocks if b >= start_block]:
            block_in_phase = blocks.index(block) + 1
            trials = phase_design[phase_design['block'] == block]
            if params.MAX_TRIALS_PER_BLOCK is not None:
                trials = trials.head(params.MAX_TRIALS_PER_BLOCK)
            tf.display_text('', win, params)                                 # blank
            t_next = GetSecs() + params.FIRST_TRIAL_DELAY_S
            checks = []                                                      # tap checks of this block

            for i, (_, row) in enumerate(trials.iterrows(), start=1):
                items = tf.trial_screen(screen_kind, i, row, params)
                text = ' | '.join(t for t, _, _ in items)                   # what was on screen, for the data file
                rec_path = os.path.join(rec_dir, params.REC_FILE_PATTERN.format(trial=int(row['trial'])))
                r = tf.run_trial(aud, win, kb, params, sounds[row['trial']], items, t_next,
                                 onsets[row['trial']], rec_path)
                # as in tapping/: earliest next trial = SAVE_MARGIN_S + ITI_S after the recording ended
                t_next = r['t_rec_start'] + r['rec_s_planned'] + params.SAVE_MARGIN_S + params.ITI_S
                late = max(0.0, r.pop('t_done') - t_next)

                # NEW: rating (the wav is already saved); the next trial comes AFTER_RATING_S after the answer
                rating = ask_rating(win, kb, params, sp, condition)
                t_next = max(t_next, rating['t_rating_off'] + sp.AFTER_RATING_S)

                out = {c: row[c] for c in params.DESIGN_COLUMNS}
                r.pop('rec_s_planned')
                out.update(session_time=session_time, setup=setup, params_file=params.__name__, trial_in_block=i,
                           pre_sound_screen=screen_kind, screen_text=text, late_start_next_s=round(late, 4), **r)
                out.update(mode=mode, phase=phase, phase_condition=condition, block_in_phase=block_in_phase,
                           **rating)
                tf.append_row(out_path, out, columns)
                checks.append(r)

                warn = ''
                if r['overflow']:
                    warn += '  ! CAPTURE OVERFLOW'
                if late > 0:
                    warn += f'  ! saving took {late:.2f} s longer than SAVE_MARGIN_S'
                if not r['online_recorded_well']:
                    warn += '  ! missing taps'
                if r['online_flag_double']:
                    warn += '  ! double taps'
                if r['online_flag_timing']:
                    warn += '  ! timing off'
                print(f"block {block} ({block_in_phase}/{len(blocks)}) trial {i}/{len(trials)} (#{row['trial']}) "
                      f"{row['sentence_id']} {row['jitter_pattern']:5s} | sounds tapped "
                      f"{r['online_n_sounds_tapped']}/{params.N_EVENTS}, double {r['online_n_sounds_double']}, "
                      f"ITI median {r['online_iti_median_ms']} ms | {rating['rating_question']} {rating['rating']} "
                      f"(RT {rating['rating_rt']:.2f} s){warn}")

            if block != blocks[-1]:
                show_and_wait(tf.fill(T['break'], values, block=block_in_phase,
                                      **tf.block_feedback(checks, T, params)), win, kb, params)
            else:
                show_and_wait(tf.fill(T['end'], values), win, kb, params)

    except tf.QuitExperiment:
        print('Quit key pressed - data saved up to the last completed trial:', out_path)

    finally:                       # always close audio and window; an error still shows its full traceback
        tf.close_audio(aud)
        if win is not None:
            win.close()
    core.quit()                    # only reached after a normal end or the quit key