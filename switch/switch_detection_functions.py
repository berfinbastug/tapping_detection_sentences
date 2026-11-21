"""Session for the switch experiment (detection): runs ONE phase (block 1, or blocks 2-3).

All trial logic is imported unchanged from detection/detection_functions.py (loading, run_trial, training, keys,
scores, screens). New here: phases, the mode from the tapping design, the condition per phase, the key mapping
carried over from phase 1, and phase/mode columns in the output.
Python 3.6 compatible (lab PC).
"""
import os
import glob

import pandas as pd
from psychtoolbox import GetSecs
from psychopy import core, gui, visual
from psychopy.hardware import keyboard

import detection_functions as ef


def read_mode(params, sp, pid):
    """The participant's mode ('switch' / 'control') from their tapping design (switch_design_table.py)."""
    for parts in sp.MODE_SOURCES:
        path = os.path.join(params.MAIN_DIR, *[p.format(pid=pid) for p in parts])
        if os.path.exists(path):
            d = pd.read_csv(path)
            if 'mode' not in d.columns:
                raise ValueError(f'{path} has no "mode" column: make the tapping design with '
                                 f'switch/switch_design_table.py')
            modes = d['mode'].unique()
            if len(modes) != 1 or modes[0] not in sp.CONDITION:
                raise ValueError(f'{path}: mode must be one of {list(sp.CONDITION)}, found {list(modes)}')
            print(f'mode read from {path}')
            return str(modes[0])
    raise FileNotFoundError(f'no tapping design for {pid} - run switch/switch_design_table.py first '
                            f'(looked in {[os.path.join(*p) for p in sp.MODE_SOURCES]})')


def read_counterbalance(params, sp, pid, mode):
    """Phase 2: the key mapping used in phase 1 (from its main or training output file)."""
    pattern = os.path.join(params.DATA_DIR, sp.PHASE1_GLOB.format(pid=pid, mode=mode))
    values = set()
    for path in sorted(glob.glob(pattern)):
        d = pd.read_csv(path, sep='\t')
        if 'counterbalance' in d.columns:
            values |= set(int(v) for v in d['counterbalance'].dropna().unique())
    if len(values) != 1:
        raise ValueError(f'phase 2 needs the counterbalance of phase 1: found {sorted(values)} in {pattern}')
    return values.pop()


def confirm(msg):
    """Second dialog: the experimenter sees what will run before the window opens."""
    print(msg)
    dlg = gui.Dlg(title='Check before starting')
    dlg.addText(msg)
    dlg.show()
    if not dlg.OK:
        core.quit()


def show_and_wait(text, win, kb, params):
    """As ef.show_and_wait, but old key presses cannot skip the screen."""
    kb.clearEvents()
    ef.show_and_wait(text, win, kb, params)


def run_phase(params, sp, setup, phase):
    """Dialog -> checks -> (phase 1: training) -> start screen -> the blocks of this phase -> end."""
    blocks = sp.PHASE_BLOCKS[phase]
    group = sp.STIM_GROUP

    # ---------- dialog (phase 1: choose the key mapping; phase 2: it comes from phase 1)
    info = {'participant_id': 'S01'}
    if phase == 1:
        info['counterbalance'] = [1, 2]
    if len(blocks) > 1:
        info['start_block'] = [str(b) for b in blocks]          # > first block only to resume after a crash
    dlg = gui.DlgFromDict(dictionary=info, title=f'Switch detection - phase {phase} ({setup})')
    if not dlg.OK:
        core.quit()
    pid = str(info['participant_id']).strip().upper()
    start_block = int(info.get('start_block', blocks[0]))

    # ---------- mode, condition, key mapping
    mode = read_mode(params, sp, pid)
    condition = sp.CONDITION[mode][phase]
    screen_kind = sp.PRE_SOUND_SCREEN[condition]
    counterbalance = int(info['counterbalance']) if phase == 1 else read_counterbalance(params, sp, pid, mode)
    with_training = phase == 1 and start_block == blocks[0]
    confirm(f'{pid}: mode = {mode}, detection phase {phase} = {condition.upper()}, blocks {blocks[0]}-{blocks[-1]}, '
            f'start at block {start_block}, counterbalance {counterbalance}, '
            f'training {"yes" if with_training else "no"} ({setup})')

    # ---------- design, sounds, screen texts
    folder = ef.participant_dir(params, pid, group)
    ef.load_stim_meta(params, folder, pid, group)
    design = ef.load_design(params, folder, pid, group)
    phase_design = design[design['block'].isin(blocks)].reset_index(drop=True)
    print(f'{pid} ({mode}, phase {phase} = {condition}): {len(phase_design)} trials - loading sounds from {folder} ...')
    sounds = ef.load_participant_sounds(params, folder, phase_design)
    sentences = ef.load_sentence_texts(params, folder, phase_design)
    training = ef.load_training(params) if with_training else None

    os.makedirs(params.DATA_DIR, exist_ok=True)
    session_time = ef.get_datetime_string()
    fmt = dict(pid=pid, mode=mode, phase=phase, time=session_time)
    out_path = os.path.join(params.DATA_DIR, sp.OUTPUT_FILE_PATTERN.format(**fmt))
    train_path = os.path.join(params.DATA_DIR, sp.TRAINING_OUTPUT_PATTERN.format(**fmt))
    if os.path.exists(out_path) or os.path.exists(train_path):
        raise FileExistsError(f'{out_path} or {train_path} exists - not overwriting')
    print('Output:', out_path)
    ef.save_stim_snapshot(params, folder, phase_design, sounds, training, pid, sp.SNAPSHOT_TAG.format(**fmt),
                          session_time)
    columns = params.OUTPUT_COLUMNS + sp.SWITCH_COLUMNS

    def log_training(row):
        ef.append_row(train_path, dict(participant=pid, group=group, session_time=session_time,
                                       counterbalance=counterbalance, **row), params.TRAINING_COLUMNS)

    max_sound_s = max(len(y) for y in sounds.values()) / params.FS
    trial_s = params.PRE_SOUND_S + max_sound_s + sp.RESPONSE_EST_S + params.ITI_S
    win = stream = None
    try:
        # ---------- audio, window, keyboard
        stream = ef.open_stream(params, ef.find_device(params))
        win = visual.Window(params.WIN_SIZE, fullscr=params.FULLSCREEN, monitor=params.MONITOR,
                            units=params.UNITS, screen=params.SCREEN)
        win.mouseVisible = False
        kb = keyboard.Keyboard()

        T = dict(params.TEXTS[condition])
        for key, (old, new) in sp.TEXT_REPLACE.items():
            T[key] = T[key].replace(old, new)
        T['start'] = sp.START_TEXT[phase][condition]
        values = dict(ef.key_labels(counterbalance, params), window=f'{params.RESPONSE_WINDOW_S:g}',
                      n_blocks=len(blocks), block_min=int(-(-params.TRIALS_PER_BLOCK * trial_s // 60)),
                      n_practice=0 if training is None else int((training[0]['phase'] == 'practice').sum()))
        keys_reminder = ef.fill(T['keys_reminder'], values)
        pre_height = params.SENTENCE_HEIGHT if screen_kind == 'sentence' else params.TEXT_HEIGHT

        # ---------- training (phase 1 only, as in detection/), then start screen
        if with_training:
            ef.run_training(stream, win, kb, params, condition, counterbalance, T, values, training, log_training)
        show_and_wait(ef.fill(T['start'], values), win, kb, params)

        # ---------- blocks of this phase
        for block in [b for b in blocks if b >= start_block]:
            block_in_phase = blocks.index(block) + 1
            trials = phase_design[phase_design['block'] == block]
            if params.MAX_TRIALS_PER_BLOCK is not None:
                trials = trials.head(params.MAX_TRIALS_PER_BLOCK)
            ef.display_text('', win, params)                                # blank
            next_trial = GetSecs() + params.FIRST_TRIAL_DELAY_S
            results = []

            for i, (_, row) in enumerate(trials.iterrows(), start=1):
                text = ef.pre_sound_text(screen_kind, i, sentences[row['trial']], params)
                r = ef.run_trial(stream, win, kb, params, sounds[row['trial']], text, pre_height, keys_reminder,
                                 next_trial, counterbalance)
                ef.display_text('', win, params)                            # blank ITI
                next_trial = r.pop('t_next_base') + params.ITI_S

                out = {c: row[c] for c in params.DESIGN_COLUMNS}
                out.update(session_time=session_time, counterbalance=counterbalance, params_file=params.__name__,
                           pre_sound_screen=screen_kind, screen_text=text,
                           correct=int(r['response'] == row['correct_response']) if r['response'] != 'none' else None,
                           **r)
                out.update(mode=mode, phase=phase, phase_condition=condition, block_in_phase=block_in_phase)
                ef.append_row(out_path, out, columns)
                results.append(dict(response=r['response'], correct_response=row['correct_response']))
                rt = r['rt_from_offset_s']
                print(f"block {block} ({block_in_phase}/{len(blocks)}) trial {i}/{len(trials)}  {row['stim_id']:16s} "
                      f"{text!r:40s} -> {r['response']:9s} {'' if rt is None else f'RT {rt:.3f} s'}")

            score = ef.score_percent(results, params)
            if block != blocks[-1]:
                tier = ('break_high' if score >= params.SCORE_HIGH else
                        'break_mid' if score >= params.SCORE_MID else 'break_low')
                show_and_wait(ef.fill(T[tier], values, block=block_in_phase, score=score), win, kb, params)
            else:
                show_and_wait(ef.fill(T['end'], values, score=score), win, kb, params)

    except ef.QuitExperiment:
        print('Quit key pressed - data saved up to the last completed trial:', out_path)

    finally:                       # always close audio and window; an error still shows its full traceback
        if stream is not None:
            stream.close()
        if win is not None:
            win.close()
    core.quit()                    # only reached after a normal end or the quit key