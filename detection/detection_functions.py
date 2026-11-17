"""Helper functions and the shared session for the detection experiment (PsychoPy + Psychtoolbox only).

run_detection_speech.py and run_detection_acoustic.py only choose the setup and the group and then
call run_session() - so both groups always run exactly the same code.
"""
import os
import json
import hashlib
import importlib
from datetime import datetime

import numpy as np
import pandas as pd
from psychtoolbox import audio, GetSecs, WaitSecs
from psychopy import core, gui, visual
from psychopy.hardware import keyboard


class QuitExperiment(Exception):
    """Raised when the quit key is pressed."""


#=====================
# SESSION
#=====================
def get_datetime_string():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def load_params(setup):
    """'lab' -> detection_params, 'laptop' -> detection_params_laptop."""
    modules = {'lab': 'detection_params', 'laptop': 'detection_params_laptop'}
    if setup not in modules:
        raise ValueError(f'unknown SETUP {setup!r}: use {list(modules)}')
    return importlib.import_module(modules[setup])


#=====================
# DESIGN AND STIMULI
#=====================
def participant_dir(params, pid, group):
    """stimuli/<pid>_<group>/ (built by build_detection_stimuli.py)."""
    folder = os.path.join(params.STIM_DIR, params.PARTICIPANT_FOLDER.format(pid=pid, group=group))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'no stimulus folder {folder} - check the participant ID, that this is the right '
                                f'run script for the {group} group, and that build_detection_stimuli.py was run')
    return folder


def load_stim_meta(params, folder, pid, group):
    """Read the participant's stimulus meta file and check it matches the params and the participant."""
    meta = json.load(open(os.path.join(folder, params.STIM_META)))
    if meta['fs'] != params.FS:
        raise ValueError(f"stimuli are {meta['fs']} Hz, params.FS is {params.FS}")
    if meta['participant'] != pid or meta['group'] != group:
        raise ValueError(f"{folder}: stimuli were built for {meta['participant']}/{meta['group']}, not {pid}/{group}")
    return meta


def load_design(params, folder, pid, group):
    """The participant's design table (with stim_row / n_samples), checked."""
    path = os.path.join(folder, params.STIM_DESIGN)
    d = pd.read_csv(path)
    missing = set(params.DESIGN_COLUMNS) - set(d.columns)
    if missing:
        raise ValueError(f'{path}: columns missing {sorted(missing)}')
    if (d['participant'] != pid).any() or (d['group'] != group).any():
        raise ValueError(f'{path}: participant/group inside the table do not match {pid}/{group}')
    per_block = d.groupby('block').size()
    if list(per_block.index) != list(range(1, params.N_BLOCKS + 1)) or (per_block != params.TRIALS_PER_BLOCK).any():
        raise ValueError(f'{path}: trials per block {per_block.to_dict()}, expected '
                         f'{params.TRIALS_PER_BLOCK} in each of {params.N_BLOCKS} blocks')
    return d.sort_values('trial').reset_index(drop=True)


def load_participant_sounds(params, folder, design):
    """Copy this participant's sounds from their .npy into memory (mono float32, gain applied). Returns {trial: signal}."""
    stims = np.load(os.path.join(folder, params.STIM_NPY), mmap_mode='r')
    sounds = {}
    for row in design.itertuples():
        y = np.array(stims[row.stim_row, :row.n_samples], dtype=np.float32) * params.STIM_GAIN
        if np.abs(y).max() >= 1:
            raise ValueError(f'trial {row.trial} ({row.stim_id}) clips with STIM_GAIN = {params.STIM_GAIN}')
        sounds[row.trial] = y
    return sounds


def sentence_text(words, sentence_type, params):
    """Words as shown on screen: sentences get a capital first letter (CAPITALIZE_FIRST_WORD), word lists not."""
    if params.CAPITALIZE_FIRST_WORD and sentence_type != 'word list':
        words = words[0].upper() + words[1:]
    return words


PRE_SOUND_KINDS = ('sentence', 'trial_number', 'fixation')


def check_pre_sound_kind(params, group):
    kind = params.PRE_SOUND_SCREEN[group]
    if kind not in PRE_SOUND_KINDS:
        raise ValueError(f'PRE_SOUND_SCREEN[{group!r}] must be one of {PRE_SOUND_KINDS}, not {kind!r}')
    return kind


def load_sentence_texts(params, folder, design):
    """{trial: the words as shown on screen}."""
    words = pd.read_csv(os.path.join(folder, params.STIM_INDEX)).set_index('trial')['words']
    return {row.trial: sentence_text(words[row.trial], row.sentence_type, params) for row in design.itertuples()}


def pre_sound_text(kind, n, sentence, params):
    """Text of the pre-sound screen: the sentence, 'Durchgang n', or the fixation cross."""
    if kind == 'sentence':
        return sentence
    if kind == 'trial_number':
        return params.TRIAL_NUMBER_TEXT.format(n=n)
    return params.FIXATION_TEXT


def load_training(params):
    """Training table (demo + practice, with stim_row) and its sounds {stim_id: mono float32}."""
    t = pd.read_csv(params.TRAINING_DESIGN)
    stims = np.load(os.path.join(params.TRAINING_STIM_DIR, params.TRAINING_NPY), mmap_mode='r')
    sounds = {}
    for row in t.drop_duplicates('stim_id').itertuples():
        y = np.array(stims[row.stim_row, :row.n_samples], dtype=np.float32) * params.STIM_GAIN
        if np.abs(y).max() >= 1:
            raise ValueError(f'training {row.stim_id} clips with STIM_GAIN = {params.STIM_GAIN}')
        sounds[row.stim_id] = y
    return t, sounds


def preview_text(row, group, params):
    """Example preview: the words (speech) or one dot per sound (acoustic), with extra space at the shift
    point of a late shift."""
    tokens = row['words'].split() if group == 'speech' else [params.DOT] * 6
    sep = ' ' if group == 'speech' else '  '
    seps = [sep] * (len(tokens) - 1)
    if isinstance(row['interval'], str) and row['shift_ms'] > 0:
        seps[int(row['interval'][1:]) - 1] = params.DEMO_GAP
    text = tokens[0] + ''.join(s + w for s, w in zip(seps, tokens[1:]))
    return sentence_text(text, row['sentence_type'], params) if group == 'speech' else text


def sound_md5(y):
    """Checksum of a sound exactly as played (float32 samples)."""
    return hashlib.md5(np.ascontiguousarray(y, dtype=np.float32).tobytes()).hexdigest()


def file_info(path):
    """Size and last-modified time of a file (to know later which stimulus build was used)."""
    st = os.stat(path)
    return dict(path=os.path.abspath(path), bytes=st.st_size,
                modified=datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'))


def save_stim_snapshot(params, folder, design, sounds, training, pid, group, session_time):
    """Keep this participant's stimulus information next to the data (new files per session, never overwritten).

    stiminfo_...csv : one row per trial (+ one per training stimulus): stim_id, npy row, words, word onsets /
                      offsets (ms), duration, smallest gap, md5 of the sound as played
    stimmeta_...json: stimulus settings (meta files), which .npy files (size, modified time), gain, setup
    """
    index = pd.read_csv(os.path.join(folder, params.STIM_INDEX)).set_index('trial')
    word_cols = [c for c in index.columns if c.startswith(('on', 'off')) and c.endswith('_ms')]
    rows = []
    for row in design.itertuples():
        ix = index.loc[row.trial]
        rows.append(dict(part='main', trial=row.trial, block=row.block, stim_id=row.stim_id, npy_row=row.stim_row,
                         n_samples=row.n_samples, sound_md5=sound_md5(sounds[row.trial]), words=ix['words'],
                         dur_ms=ix['dur_ms'], min_gap_ms=ix['min_gap_ms'], **{c: ix[c] for c in word_cols}))
    meta = dict(participant=pid, group=group, session_time=session_time, params_file=params.__name__,
                stim_gain=params.STIM_GAIN, fs=params.FS,
                main=dict(meta=json.load(open(os.path.join(folder, params.STIM_META))),
                          npy=file_info(os.path.join(folder, params.STIM_NPY))))
    if training is not None:
        t_table, t_sounds = training
        t_index = pd.read_csv(os.path.join(params.TRAINING_STIM_DIR, params.TRAINING_INDEX)).set_index('stim_id')
        for row in t_table.drop_duplicates('stim_id').itertuples():
            ix = t_index.loc[row.stim_id]
            rows.append(dict(part='training', trial=None, block=None, stim_id=row.stim_id, npy_row=row.stim_row,
                             n_samples=row.n_samples, sound_md5=sound_md5(t_sounds[row.stim_id]), words=ix['words'],
                             dur_ms=ix['dur_ms'], min_gap_ms=ix['min_gap_ms'],
                             **{c: ix[c] for c in word_cols if c in ix}))
        meta['training'] = dict(meta=json.load(open(os.path.join(params.TRAINING_STIM_DIR, params.TRAINING_META))),
                                npy=file_info(os.path.join(params.TRAINING_STIM_DIR, params.TRAINING_NPY)),
                                design=file_info(params.TRAINING_DESIGN))
    fmt = dict(pid=pid, group=group, time=session_time)
    info_path = os.path.join(params.DATA_DIR, params.STIM_INFO_PATTERN.format(**fmt))
    meta_path = os.path.join(params.DATA_DIR, params.STIM_META_PATTERN.format(**fmt))
    for path in (info_path, meta_path):
        if os.path.exists(path):
            raise FileExistsError(f'{path} exists - not overwriting')
    pd.DataFrame(rows).to_csv(info_path, index=False)
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2, default=str)
    print('Stimulus snapshot:', info_path)


def to_channels(y, n_channels):
    """Mono -> (n_samples, n_channels) float64 for PsychPortAudio."""
    return np.tile(np.asarray(y, dtype=np.float64)[:, None], (1, n_channels))


#=====================
# AUDIO
#=====================
def find_device(params):
    """DeviceIndex of the one output device whose name contains params.DEVICE_NAME."""
    devices = [d for d in audio.get_devices()
               if params.DEVICE_NAME in d['DeviceName'] and d['NrOutputChannels'] >= params.N_OUT_CHANNELS]
    if len(devices) != 1:
        names = [d['DeviceName'] for d in audio.get_devices()]
        raise RuntimeError(f'{len(devices)} output devices match {params.DEVICE_NAME!r}. Devices: {names}')
    print('Audio device:', devices[0]['DeviceName'])
    return devices[0]['DeviceIndex']


def open_stream(params, device_id):
    """Open the stream and play 0.1 s of silence once (wakes the device up)."""
    stream = audio.Stream(freq=params.FS, channels=params.N_OUT_CHANNELS, device_id=device_id,
                          latency_class=[params.LATENCY_CLASS])
    stream.fill_buffer(np.zeros((int(params.FS * 0.1), params.N_OUT_CHANNELS)))
    stream.start(wait_for_start=1)
    stream.stop(wait_for_end_playback=1)
    return stream


#=====================
# SCREEN AND KEYS
#=====================
def display_text(text, win, params, height=None):
    """Show only this text (empty string = blank screen). Returns GetSecs() right after the flip."""
    if text:
        visual.TextStim(win, text=text, height=height or params.TEXT_HEIGHT, wrapWidth=params.WRAP_WIDTH).draw()
    win.flip()
    return GetSecs()


def wait_continue(kb, params):
    """Wait for a continue key; the quit key raises QuitExperiment."""
    keys = kb.waitKeys(keyList=params.CONTINUE_KEYS + [params.QUIT_KEY], waitRelease=True)
    if keys and keys[0].name == params.QUIT_KEY:
        raise QuitExperiment


def show_and_wait(text, win, kb, params):
    print(text)
    display_text(text, win, params)
    wait_continue(kb, params)


def wait_choice(kb, params):
    """LEFT ('2') or RIGHT ('3') on a choice screen -> 'left' / 'right'; the quit key raises QuitExperiment."""
    keys = kb.waitKeys(keyList=params.RESPONSE_KEYS + [params.QUIT_KEY], waitRelease=True)
    if keys[0].name == params.QUIT_KEY:
        raise QuitExperiment
    return 'left' if keys[0].name == params.RESPONSE_KEYS[0] else 'right'


def show_and_choose(text, win, kb, params):
    print(text)
    display_text(text, win, params)
    return wait_choice(kb, params)


def fill(text, values, **extra):
    """Fill the {placeholders} of a text."""
    return text.format(**{**values, **extra})


def key_labels(counterbalance, params):
    """{'left': ..., 'right': ...} for the texts, from KEY_MAP and RESPONSE_LABELS."""
    m = params.KEY_MAP[counterbalance]
    return dict(left=params.RESPONSE_LABELS[m['2']], right=params.RESPONSE_LABELS[m['3']])


def key_to_response(key_name, counterbalance, params):
    """'regular' / 'irregular' for a response key, 'none' for no key."""
    if key_name is None:
        return 'none'
    return params.KEY_MAP[counterbalance][key_name]


#=====================
# OUTPUT
#=====================
def append_row(path, row, columns):
    """Append one trial to the tab-separated output file (header written with the first row)."""
    new = not os.path.exists(path)
    pd.DataFrame([row]).reindex(columns=columns).to_csv(path, sep='\t', mode='a', header=new, index=False)


#=====================
# ONE TRIAL (main and practice)
#=====================
def run_trial(stream, win, kb, params, y, pre_text, pre_height, keys_reminder, next_trial, counterbalance):
    """pre-sound screen at next_trial -> sound PRE_SOUND_S later + key reminder -> response window.

    Does NOT change the screen after the response (the caller shows blank or feedback).
    Returns a dict with response, timing and 't_next_base' (key press, or end of window if no answer).
    """
    stream.fill_buffer(to_channels(y, params.N_OUT_CHANNELS))

    WaitSecs(max(0.0, next_trial - GetSecs()))
    t_pre = display_text(pre_text, win, params, height=pre_height)
    t_scheduled = t_pre + params.PRE_SOUND_S
    t_onset = stream.start(when=t_scheduled, wait_for_start=1)
    t_keys = display_text(keys_reminder, win, params)
    sound_dur = len(y) / params.FS
    t_offset = t_onset + sound_dur

    t_listen = t_onset if params.RESPOND_DURING_SOUND else t_offset
    WaitSecs(max(0.0, t_listen - GetSecs()))
    kb.clearEvents()                                  # presses before t_listen are discarded
    kb.clock.reset()
    t_reset = GetSecs()
    window_end = t_offset + params.RESPONSE_WINDOW_S
    keys = kb.waitKeys(maxWait=max(0.0, window_end - t_reset),
                       keyList=params.RESPONSE_KEYS + [params.QUIT_KEY], waitRelease=True)
    if keys and keys[0].name == params.QUIT_KEY:
        raise QuitExperiment
    stream.stop(wait_for_end_playback=1)              # sound is over; never cuts it

    if keys:
        key = keys[0]
        t_press = t_reset + key.rt
        key_name, tDown, press_dur = key.name, key.tDown, key.duration
    else:
        t_press, key_name, tDown, press_dur = None, None, None, None
    return dict(key_name=key_name if key_name else 'no_response',
                response=key_to_response(key_name, counterbalance, params),
                rt_from_offset_s=None if t_press is None else t_press - t_offset,
                rt_from_onset_s=None if t_press is None else t_press - t_onset,
                key_tDown=tDown, button_press_duration=press_dur,
                t_pre_sound_on=t_pre, t_onset_scheduled=t_scheduled, t_onset=t_onset, t_keys_on=t_keys,
                t_offset=t_offset, sound_dur_s=sound_dur,
                t_next_base=t_press if t_press is not None else window_end)


def score_percent(results, params):
    """Block score in % (integer): 'balanced' or 'plain' (SCORE_METHOD). No answer = wrong."""
    d = pd.DataFrame(results)
    hit = d['response'] == d['correct_response']
    if params.SCORE_METHOD == 'balanced' and d['correct_response'].nunique() == 2:
        return int(round(100 * hit.groupby(d['correct_response']).mean().mean()))
    return int(round(100 * hit.mean()))


#=====================
# TRAINING: examples and practice
#=====================
def play_example(stream, win, params, row, y, group):
    """Preview (words / dots) -> '+' -> sound -> short pause. Returns its timing."""
    stream.fill_buffer(to_channels(y, params.N_OUT_CHANNELS))
    text = preview_text(row, group, params)
    height = params.SENTENCE_HEIGHT if group == 'speech' else params.TEXT_HEIGHT
    t_pre = display_text(text, win, params, height=height)
    WaitSecs(max(0.0, t_pre + params.DEMO_PREVIEW_S - GetSecs()))
    t_fix = display_text(params.FIXATION_TEXT or '+', win, params)
    t_onset = stream.start(when=t_fix + params.DEMO_FIX_TO_SOUND_S, wait_for_start=1)
    stream.stop(wait_for_end_playback=1)
    t_offset = t_onset + len(y) / params.FS
    WaitSecs(max(0.0, t_offset + params.DEMO_AFTER_SOUND_S - GetSecs()))
    return dict(screen_text=text, t_pre_sound_on=t_pre, t_onset=t_onset, t_offset=t_offset)


def run_training(stream, win, kb, params, group, counterbalance, T, values, training, log):
    """Screens 1-10: explanations, examples (repeatable), task, how to answer, practice (repeated as needed)."""
    table, sounds = training
    demo = table[table['phase'] == 'demo'].set_index('role')
    practice = table[table['phase'] == 'practice'].reset_index(drop=True)
    rng = np.random.default_rng()

    def example(role, rnd):
        row = demo.loc[role]
        out = play_example(stream, win, params, row, sounds[row['stim_id']], group)
        log(dict(phase='demo', round=rnd, trial_in_round=1 if role == 'demo_regular' else 2, item_id=row['item_id'],
                 sentence_type=row['sentence_type'], role=role, interval=row['interval'], shift_ms=row['shift_ms'],
                 stim_id=row['stim_id'], correct_response=row['correct_response'], **out))

    # explanations + examples
    show_and_wait(T['welcome'], win, kb, params)
    show_and_wait(T['regular'], win, kb, params)
    example('demo_regular', 1)
    show_and_wait(T['irregular'], win, kb, params)
    example('demo_irregular', 1)
    rnd = 1
    while show_and_choose(T['check'], win, kb, params) == 'left':      # LEFT = hear the examples again
        rnd += 1
        example('demo_regular', rnd)
        example('demo_irregular', rnd)

    show_and_wait(T['task'], win, kb, params)
    show_and_wait(fill(T['how'], values), win, kb, params)

    # practice
    keys_reminder = fill(T['keys_reminder'], values)
    kind = check_pre_sound_kind(params, group)
    pre_height = params.SENTENCE_HEIGHT if kind == 'sentence' else params.TEXT_HEIGHT
    show_and_wait(fill(T['practice_intro'], values), win, kb, params)
    rnd = 0
    while True:
        rnd += 1
        order = practice
        if rnd > 1 and params.RESHUFFLE_REPEATS:
            for _ in range(1000):                                       # new order, no item twice in a row
                order = practice.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
                if (order['item_id'].values[1:] != order['item_id'].values[:-1]).all():
                    break
        display_text('', win, params)
        next_trial = GetSecs() + params.FIRST_TRIAL_DELAY_S
        n_correct = 0
        for i, row in order.iterrows():
            pre_text = pre_sound_text(kind, i + 1, sentence_text(row['words'], row['sentence_type'], params), params)
            r = run_trial(stream, win, kb, params, sounds[row['stim_id']], pre_text, pre_height, keys_reminder,
                          next_trial, counterbalance)
            correct = r['response'] == row['correct_response']
            n_correct += int(correct)
            answer = params.RESPONSE_LABELS[row['correct_response']]
            fb = (T['fb_slow'] if r['response'] == 'none' else T['fb_correct'] if correct else T['fb_wrong'])
            display_text(fill(fb, values, answer=answer), win, params)
            WaitSecs(params.FEEDBACK_S)
            display_text('', win, params)
            next_trial = GetSecs() + params.ITI_S
            log(dict(phase='practice', round=rnd, trial_in_round=i + 1, item_id=row['item_id'],
                     sentence_type=row['sentence_type'], role=row['role'], interval=row['interval'],
                     shift_ms=row['shift_ms'], stim_id=row['stim_id'], correct_response=row['correct_response'],
                     screen_text=pre_text, correct=int(correct), **r))
            print(f"practice round {rnd} trial {i + 1}: {row['stim_id']:18s} -> {r['response']:9s} "
                  f"{'correct' if correct else 'wrong'}")
        result = dict(values, n_correct=n_correct)
        if n_correct < params.PRACTICE_PASS and rnd < params.MAX_AUTO_ROUNDS:
            show_and_wait(fill(T['practice_repeat'], result), win, kb, params)       # automatic repeat
            continue
        if show_and_choose(fill(T['practice_result'], result), win, kb, params) == 'right':
            break                                                                    # RIGHT = start


#=====================
# THE SESSION (shared by both groups)
#=====================
def run_session(params, setup, group):
    """Dialog -> startup -> training (examples + practice) -> start screen -> blocks with score breaks -> end.
    Every trial is saved immediately (main trials and training in separate files).
    """
    # ---------- dialog
    info = dict(params.DIALOG_FIELDS)
    dlg = gui.DlgFromDict(dictionary=info, title=f'Detection - {group} group ({setup})')
    if not dlg.OK:
        core.quit()
    pid = str(info['participant_id']).strip()
    counterbalance, start_block = int(info['counterbalance']), int(info['start_block'])

    # ---------- design, sounds, screen texts, output files
    folder = participant_dir(params, pid, group)
    load_stim_meta(params, folder, pid, group)
    design = load_design(params, folder, pid, group)
    print(f'{pid} ({group}): {len(design)} trials - loading sounds from {folder} ...')
    sounds = load_participant_sounds(params, folder, design)
    screen_kind = check_pre_sound_kind(params, group)
    sentences = load_sentence_texts(params, folder, design)
    training = load_training(params) if start_block == 1 else None

    os.makedirs(params.DATA_DIR, exist_ok=True)
    session_time = get_datetime_string()
    out_path = os.path.join(params.DATA_DIR,
                            params.OUTPUT_FILE_PATTERN.format(pid=pid, group=group, time=session_time))
    train_path = os.path.join(params.DATA_DIR,
                              params.TRAINING_OUTPUT_PATTERN.format(pid=pid, group=group, time=session_time))
    print('Output:', out_path)
    save_stim_snapshot(params, folder, design, sounds, training, pid, group, session_time)

    def log_training(row):
        append_row(train_path, dict(participant=pid, group=group, session_time=session_time,
                                    counterbalance=counterbalance, **row), params.TRAINING_COLUMNS)

    # ---------- audio, window, keyboard
    stream = open_stream(params, find_device(params))
    win = visual.Window(params.WIN_SIZE, fullscr=params.FULLSCREEN, monitor=params.MONITOR,
                        units=params.UNITS, screen=params.SCREEN)
    win.mouseVisible = False
    kb = keyboard.Keyboard()

    T = params.TEXTS[group]
    values = dict(key_labels(counterbalance, params), window=f'{params.RESPONSE_WINDOW_S:g}',
                  n_blocks=params.N_BLOCKS,
                  n_practice=0 if training is None else int((training[0]['phase'] == 'practice').sum()))
    keys_reminder = fill(T['keys_reminder'], values)
    pre_height = params.SENTENCE_HEIGHT if screen_kind == 'sentence' else params.TEXT_HEIGHT

    try:
        # ---------- training (skipped when resuming at block 2 or 3)
        if start_block == 1:
            run_training(stream, win, kb, params, group, counterbalance, T, values, training, log_training)
        show_and_wait(fill(T['start'], values), win, kb, params)

        # ---------- blocks
        for block in range(start_block, params.N_BLOCKS + 1):
            trials = design[design['block'] == block]
            if params.MAX_TRIALS_PER_BLOCK is not None:
                trials = trials.head(params.MAX_TRIALS_PER_BLOCK)
            display_text('', win, params)                                   # blank
            next_trial = GetSecs() + params.FIRST_TRIAL_DELAY_S
            results = []

            for i, (_, row) in enumerate(trials.iterrows(), start=1):
                text = pre_sound_text(screen_kind, i, sentences[row['trial']], params)
                r = run_trial(stream, win, kb, params, sounds[row['trial']], text, pre_height, keys_reminder,
                              next_trial, counterbalance)
                display_text('', win, params)                               # blank ITI
                next_trial = r.pop('t_next_base') + params.ITI_S

                out = {c: row[c] for c in params.DESIGN_COLUMNS}
                out.update(session_time=session_time, counterbalance=counterbalance, params_file=params.__name__,
                           pre_sound_screen=screen_kind, screen_text=text,
                           correct=int(r['response'] == row['correct_response']) if r['response'] != 'none' else None,
                           **r)
                append_row(out_path, out, params.OUTPUT_COLUMNS)
                results.append(dict(response=r['response'], correct_response=row['correct_response']))
                rt = r['rt_from_offset_s']
                print(f"block {block} trial {i}/{len(trials)}  {row['stim_id']:16s} {text!r:40s} -> "
                      f"{r['response']:9s} {'' if rt is None else f'RT {rt:.3f} s'}")

            score = score_percent(results, params)
            if block < params.N_BLOCKS:
                tier = 'break_high' if score >= params.SCORE_HIGH else 'break_mid' if score >= params.SCORE_MID else 'break_low'
                show_and_wait(fill(T[tier], values, block=block, score=score), win, kb, params)
            else:
                show_and_wait(fill(T['end'], values, score=score), win, kb, params)

    except QuitExperiment:
        print('Quit key pressed - data saved up to the last completed trial:', out_path)

    finally:                       # always close audio and window; an error still shows its full traceback
        stream.close()
        win.close()
    core.quit()                    # only reached after a normal end or the quit key