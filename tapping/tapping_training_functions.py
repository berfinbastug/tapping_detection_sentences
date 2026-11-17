"""Helper functions and the session of the tapping TRAINING (PsychoPy + Psychtoolbox only).

run_tapping_training_speech.py and run_tapping_training_acoustic.py only choose the setup and the group and then
call run_training() - so both groups always run exactly the same code. The training is separate from the main
experiment: afterwards the experimenter starts run_tapping_speech.py / run_tapping_acoustic.py.

The audio, recording, screen and tap-check functions are copies of the ones in tapping_functions.py (one file per
task); the tap check here can additionally leave out the intervals around a shift.
"""
import os
import json
import shutil
import importlib
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
from psychtoolbox import audio, PsychPortAudio, GetSecs, WaitSecs
from psychopy import core, gui, visual
from psychopy.hardware import keyboard


class QuitExperiment(Exception):
    """Raised when the quit key is pressed."""


class TrainingFailed(Exception):
    """Raised when the last allowed attempt failed: the window closes, the experimenter decides."""


#=====================
# SESSION
#=====================
def get_datetime_string():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def load_params(setup):
    """'lab' -> tapping_training_params, 'laptop' -> tapping_training_params_laptop."""
    modules = {'lab': 'tapping_training_params', 'laptop': 'tapping_training_params_laptop'}
    if setup not in modules:
        raise ValueError(f'unknown SETUP {setup!r}: use {list(modules)}')
    return importlib.import_module(modules[setup])


def check_params(params):
    """Stop early on settings that cannot work together."""
    if params.LOOPBACK_CHANNEL is not None and separate_devices(params):
        raise ValueError('LOOPBACK_CHANNEL needs the sound to come out of the recording device: '
                         'set OUTPUT_DEVICE_NAME / OUTPUT_DEVICE_INDEX to None, or LOOPBACK_CHANNEL to None')
    for c in params.REC_CHANNELS + [params.MIC_CHANNEL]:
        if not 0 <= c < params.N_IN:
            raise ValueError(f'input channel {c} does not exist with N_IN = {params.N_IN}')
    if params.MIC_CHANNEL not in params.REC_CHANNELS:
        raise ValueError('REC_CHANNELS must contain MIC_CHANNEL')
    if len(params.REC_CHANNEL_NAMES) != len(params.REC_CHANNELS):
        raise ValueError('REC_CHANNEL_NAMES needs one name per entry of REC_CHANNELS')
    if params.PRE_SOUND_S - params.REC_PRE_S < 0.2:
        raise ValueError('PRE_SOUND_S - REC_PRE_S is too small (the recording must start >= 0.2 s after the screen)')
    if params.MAX_EXAMPLE_ATTEMPTS < 1 or params.MAX_TEST_RUNS < 1:
        raise ValueError('MAX_EXAMPLE_ATTEMPTS and MAX_TEST_RUNS must be >= 1')


#=====================
# TRAINING STIMULI
#=====================
def load_training(params):
    """The training table, {stim_id: onsets (s from file start)}, {stim_id: kinds}, {stim_id: sound}, meta."""
    folder = params.STIM_DIR
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'no {folder} - run make_training_design_tapping.py and '
                                f'build_tapping_training_stimuli.py first')
    meta = json.load(open(os.path.join(folder, params.STIM_META)))
    for key, value in (('fs', params.FS), ('ioi_ms', params.IOI_MS), ('n_events', params.N_EVENTS)):
        if meta[key] != value:
            raise ValueError(f'{folder}: training stimuli have {key} = {meta[key]}, the params say {value}')
    design = pd.read_csv(os.path.join(folder, params.STIM_DESIGN))
    for phase in ('listen', 'tap_example', 'stumble', 'test'):
        if not (design['phase'] == phase).any():
            raise ValueError(f'{folder}: no {phase!r} trial in {params.STIM_DESIGN}')
    ev = pd.read_csv(os.path.join(folder, params.STIM_EVENTS)).sort_values(['stim_id', 'event'])
    onsets, kinds = {}, {}
    for sid, g in ev.groupby('stim_id'):
        if list(g['event']) != list(range(1, params.N_EVENTS + 1)):
            raise ValueError(f'{params.STIM_EVENTS}: {sid} does not have events 1-{params.N_EVENTS}')
        onsets[sid] = g['actual_onset_ms'].to_numpy(float) / 1000
        kinds[sid] = list(g['kind'])
    stims = np.load(os.path.join(folder, params.STIM_NPY), mmap_mode='r')
    sounds = {}
    for r in design.drop_duplicates('stim_id').itertuples():
        y = np.array(stims[r.stim_row, :r.n_samples], dtype=np.float32) * params.STIM_GAIN
        if np.abs(y).max() >= 1:
            raise ValueError(f'{r.stim_id} clips with STIM_GAIN = {params.STIM_GAIN}')
        sounds[r.stim_id] = y
    return design, onsets, kinds, sounds, meta


def save_snapshot(params, pid, group, setup, session_time):
    """The training meta file + settings (json) and a copy of events_stim.csv next to the data."""
    fmt = dict(pid=pid, group=group, time=session_time)
    meta_path = os.path.join(params.DATA_DIR, params.STIM_META_PATTERN.format(**fmt))
    ev_path = os.path.join(params.DATA_DIR, params.EVENTS_COPY_PATTERN.format(**fmt))
    for p in (meta_path, ev_path):
        if os.path.exists(p):
            raise FileExistsError(f'{p} exists - not overwriting')
    meta = dict(participant=pid, group=group, session_time=session_time, setup=setup, params_file=params.__name__,
                stim_gain=params.STIM_GAIN, fs=params.FS,
                stim_meta=json.load(open(os.path.join(params.STIM_DIR, params.STIM_META))),
                rules=dict(recorded_well_min_sounds=params.RECORDED_WELL_MIN_SOUNDS,
                           double_tap_min_sounds=params.DOUBLE_TAP_MIN_SOUNDS, iti_cap_ms=params.ITI_CAP_MS,
                           iti_off_max_frac=params.ITI_OFF_MAX_FRAC, exclude_after_shift=params.EXCLUDE_AFTER_SHIFT,
                           max_example_attempts=params.MAX_EXAMPLE_ATTEMPTS, max_test_runs=params.MAX_TEST_RUNS,
                           test_pass_min=params.TEST_PASS_MIN),
                recording=dict(input_device=params.INPUT_DEVICE_NAME, output_device=params.OUTPUT_DEVICE_NAME,
                               rec_channels=params.REC_CHANNELS, rec_channel_names=params.REC_CHANNEL_NAMES,
                               rec_pre_s=params.REC_PRE_S, rec_post_s=params.REC_POST_S,
                               rec_subtype=params.REC_SUBTYPE, latency_class=params.LATENCY_CLASS))
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2, default=str)
    shutil.copyfile(os.path.join(params.STIM_DIR, params.STIM_EVENTS), ev_path)


def sentence_text(words, sentence_type, params):
    """Sentences get a capital first letter (CAPITALIZE_FIRST_WORD), word lists not."""
    if params.CAPITALIZE_FIRST_WORD and sentence_type != 'word list':
        words = words[0].upper() + words[1:]
    return words


def trial_screen(group, n, row, params):
    """[(text, height, y)]: speech = number on top + the words; acoustic = 'Durchgang n'."""
    if params.PRE_SOUND_SCREEN[group] == 'sentence':
        return [(params.NUMBER_ABOVE_TEXT.format(n=n), params.NUMBER_HEIGHT, params.NUMBER_POS_Y),
                (sentence_text(row['words'], row['sentence_type'], params), params.SENTENCE_HEIGHT, 0)]
    return [(params.TRIAL_NUMBER_TEXT.format(n=n), params.TEXT_HEIGHT, 0)]


#=====================
# AUDIO (copy of tapping_functions.py)
#=====================
def separate_devices(params):
    return params.OUTPUT_DEVICE_NAME is not None or params.OUTPUT_DEVICE_INDEX is not None


def find_device(devices, name, index, n_in, n_out, role):
    if index is not None:
        match = [d for d in devices if d['DeviceIndex'] == index]
    else:
        match = [d for d in devices if name.lower() in d['DeviceName'].lower()
                 and d['NrInputChannels'] >= n_in and d['NrOutputChannels'] >= n_out]
    if len(match) != 1:
        raise RuntimeError(f'{len(match)} devices match {name!r} with >= {n_in} inputs and >= {n_out} outputs '
                           f'({role}): set the DeviceIndex in the params (see the list above)')
    print(f"-> {role}: DeviceIndex {int(match[0]['DeviceIndex'])}, {match[0]['DeviceName']}")
    return int(match[0]['DeviceIndex'])


def open_audio(params, max_sound_s):
    devices = audio.get_devices()
    print('\nAudio devices:')
    for d in devices:
        print(f"  DeviceIndex {int(d['DeviceIndex']):>3} | {d['HostAudioAPIName']:<12} | in {int(d['NrInputChannels']):>2} "
              f"out {int(d['NrOutputChannels']):>2} | {d['DeviceName']}")
    separate = separate_devices(params)
    in_index = find_device(devices, params.INPUT_DEVICE_NAME, params.INPUT_DEVICE_INDEX, params.N_IN,
                           0 if separate else params.N_OUT, 'recording' if separate else 'sound + recording')
    if separate:
        out_index = find_device(devices, params.OUTPUT_DEVICE_NAME or '', params.OUTPUT_DEVICE_INDEX, 0,
                                params.N_OUT, 'sound')
        play = audio.Stream(freq=params.FS, device_id=out_index, mode=1, latency_class=params.LATENCY_CLASS,
                            channels=params.N_OUT)
        cap = audio.Stream(freq=params.FS, device_id=in_index, mode=2, latency_class=params.LATENCY_CLASS,
                           channels=params.N_IN)
    else:
        out_index = in_index
        play = cap = audio.Stream(freq=params.FS, device_id=in_index, mode=3, latency_class=params.LATENCY_CLASS,
                                  channels=[params.N_OUT, params.N_IN])
    cap.get_audio_data(secs_allocate=max_sound_s + params.REC_PRE_S + params.REC_POST_S + params.CAPTURE_SPARE_S)
    PsychPortAudio('FillBuffer', play.handle, np.zeros((int(0.1 * params.FS), params.N_OUT), np.float32))
    if separate:
        PsychPortAudio('Start', cap.handle, 0, 0, 1)
    PsychPortAudio('Start', play.handle, 1, 0, 1)
    PsychPortAudio('Stop', play.handle, 1)
    if separate:
        PsychPortAudio('Stop', cap.handle)
    cap.get_audio_data()
    return dict(play=play, cap=cap, separate=separate, in_index=in_index, out_index=out_index)


def close_audio(aud):
    if aud is None:
        return
    aud['play'].close()
    if aud['separate']:
        aud['cap'].close()


def make_buffer(y, params):
    pre, post = int(round(params.REC_PRE_S * params.FS)), int(round(params.REC_POST_S * params.FS))
    buf = np.r_[np.zeros(pre, np.float32), np.asarray(y, np.float32), np.zeros(post, np.float32)]
    return np.ascontiguousarray(np.tile(buf[:, None], (1, params.N_OUT)))


def start_play_and_record(aud, buffer, t_start):
    play, cap = aud['play'], aud['cap']
    cap.get_audio_data()
    play.fill_buffer(buffer)
    if aud['separate']:
        PsychPortAudio('Start', cap.handle, 0, 0, 1)
    play.start(when=t_start, wait_for_start=0)


def finish_play_and_record(aud, t_end, kb, params):
    play, cap = aud['play'], aud['cap']
    quit_pressed = False
    while GetSecs() < t_end:
        if kb.getKeys(keyList=[params.QUIT_KEY], waitRelease=False):
            quit_pressed = True
            break
        WaitSecs(params.QUIT_POLL_S)
    stopped = PsychPortAudio('Stop', play.handle, 0 if quit_pressed else 1)
    if aud['separate']:
        PsychPortAudio('Stop', cap.handle)
    rec, _absrecpos, overflow, t_capture_start = cap.get_audio_data()
    if quit_pressed:
        raise QuitExperiment
    rec = np.asarray(rec, dtype=np.float32)
    if rec.ndim == 1:
        rec = rec[:, None]
    return dict(rec=rec, t_rec_start=float(stopped[0]), t_capture_start=float(t_capture_start),
                overflow=bool(overflow))


def save_recording(path, rec, params):
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists - not overwriting')
    sf.write(path, rec[:, params.REC_CHANNELS], params.FS, subtype=params.REC_SUBTYPE)


#=====================
# TAP CHECK (works on a copy; the saved wav is not changed)
#=====================
def envelope(x, win_ms, fs):
    n = max(1, int(round(win_ms / 1000 * fs)))
    c = np.r_[0.0, np.cumsum(np.asarray(x, float) ** 2)]
    i = np.arange(len(x))
    lo, hi = np.clip(i - n // 2, 0, len(x)), np.clip(i + (n - 1) // 2 + 1, 0, len(x))
    return np.sqrt(np.maximum(c[hi] - c[lo], 0) / n)


def detect_taps(x, params):
    sos = butter(4, params.TAP_HIGHPASS_HZ, 'highpass', fs=params.FS, output='sos')
    env = envelope(sosfiltfilt(sos, np.asarray(x, float)), params.TAP_SMOOTH_MS, params.FS)
    med = np.median(env)
    thr = med + params.TAP_THRESHOLD_MAD * np.median(np.abs(env - med))
    n_ref = params.TAP_REFRACTORY_MS / 1000 * params.FS
    taps, last = [], -np.inf
    for i in np.flatnonzero(env > thr):
        if i - last > n_ref:
            taps.append(i)
            last = i
    return np.array(taps) / params.FS


def tap_check(mic, sound_start_in_rec, onsets_s, params, shifted_event=None):
    """The break-screen rules of the main experiment (see tapping_functions.tap_check), plus: if shifted_event
    (1-based number of the first sound that moved) is given, the interval with the shift and the next
    EXCLUDE_AFTER_SHIFT intervals are left out of the timing check. Also returns the taps (s from file start)."""
    taps = detect_taps(mic, params)
    on = sound_start_in_rec + np.asarray(onsets_s, float)
    half = params.IOI_MS / 2000
    edges = np.r_[on[0] - half, (on[:-1] + on[1:]) / 2, on[-1] + half]
    which = np.searchsorted(edges, taps, side='right') - 1
    during = (which >= 0) & (which < len(on))
    per_sound = np.bincount(which[during], minlength=len(on))
    first_tap = np.full(len(on), np.nan)
    for k in range(len(on)):
        if per_sound[k] == 1:
            first_tap[k] = taps[during][which[during] == k][0]
    iti = np.diff(first_tap) * 1000                            # iti[j] = interval between sound j+1 and j+2
    n_excluded = 0
    if shifted_event is not None and not pd.isna(shifted_event):
        j0 = int(shifted_event) - 2                             # the interval that ends on the moved sound
        excl = np.arange(j0, j0 + params.EXCLUDE_AFTER_SHIFT + 1)
        excl = excl[(excl >= 0) & (excl < len(iti))]
        n_excluded = int(np.sum(~np.isnan(iti[excl])))
        iti[excl] = np.nan
    iti = iti[~np.isnan(iti)]
    off_frac = float(np.mean(np.abs(iti - params.IOI_MS) > params.ITI_CAP_MS)) if len(iti) else 1.0
    n_tapped, n_double = int(np.sum(per_sound >= 1)), int(np.sum(per_sound >= 2))
    well = n_tapped >= params.RECORDED_WELL_MIN_SOUNDS
    double = n_double >= params.DOUBLE_TAP_MIN_SOUNDS
    timing = off_frac > params.ITI_OFF_MAX_FRAC
    return dict(online_n_taps=len(taps), online_n_taps_during=int(during.sum()),
                online_n_sounds_tapped=n_tapped, online_n_sounds_double=n_double,
                online_iti_median_ms=round(float(np.median(iti)), 1) if len(iti) else None,
                online_iti_off_frac=round(off_frac, 3), online_iti_excluded=n_excluded,
                online_recorded_well=well, online_flag_double=double, online_flag_timing=timing,
                online_passed=bool(well and not double and not timing),
                taps_in_file_s=taps - sound_start_in_rec)


def problem_lines(checks, T):
    """The fb_ lines for the problems that occurred in any of these checks (missing, double, timing)."""
    lines = []
    if any(not c['online_recorded_well'] for c in checks):
        lines.append(T['fb_missing'])
    if any(c['online_flag_double'] for c in checks):
        lines.append(T['fb_double'])
    if any(c['online_flag_timing'] for c in checks):
        lines.append(T['fb_timing'])
    return lines


#=====================
# SCREEN AND KEYS
#=====================
def display_text(text, win, params, height=None, y=0):
    if text:
        visual.TextStim(win, text=text, height=height or params.TEXT_HEIGHT, pos=(0, y),
                        wrapWidth=params.WRAP_WIDTH).draw()
    win.flip()
    return GetSecs()


def display_screen(items, win, params):
    for text, height, y in items:
        visual.TextStim(win, text=text, height=height, pos=(0, y), wrapWidth=params.WRAP_WIDTH).draw()
    win.flip()
    return GetSecs()


def draw_tap_figure(win, params, onsets_s, kinds, taps_s):
    """Row of the 30 sounds (■ noise, ● word / tone) on a time line, the taps as ticks underneath. No text."""
    t0, t1 = onsets_s[0] - 0.3, onsets_s[-1] + 0.5
    x_of = lambda t: (t - t0) / (t1 - t0) * params.FIG_WIDTH - params.FIG_WIDTH / 2
    s = params.FIG_MARK_SIZE
    for t, kind in zip(onsets_s, kinds):
        if kind == 'noise':
            visual.Rect(win, width=s, height=s, pos=(x_of(t), params.FIG_Y), fillColor=params.FIG_COLOR,
                        lineColor=params.FIG_COLOR).draw()
        else:
            visual.Circle(win, radius=s / 2, pos=(x_of(t), params.FIG_Y), fillColor=params.FIG_COLOR,
                          lineColor=params.FIG_COLOR).draw()
    for t in taps_s:
        if t0 <= t <= t1:
            x = x_of(t)
            visual.Line(win, start=(x, params.FIG_TAP_Y - params.FIG_TAP_LEN / 2),
                        end=(x, params.FIG_TAP_Y + params.FIG_TAP_LEN / 2), lineColor=params.FIG_COLOR,
                        lineWidth=params.FIG_TAP_WIDTH).draw()


def wait_until(t, kb, params):
    while True:
        left = t - GetSecs()
        if left <= 0:
            return
        if kb.getKeys(keyList=[params.QUIT_KEY], waitRelease=False):
            raise QuitExperiment
        WaitSecs(min(left, params.QUIT_POLL_S))


def wait_continue(kb, params):
    keys = kb.waitKeys(keyList=params.CONTINUE_KEYS + [params.QUIT_KEY], waitRelease=True)
    if keys and keys[0].name == params.QUIT_KEY:
        raise QuitExperiment


def wait_choice(kb, params):
    """LEFT ('2') or RIGHT ('3') -> 'left' / 'right'; the quit key raises QuitExperiment."""
    keys = kb.waitKeys(keyList=params.RESPONSE_KEYS + [params.QUIT_KEY], waitRelease=True)
    if keys[0].name == params.QUIT_KEY:
        raise QuitExperiment
    return 'left' if keys[0].name == params.RESPONSE_KEYS[0] else 'right'


def show_and_wait(text, win, kb, params):
    print(text)
    display_text(text, win, params)
    wait_continue(kb, params)


def fill(text, values, **extra):
    return text.format(**{**values, **extra})


#=====================
# OUTPUT
#=====================
def append_row(path, row, columns):
    new = not os.path.exists(path)
    pd.DataFrame([row]).reindex(columns=columns).to_csv(path, sep='\t', mode='a', header=new, index=False)


#=====================
# ONE TRIAL (as in the main experiment)
#=====================
def run_trial(aud, win, kb, params, y, screen_items, t_screen, rec_path):
    """trial screen at t_screen -> sound PRE_SOUND_S later (the screen stays) -> blank when the sound ends;
    recorded from REC_PRE_S before to REC_POST_S after the sound -> raw wav saved."""
    buffer = make_buffer(y, params)
    wait_until(t_screen, kb, params)
    t_screen_on = display_screen(screen_items, win, params)
    t_sound_planned = t_screen_on + params.PRE_SOUND_S
    t_rec_scheduled = t_sound_planned - params.REC_PRE_S
    start_play_and_record(aud, buffer, t_rec_scheduled)
    wait_until(t_sound_planned + len(y) / params.FS, kb, params)
    t_screen_off = display_text('', win, params)
    r = finish_play_and_record(aud, t_rec_scheduled + len(buffer) / params.FS, kb, params)
    rec = r['rec']
    t_sound_onset = r['t_rec_start'] + params.REC_PRE_S
    sound_start_in_rec = r['t_rec_start'] - r['t_capture_start'] + params.REC_PRE_S
    save_recording(rec_path, rec, params)
    return dict(t_screen_on=t_screen_on, t_screen_off=t_screen_off, t_rec_scheduled=t_rec_scheduled,
                t_rec_start=r['t_rec_start'], t_capture_start=r['t_capture_start'],
                t_sound_onset=t_sound_onset, t_sound_offset=t_sound_onset + len(y) / params.FS,
                sound_start_in_rec_s=sound_start_in_rec,
                rec_file=os.path.basename(rec_path), rec_channels=','.join(params.REC_CHANNEL_NAMES),
                rec_n_samples=len(rec), rec_s=len(rec) / params.FS, overflow=r['overflow'],
                peak_per_channel=','.join(f'{p:.4f}' for p in np.abs(rec[:, params.REC_CHANNELS]).max(axis=0)),
                mic=rec[:, params.MIC_CHANNEL], rec_s_planned=len(buffer) / params.FS)


#=====================
# THE TRAINING SESSION (shared by both groups)
#=====================
def run_training(params, setup, group):
    """Dialog -> examples (listen, tap, stumble) with feedback and retries -> test run (repeated as needed) -> end.
    Every training trial is saved immediately: one row in the training file and one raw wav. If the last allowed
    attempt fails, the window closes."""
    check_params(params)
    info = dict(params.DIALOG_FIELDS)
    dlg = gui.DlgFromDict(dictionary=info, title=f'Tapping TRAINING - {group} group ({setup})')
    if not dlg.OK:
        core.quit()
    pid = str(info['participant_id']).strip().upper()

    design, onsets, kinds, sounds, _meta = load_training(params)
    max_sound_s = max(len(y) for y in sounds.values()) / params.FS
    os.makedirs(params.DATA_DIR, exist_ok=True)
    session_time = get_datetime_string()
    fmt = dict(pid=pid, group=group, time=session_time)
    out_path = os.path.join(params.DATA_DIR, params.OUTPUT_FILE_PATTERN.format(**fmt))
    rec_dir = os.path.join(params.DATA_DIR, params.REC_DIR_PATTERN.format(**fmt))
    if os.path.exists(out_path) or os.path.exists(rec_dir):
        raise FileExistsError(f'{out_path} or {rec_dir} exists - not overwriting')
    os.makedirs(rec_dir)
    save_snapshot(params, pid, group, setup, session_time)
    print('Training output:', out_path, '| recordings:', rec_dir)

    T = params.TEXTS[group]
    example = design[design.phase == 'tap_example'].iloc[0]
    values = dict(words=sentence_text(example['words'], example['sentence_type'], params))

    aud = win = None
    try:
        aud = open_audio(params, max_sound_s)
        win = visual.Window(params.WIN_SIZE, fullscr=params.FULLSCREEN, monitor=params.MONITOR,
                            units=params.UNITS, screen=params.SCREEN)
        win.mouseVisible = False
        kb = keyboard.Keyboard()

        def play(phase, rnd, n, row, t_screen, check=True):
            """One training trial: play + record + save + tap check + one row in the training file."""
            items = trial_screen(group, n, row, params)
            rec_path = os.path.join(rec_dir, params.REC_FILE_PATTERN.format(phase=phase, rnd=rnd, trial=n))
            r = run_trial(aud, win, kb, params, sounds[row['stim_id']], items, t_screen, rec_path)
            c = tap_check(r.pop('mic'), r['sound_start_in_rec_s'], onsets[row['stim_id']], params,
                          row['jitter_event'] if row['jittered'] else None)
            taps = c.pop('taps_in_file_s')
            out = {k: row[k] for k in ('item_id', 'sentence_type', 'words', 'stim_id', 'jittered', 'jitter_part',
                                       'jitter_interval', 'jitter_event', 'shift_ms', 'stim_row', 'n_samples',
                                       'tokens')}
            out.update(participant=pid, group=group, session_time=session_time, setup=setup,
                       params_file=params.__name__, phase=phase, round=rnd, trial=n,
                       screen_text=' | '.join(t for t, _, _ in items), checked=check,
                       **{k: v for k, v in r.items() if k != 'rec_s_planned'}, **c)
            append_row(out_path, out, params.OUTPUT_COLUMNS)
            print(f"{phase} round {rnd} trial {n} {row['stim_id']}: sounds tapped {c['online_n_sounds_tapped']}/"
                  f"{params.N_EVENTS}, double {c['online_n_sounds_double']}, ITI off {c['online_iti_off_frac']:.2f} "
                  f"-> {'PASSED' if c['online_passed'] else 'not passed'}" + ('' if check else ' (not checked)'))
            return r, c, taps

        def example_loop(phase, with_figure):
            """Tap example / stumble example: repeat until passed (and RIGHT); MAX_EXAMPLE_ATTEMPTS failures close."""
            row = design[design.phase == phase].iloc[0]
            fails, rnd = 0, 0
            while True:
                rnd += 1
                display_text('', win, params)
                _, c, taps = play(phase, rnd, 1, row, GetSecs() + params.FIRST_TRIAL_DELAY_S)
                if with_figure:
                    show_and_wait(T['see_taps'], win, kb, params)
                if c['online_passed']:
                    text = T['fb_pass']
                else:
                    fails += 1
                    if fails >= params.MAX_EXAMPLE_ATTEMPTS:
                        raise TrainingFailed(f'{phase}: {fails} failed attempts')
                    text = '\n\n'.join(problem_lines([c], T) + [T['fb_retry']])
                print(text)
                if with_figure:
                    draw_tap_figure(win, params, onsets[row['stim_id']], kinds[row['stim_id']], taps)
                    display_text(text, win, params, y=params.FIG_TEXT_Y)
                else:
                    display_text(text, win, params)
                if not c['online_passed']:
                    wait_continue(kb, params)                       # any key: try again
                elif wait_choice(kb, params) == 'right':            # LEFT = again, RIGHT = continue
                    return

        # ---------- 1-5: explanation and the listening example
        show_and_wait(T['welcome'], win, kb, params)
        show_and_wait(T['hear'], win, kb, params)
        show_and_wait(T['example_intro'], win, kb, params)
        display_text('', win, params)
        play('listen', 1, 1, design[design.phase == 'listen'].iloc[0], GetSecs() + params.FIRST_TRIAL_DELAY_S,
             check=False)
        show_and_wait(fill(T['structure'], values), win, kb, params)

        # ---------- 6-8: tapping example with the figure
        show_and_wait(T['task'], win, kb, params)
        example_loop('tap_example', with_figure=True)

        # ---------- 9-12: stumble example with written feedback
        show_and_wait(T['shifts'], win, kb, params)
        show_and_wait(T['stumble_intro'], win, kb, params)
        example_loop('stumble', with_figure=False)

        # ---------- 13+: test run (automatic, like the main trials)
        show_and_wait(T['test_intro'], win, kb, params)
        test = design[design.phase == 'test'].sort_values('trial')
        for run in range(1, params.MAX_TEST_RUNS + 1):
            display_text('', win, params)
            t_next = GetSecs() + params.FIRST_TRIAL_DELAY_S
            checks = []
            for _, row in test.iterrows():
                r, c, _ = play('test', run, int(row['trial']), row, t_next)
                t_next = r['t_rec_start'] + r['rec_s_planned'] + params.SAVE_MARGIN_S + params.ITI_S
                checks.append(c)
            n_pass = sum(c['online_passed'] for c in checks)
            res = dict(values, n_pass=n_pass, n_test=len(checks),
                       feedback=''.join(f + '\n\n' for f in problem_lines(checks, T)))
            print(f'test run {run}: {n_pass} of {len(checks)} passed')
            if n_pass >= params.TEST_PASS_MIN:
                show_and_wait(fill(T['test_pass'], res), win, kb, params)
                break
            if run == params.MAX_TEST_RUNS:
                raise TrainingFailed(f'test run: {run} runs without passing ({n_pass} of {len(checks)} passed)')
            show_and_wait(fill(T['test_fail'], res), win, kb, params)

        show_and_wait(T['end'], win, kb, params)
        print('\nTraining completed. Start the main experiment next.')

    except QuitExperiment:
        print('Quit key pressed - training data saved up to the last completed trial:', out_path)
    except TrainingFailed as e:
        print(f'\n!!! TRAINING STOPPED: {e}. The window was closed - decide how to go on. Data: {out_path}')

    finally:
        close_audio(aud)
        if win is not None:
            win.close()
    core.quit()