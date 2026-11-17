"""Helper functions and the shared session for the tapping experiment (PsychoPy + Psychtoolbox only).

run_tapping_speech.py and run_tapping_acoustic.py only choose the setup and the group and then call
run_session() - so both groups always run exactly the same code.

Per trial: trial screen (number + sentence / 'Durchgang n') -> sound starts, the screen stays until the sound ends;
recorded at the same time
-> the RAW recording is saved as a wav (no filtering) -> one row in the data file. Trials follow each other
automatically; there is a break screen after every block.
"""
import os
import json
import shutil
import hashlib
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


#=====================
# SESSION
#=====================
def get_datetime_string():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def load_params(setup):
    """'lab' -> tapping_params, 'laptop' -> tapping_params_laptop."""
    modules = {'lab': 'tapping_params', 'laptop': 'tapping_params_laptop'}
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
    if params.REC_PRE_S < 0 or params.REC_POST_S < 0:
        raise ValueError('REC_PRE_S and REC_POST_S must be >= 0')
    if params.PRE_SOUND_S - params.REC_PRE_S < 0.2:
        raise ValueError('the recording must start at least 0.2 s after the trial screen appears: '
                         'PRE_SOUND_S - REC_PRE_S is too small')


#=====================
# DESIGN AND STIMULI
#=====================
def participant_dir(params, pid, group):
    """stimuli_tapping/<pid>_<group>/ (built by build_tapping_stimuli.py)."""
    folder = os.path.join(params.STIM_DIR, params.PARTICIPANT_FOLDER.format(pid=pid, group=group))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'no stimulus folder {folder} - check the participant ID, that this is the right '
                                f'run script for the {group} group, and that build_tapping_stimuli.py was run')
    return folder


def load_stim_meta(params, folder, pid, group):
    """Read the participant's stimulus meta file and check it matches the params and the participant."""
    meta = json.load(open(os.path.join(folder, params.STIM_META)))
    if meta['participant'] != pid or meta['group'] != group:
        raise ValueError(f"{folder}: stimuli were built for {meta['participant']}/{meta['group']}, not {pid}/{group}")
    for key, value in (('fs', params.FS), ('ioi_ms', params.IOI_MS), ('n_events', params.N_EVENTS)):
        if meta[key] != value:
            raise ValueError(f'{folder}: stimuli have {key} = {meta[key]}, the params say {value}')
    return meta


def load_design(params, folder, pid, group):
    """The participant's design table (with stim_row / n_samples / words / tokens), checked."""
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


def load_events(params, folder, design):
    """{trial: actual onsets of the 30 sounds in s from the start of the stimulus file}, checked."""
    path = os.path.join(folder, params.STIM_EVENTS)
    ev = pd.read_csv(path).sort_values(['trial', 'event'])
    if set(ev['trial']) != set(design['trial']):
        raise ValueError(f'{path}: trials differ from the design table')
    onsets = {}
    for trial, g in ev.groupby('trial'):
        if list(g['event']) != list(range(1, params.N_EVENTS + 1)):
            raise ValueError(f'{path}: trial {trial} does not have events 1-{params.N_EVENTS}')
        onsets[trial] = g['actual_onset_ms'].to_numpy(float) / 1000
    return onsets


def load_participant_sounds(params, folder, design):
    """Copy this participant's sounds from their .npy into memory (mono float32, gain applied). Returns {trial: signal}."""
    stims = np.load(os.path.join(folder, params.STIM_NPY), mmap_mode='r')
    sounds = {}
    for row in design.itertuples():
        y = np.array(stims[row.stim_row, :row.n_samples], dtype=np.float32) * params.STIM_GAIN
        if np.abs(y).max() >= 1:
            raise ValueError(f'trial {row.trial} clips with STIM_GAIN = {params.STIM_GAIN}')
        sounds[row.trial] = y
    return sounds


def sentence_text(words, sentence_type, params):
    """Words as shown on screen: sentences get a capital first letter (CAPITALIZE_FIRST_WORD), word lists not."""
    if params.CAPITALIZE_FIRST_WORD and sentence_type != 'word list':
        words = words[0].upper() + words[1:]
    return words


PRE_SOUND_KINDS = ('sentence', 'trial_number')


def check_pre_sound_kind(params, group):
    kind = params.PRE_SOUND_SCREEN[group]
    if kind not in PRE_SOUND_KINDS:
        raise ValueError(f'PRE_SOUND_SCREEN[{group!r}] must be one of {PRE_SOUND_KINDS}, not {kind!r}')
    return kind


def trial_screen(kind, n, row, params):
    """What the trial screen shows (n = trial within the block), as a list of (text, height, y position):
    'sentence'     -> the trial number on top (NUMBER_ABOVE_TEXT at NUMBER_POS_Y) + the sentence in the centre
    'trial_number' -> 'Durchgang n' in the centre
    """
    if kind == 'sentence':
        return [(params.NUMBER_ABOVE_TEXT.format(n=n), params.NUMBER_HEIGHT, params.NUMBER_POS_Y),
                (sentence_text(row['words'], row['sentence_type'], params), params.SENTENCE_HEIGHT, 0)]
    return [(params.TRIAL_NUMBER_TEXT.format(n=n), params.TEXT_HEIGHT, 0)]


def sound_md5(y):
    """Checksum of a sound exactly as played (float32 samples)."""
    return hashlib.md5(np.ascontiguousarray(y, dtype=np.float32).tobytes()).hexdigest()


def file_md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def file_info(path):
    """Size, last-modified time and checksum of a file (to know later which stimulus build was used)."""
    st = os.stat(path)
    return dict(path=os.path.abspath(path), bytes=st.st_size,
                modified=datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'))


def save_stim_snapshot(params, folder, design, sounds, pid, group, setup, session_time):
    """Keep this participant's stimulus information next to the data (new files per session, never overwritten).

    tapping_stiminfo_...csv : one row per trial: sentence, words, noise tokens, npy row, md5 of the sound as played
    tapping_stimmeta_...json: the stimulus meta file, which .npy / tables were used, gain, setup, recording settings
    tapping_events_...csv   : copy of events_stim.csv (the onset of every sound: needed to analyse the taps)
    """
    fmt = dict(pid=pid, group=group, time=session_time)
    paths = {k: os.path.join(params.DATA_DIR, getattr(params, k).format(**fmt))
             for k in ('STIM_INFO_PATTERN', 'STIM_META_PATTERN', 'EVENTS_COPY_PATTERN')}
    for path in paths.values():
        if os.path.exists(path):
            raise FileExistsError(f'{path} exists - not overwriting')

    rows = [dict(trial=r.trial, block=r.block, sentence_id=r.sentence_id, words=r.words, tokens=r.tokens,
                 npy_row=r.stim_row, n_samples=r.n_samples, sound_md5=sound_md5(sounds[r.trial]))
            for r in design.itertuples()]
    pd.DataFrame(rows).to_csv(paths['STIM_INFO_PATTERN'], index=False)

    events_path = os.path.join(folder, params.STIM_EVENTS)
    meta = dict(participant=pid, group=group, session_time=session_time, setup=setup, params_file=params.__name__,
                stim_gain=params.STIM_GAIN, fs=params.FS,
                stim_meta=json.load(open(os.path.join(folder, params.STIM_META))),
                npy=file_info(os.path.join(folder, params.STIM_NPY)),
                design=dict(file_info(os.path.join(folder, params.STIM_DESIGN)),
                            md5=file_md5(os.path.join(folder, params.STIM_DESIGN))),
                events=dict(file_info(events_path), md5=file_md5(events_path)),
                recording=dict(input_device=params.INPUT_DEVICE_NAME, output_device=params.OUTPUT_DEVICE_NAME,
                               n_in=params.N_IN, rec_channels=params.REC_CHANNELS,
                               rec_channel_names=params.REC_CHANNEL_NAMES, rec_pre_s=params.REC_PRE_S,
                               rec_post_s=params.REC_POST_S, rec_subtype=params.REC_SUBTYPE,
                               latency_class=params.LATENCY_CLASS),
                timing=dict(pre_sound_s=params.PRE_SOUND_S,
                            save_margin_s=params.SAVE_MARGIN_S, iti_s=params.ITI_S))
    with open(paths['STIM_META_PATTERN'], 'w') as fh:
        json.dump(meta, fh, indent=2, default=str)
    shutil.copyfile(events_path, paths['EVENTS_COPY_PATTERN'])
    print('Stimulus snapshot:', paths['STIM_INFO_PATTERN'])


#=====================
# AUDIO
#=====================
def separate_devices(params):
    """True = sound and recording on two devices (laptop); False = one full-duplex device (lab)."""
    return params.OUTPUT_DEVICE_NAME is not None or params.OUTPUT_DEVICE_INDEX is not None


def find_device(devices, name, index, n_in, n_out, role):
    """The DeviceIndex of the one device matching `name` (or `index`) with at least n_in inputs / n_out outputs."""
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
    """Open the streams, allocate the capture buffer, wake the device(s) up.

    Lab (one device): one full-duplex stream (mode 3) -> playback and capture start together on one clock.
    Laptop (two devices): a playback stream and a capture stream (separate clocks; aligned by timestamps).
    Returns dict(play, cap, separate, in_index, out_index).
    """
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
    # the capture buffer must be allocated before the first start
    cap.get_audio_data(secs_allocate=max_sound_s + params.REC_PRE_S + params.REC_POST_S + params.CAPTURE_SPARE_S)

    # warm-up: 0.1 s of silence once (and record it), then empty the capture buffer
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
    """REC_PRE_S silence + the sound + REC_POST_S silence, on every output: (n, N_OUT) float32.
    Playing the silences keeps the recording running before and after the sound (also in full-duplex mode)."""
    pre, post = int(round(params.REC_PRE_S * params.FS)), int(round(params.REC_POST_S * params.FS))
    buf = np.r_[np.zeros(pre, np.float32), np.asarray(y, np.float32), np.zeros(post, np.float32)]
    return np.ascontiguousarray(np.tile(buf[:, None], (1, params.N_OUT)))


def start_play_and_record(aud, buffer, t_start):
    """Load the buffer and schedule it at t_start (GetSecs time); returns at once. Recording starts now
    (two devices) or together with the playback (one full-duplex device)."""
    play, cap = aud['play'], aud['cap']
    cap.get_audio_data()                                   # empty leftovers of the previous trial
    play.fill_buffer(buffer)
    if aud['separate']:
        PsychPortAudio('Start', cap.handle, 0, 0, 1)       # runs until stopped
    play.start(when=t_start, wait_for_start=0)


def finish_play_and_record(aud, t_end, kb, params):
    """Wait until the buffer has played (checking the quit key), stop, and fetch the recording.
    Returns dict(rec (n, N_IN) float32, t_rec_start, t_capture_start, overflow)."""
    play, cap = aud['play'], aud['cap']
    quit_pressed = False
    while GetSecs() < t_end:
        if kb.getKeys(keyList=[params.QUIT_KEY], waitRelease=False):
            quit_pressed = True
            break
        WaitSecs(params.QUIT_POLL_S)
    # 'Stop' returns (startTime, endPositionSecs, xruns, estStopTime): startTime = when the buffer really started
    stopped = PsychPortAudio('Stop', play.handle, 0 if quit_pressed else 1)   # 1 = wait until the buffer has played
    if aud['separate']:
        PsychPortAudio('Stop', cap.handle)
    t_rec_start = stopped[0]
    rec, _absrecpos, overflow, t_capture_start = cap.get_audio_data()
    if quit_pressed:
        raise QuitExperiment
    rec = np.asarray(rec, dtype=np.float32)
    if rec.ndim == 1:
        rec = rec[:, None]
    return dict(rec=rec, t_rec_start=float(t_rec_start), t_capture_start=float(t_capture_start),
                overflow=bool(overflow))


def save_recording(path, rec, params):
    """The RAW recording of the channels in REC_CHANNELS as a wav (no filtering, no scaling). Never overwrites."""
    if os.path.exists(path):
        raise FileExistsError(f'{path} exists - not overwriting')
    sf.write(path, rec[:, params.REC_CHANNELS], params.FS, subtype=params.REC_SUBTYPE)


#=====================
# ONLINE TAP CHECK (console only: works on a copy, the saved wav is not changed)
#=====================
def envelope(x, win_ms, fs):
    n = max(1, int(round(win_ms / 1000 * fs)))
    c = np.r_[0.0, np.cumsum(np.asarray(x, float) ** 2)]
    i = np.arange(len(x))
    lo, hi = np.clip(i - n // 2, 0, len(x)), np.clip(i + (n - 1) // 2 + 1, 0, len(x))
    return np.sqrt(np.maximum(c[hi] - c[lo], 0) / n)


def detect_taps(x, params):
    """Tap onsets (s from the start of the recording), as in tapping_hardware_test.py:
    high-pass, 1 ms RMS envelope, threshold = median + TAP_THRESHOLD_MAD x MAD, refractory period."""
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


def tap_check(mic, sound_start_in_rec, onsets_s, params):
    """Quick check of one trial's taps (console + break-screen feedback; the saved wav is not touched).

    Every sound gets a window from halfway to the previous sound to halfway to the next one (the first / last
    window extends half an IOI before / after). Taps are counted per window.
      recorded_well : at least RECORDED_WELL_MIN_SOUNDS sounds got a tap
      flag_double   : at least DOUBLE_TAP_MIN_SOUNDS sounds got 2+ taps
      flag_timing   : more than ITI_OFF_MAX_FRAC of the inter-tap intervals are outside IOI_MS +- ITI_CAP_MS
                      (intervals only between neighbouring sounds that got exactly one tap each)
    """
    taps = detect_taps(mic, params)
    on = sound_start_in_rec + np.asarray(onsets_s, float)
    half = params.IOI_MS / 2000
    edges = np.r_[on[0] - half, (on[:-1] + on[1:]) / 2, on[-1] + half]
    which = np.searchsorted(edges, taps, side='right') - 1                # window of every tap (-1 / n = outside)
    during = (which >= 0) & (which < len(on))
    per_sound = np.bincount(which[during], minlength=len(on))
    first_tap = np.full(len(on), np.nan)
    for k in range(len(on)):
        if per_sound[k] == 1:
            first_tap[k] = taps[during][which[during] == k][0]
    iti = np.diff(first_tap) * 1000                                       # nan where a neighbour has 0 or 2+ taps
    iti = iti[~np.isnan(iti)]
    off_frac = float(np.mean(np.abs(iti - params.IOI_MS) > params.ITI_CAP_MS)) if len(iti) else 1.0
    n_tapped, n_double = int(np.sum(per_sound >= 1)), int(np.sum(per_sound >= 2))
    return dict(online_n_taps=len(taps), online_n_taps_during=int(during.sum()),
                online_n_sounds_tapped=n_tapped, online_n_sounds_double=n_double,
                online_iti_median_ms=round(float(np.median(iti)), 1) if len(iti) else None,
                online_iti_off_frac=round(off_frac, 3),
                online_recorded_well=n_tapped >= params.RECORDED_WELL_MIN_SOUNDS,
                online_flag_double=n_double >= params.DOUBLE_TAP_MIN_SOUNDS,
                online_flag_timing=off_frac > params.ITI_OFF_MAX_FRAC)


def block_feedback(checks, T, params):
    """Break-screen values from the tap checks of one block: n_good, n_played and the feedback texts that apply."""
    n_bad = sum(not c['online_recorded_well'] for c in checks)
    n_double = sum(c['online_flag_double'] for c in checks)
    n_timing = sum(c['online_flag_timing'] for c in checks)
    fb = [T[key] for key, n in (('fb_missing', n_bad), ('fb_double', n_double), ('fb_timing', n_timing))
          if n >= params.BLOCK_WARN_MIN_TRIALS]
    return dict(n_good=len(checks) - n_bad, n_played=len(checks), feedback=''.join(f + '\n\n' for f in fb))


#=====================
# SCREEN AND KEYS
#=====================
def display_text(text, win, params, height=None):
    """Show only this text (empty string = blank screen). Returns GetSecs() right after the flip."""
    if text:
        visual.TextStim(win, text=text, height=height or params.TEXT_HEIGHT, wrapWidth=params.WRAP_WIDTH).draw()
    win.flip()
    return GetSecs()


def display_screen(items, win, params):
    """Show several texts at once: items = [(text, height, y position), ...]. Returns GetSecs() after the flip."""
    for text, height, y in items:
        visual.TextStim(win, text=text, height=height, pos=(0, y), wrapWidth=params.WRAP_WIDTH).draw()
    win.flip()
    return GetSecs()


def wait_until(t, kb, params):
    """Wait until GetSecs() = t; the quit key raises QuitExperiment."""
    while True:
        left = t - GetSecs()
        if left <= 0:
            return
        if kb.getKeys(keyList=[params.QUIT_KEY], waitRelease=False):
            raise QuitExperiment
        WaitSecs(min(left, params.QUIT_POLL_S))


def wait_continue(kb, params):
    """Wait for a continue key; the quit key raises QuitExperiment."""
    keys = kb.waitKeys(keyList=params.CONTINUE_KEYS + [params.QUIT_KEY], waitRelease=True)
    if keys and keys[0].name == params.QUIT_KEY:
        raise QuitExperiment


def show_and_wait(text, win, kb, params):
    print(text)
    display_text(text, win, params)
    wait_continue(kb, params)


def fill(text, values, **extra):
    """Fill the {placeholders} of a text."""
    return text.format(**{**values, **extra})


#=====================
# OUTPUT
#=====================
def append_row(path, row, columns):
    """Append one trial to the tab-separated output file (header written with the first row)."""
    new = not os.path.exists(path)
    pd.DataFrame([row]).reindex(columns=columns).to_csv(path, sep='\t', mode='a', header=new, index=False)


#=====================
# ONE TRIAL
#=====================
def run_trial(aud, win, kb, params, y, screen_items, t_screen, onsets_s, rec_path):
    """trial screen at t_screen -> sound PRE_SOUND_S later (the screen stays) -> blank screen when the sound ends.
    Recorded from REC_PRE_S before the sound to REC_POST_S after it -> raw wav saved.

    The audio is scheduled right after the trial screen appears (on the planned time). Returns a dict with the
    timing, the recording info and the online tap check.
    """
    buffer = make_buffer(y, params)
    wait_until(t_screen, kb, params)
    t_screen_on = display_screen(screen_items, win, params)
    t_sound_planned = t_screen_on + params.PRE_SOUND_S
    t_rec_scheduled = t_sound_planned - params.REC_PRE_S
    start_play_and_record(aud, buffer, t_rec_scheduled)

    # the screen goes blank when the sound ends (the recording goes on for REC_POST_S)
    wait_until(t_sound_planned + len(y) / params.FS, kb, params)
    t_screen_off = display_text('', win, params)

    r = finish_play_and_record(aud, t_rec_scheduled + len(buffer) / params.FS, kb, params)
    rec = r['rec']
    t_sound_onset = r['t_rec_start'] + params.REC_PRE_S
    # where sample 0 of the stimulus file is in the recording (s): playback start - capture start + REC_PRE_S
    sound_start_in_rec = r['t_rec_start'] - r['t_capture_start'] + params.REC_PRE_S
    save_recording(rec_path, rec, params)

    check = tap_check(rec[:, params.MIC_CHANNEL], sound_start_in_rec, onsets_s, params)
    return dict(t_screen_on=t_screen_on, t_screen_off=t_screen_off, t_rec_scheduled=t_rec_scheduled,
                t_rec_start=r['t_rec_start'], t_capture_start=r['t_capture_start'],
                t_sound_onset=t_sound_onset, t_sound_offset=t_sound_onset + len(y) / params.FS,
                sound_start_in_rec_s=sound_start_in_rec,
                rec_file=os.path.basename(rec_path), rec_channels=','.join(params.REC_CHANNEL_NAMES),
                rec_n_samples=len(rec), rec_s=len(rec) / params.FS, overflow=r['overflow'],
                peak_per_channel=','.join(f'{p:.4f}' for p in np.abs(rec[:, params.REC_CHANNELS]).max(axis=0)),
                **check, rec_s_planned=len(buffer) / params.FS, t_done=GetSecs())


#=====================
# THE SESSION (shared by both groups)
#=====================
def run_session(params, setup, group):
    """Dialog -> startup checks -> start screen -> blocks (automatic trials) with break screens -> end.
    Every trial is saved immediately: one row in the data file and one raw wav.
    """
    check_params(params)

    # ---------- dialog
    info = dict(params.DIALOG_FIELDS)
    dlg = gui.DlgFromDict(dictionary=info, title=f'Tapping - {group} group ({setup})')
    if not dlg.OK:
        core.quit()
    pid = str(info['participant_id']).strip().upper()          # same rule as the stimulus builder
    start_block = int(info['start_block'])

    # ---------- design, sounds, screen texts
    folder = participant_dir(params, pid, group)
    load_stim_meta(params, folder, pid, group)
    design = load_design(params, folder, pid, group)
    onsets = load_events(params, folder, design)
    print(f'{pid} ({group}): {len(design)} trials - loading sounds from {folder} ...')
    sounds = load_participant_sounds(params, folder, design)
    screen_kind = check_pre_sound_kind(params, group)
    max_sound_s = max(len(y) for y in sounds.values()) / params.FS

    # ---------- output files (new for every session)
    os.makedirs(params.DATA_DIR, exist_ok=True)
    session_time = get_datetime_string()
    fmt = dict(pid=pid, group=group, time=session_time)
    out_path = os.path.join(params.DATA_DIR, params.OUTPUT_FILE_PATTERN.format(**fmt))
    rec_dir = os.path.join(params.DATA_DIR, params.REC_DIR_PATTERN.format(**fmt))
    if os.path.exists(out_path) or os.path.exists(rec_dir):
        raise FileExistsError(f'{out_path} or {rec_dir} exists - not overwriting')
    os.makedirs(rec_dir)
    print('Output:', out_path, '| recordings:', rec_dir)
    save_stim_snapshot(params, folder, design, sounds, pid, group, setup, session_time)

    trial_s = max_sound_s + params.REC_POST_S + params.SAVE_MARGIN_S + params.ITI_S + params.PRE_SOUND_S
    n_run = len(design) if params.MAX_TRIALS_PER_BLOCK is None else \
        params.N_BLOCKS * min(params.MAX_TRIALS_PER_BLOCK, params.TRIALS_PER_BLOCK)
    print(f'about {trial_s:.1f} s per trial -> {n_run} trials = {n_run * trial_s / 60:.0f} min without breaks')

    aud = win = None
    try:
        # ---------- audio, window, keyboard
        aud = open_audio(params, max_sound_s)
        win = visual.Window(params.WIN_SIZE, fullscr=params.FULLSCREEN, monitor=params.MONITOR,
                            units=params.UNITS, screen=params.SCREEN)
        win.mouseVisible = False
        kb = keyboard.Keyboard()
        T = params.TEXTS[group]
        values = dict(n_blocks=params.N_BLOCKS, n_trials=params.TRIALS_PER_BLOCK,
                      block_min=int(np.ceil(params.TRIALS_PER_BLOCK * trial_s / 60)))

        # ---------- instructions (only at the start of a session; a resumed session shows only 'start')
        if start_block == 1:
            for key in params.INTRO_SCREENS:
                show_and_wait(fill(T[key], values), win, kb, params)
        show_and_wait(fill(T['start'], values), win, kb, params)

        # ---------- blocks
        for block in range(start_block, params.N_BLOCKS + 1):
            trials = design[design['block'] == block]
            if params.MAX_TRIALS_PER_BLOCK is not None:
                trials = trials.head(params.MAX_TRIALS_PER_BLOCK)
            display_text('', win, params)                                    # blank
            t_next = GetSecs() + params.FIRST_TRIAL_DELAY_S
            checks = []                                                      # tap checks of this block

            for i, (_, row) in enumerate(trials.iterrows(), start=1):
                items = trial_screen(screen_kind, i, row, params)
                text = ' | '.join(t for t, _, _ in items)                   # what was on screen, for the data file
                rec_path = os.path.join(rec_dir, params.REC_FILE_PATTERN.format(trial=int(row['trial'])))
                r = run_trial(aud, win, kb, params, sounds[row['trial']], items, t_next,
                              onsets[row['trial']], rec_path)
                # fixed pacing: the next trial screen comes SAVE_MARGIN_S + ITI_S after the recording ended,
                # whatever saving took (if saving took longer, the trial starts late: logged in late_start_next_s)
                t_next = r['t_rec_start'] + r['rec_s_planned'] + params.SAVE_MARGIN_S + params.ITI_S
                late = max(0.0, r.pop('t_done') - t_next)

                out = {c: row[c] for c in params.DESIGN_COLUMNS}
                r.pop('rec_s_planned')
                out.update(session_time=session_time, setup=setup, params_file=params.__name__, trial_in_block=i,
                           pre_sound_screen=screen_kind, screen_text=text, late_start_next_s=round(late, 4), **r)
                append_row(out_path, out, params.OUTPUT_COLUMNS)
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
                print(f"block {block} trial {i}/{len(trials)} (#{row['trial']}) {row['sentence_id']} "
                      f"{row['jitter_pattern']:5s} | sounds tapped {r['online_n_sounds_tapped']}/{params.N_EVENTS}, "
                      f"double {r['online_n_sounds_double']}, ITI median {r['online_iti_median_ms']} ms "
                      f"| peaks {r['peak_per_channel']}{warn}")

            if block < params.N_BLOCKS:
                show_and_wait(fill(T['break'], values, block=block, **block_feedback(checks, T, params)),
                              win, kb, params)
            else:
                show_and_wait(fill(T['end'], values), win, kb, params)

    except QuitExperiment:
        print('Quit key pressed - data saved up to the last completed trial:', out_path)

    finally:                       # always close audio and window; an error still shows its full traceback
        close_audio(aud)
        if win is not None:
            win.close()
    core.quit()                    # only reached after a normal end or the quit key