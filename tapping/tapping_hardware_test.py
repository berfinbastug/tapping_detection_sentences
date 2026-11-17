# %% TAPPING HARDWARE TEST: TASCAM US-2x2HR + desk microphone (PsychoPy + Psychtoolbox / PsychPortAudio)
"""
A small check that the setup records your taps, before the real experiment.

Per test trial:
    plays a sound through the TASCAM (both outputs) and, AT THE SAME TIME, records its inputs
    (full-duplex PsychPortAudio stream, mode 3: playback and capture start on the same audio clock)
    -> saves the recording (wav), finds the taps in the microphone channel, and writes a figure + table

What you learn
    1 does the mic pick up the taps (and nothing else)?  -> tap count, figure
    2 are the taps where you expect them?                -> asynchrony of every tap to the nearest sound onset
    3 (with a LOOPBACK cable) the real audio latency     -> where the sounds actually come out vs where they are
                                                            in the file; taps are then measured against the
                                                            sounds as recorded = no latency guesswork

Wiring (TASCAM US-2x2HR, 2 inputs / 2 outputs)
    input 2 (MIC_CHANNEL = 1)       the desk microphone (+48V on only if it is a condenser mic that needs it)
    input 1 (LOOPBACK_CHANNEL = 0)  optional: a cable from one output (e.g. LINE OUT L) back into input 1.
                                    No cable -> set LOOPBACK_CHANNEL = None.
    headphones on the TASCAM's headphone output

Stimulus: STIMULUS = 'clicks'  -> N_CLICKS short clicks on a 450 ms grid (simplest)
          STIMULUS = 'trial'   -> one real trial from build_tapping_stimuli.py (npy + events_stim.csv)

Outputs: OUT_DIR/<time>_trialNN_rec.wav (all inputs, float32), _taps.csv, _plot.png, and a summary printed at the end.
"""
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use('Agg')                       # figures go to png files; no window that could block PsychoPy
import matplotlib.pyplot as plt

from psychtoolbox import audio, PsychPortAudio, GetSecs, WaitSecs
from psychopy import core, visual
from psychopy.hardware import keyboard

# %% SETTINGS
DEVICE_NAME   = 'US-2x2'        # part of the TASCAM's name in the device list (printed when you run this)
DEVICE_INDEX  = None            # if several devices match, put the right DeviceIndex here (see printout)
# where the sound plays:                                                                          # NEW
#   None                    -> through the TASCAM (headphones / line out): ONE full-duplex stream, same clock for
#                              sound and recording, loopback possible. Use this for the real experiment.
#   'MacBook Pro Speakers'  -> through that device, recorded by the TASCAM: TWO streams with separate clocks; the
#                              sound-to-recording alignment then relies on timestamps, and no loopback is possible
OUTPUT_DEVICE_NAME  = 'Earpods'                                                      # NEW
OUTPUT_DEVICE_INDEX = None                                                                        # NEW
FS            = 44100           # same as the stimuli
LATENCY_CLASS = 3               # 3 = aggressive low latency (as in your old code); try 1 if the stream fails to open
N_OUT, N_IN   = 2, 2            # TASCAM: 2 outputs, 2 inputs
MIC_CHANNEL      = 1            # 0-based input index: 1 = input 2
LOOPBACK_CHANNEL = None         # 0-based input index of a loopback cable (e.g. 0 = input 1), or None

STIMULUS   = 'clicks'           # 'clicks' or 'trial'
N_CLICKS, IOI_MS, CLICK_MS, CLICK_HZ, CLICK_LEVEL = 30, 450, 10, 1000, 0.3
STIM_DIR   = 'stimuli_tapping/P01_speech'   # only for STIMULUS = 'trial'
TRIAL      = 1                              # only for STIMULUS = 'trial': trial number in design_stim.csv
LEAD_S, POST_S = 0.5, 1.5       # silence before the sound and after it (keeps recording for late taps)
N_TEST_TRIALS  = 3

# tap detection (microphone channel)
TAP_HIGHPASS_HZ   = 1000        # CHANGED: the mic signal is mostly low rumble (< 50 Hz); taps are clicks -> high-pass
TAP_THRESHOLD_MAD = 20          # CHANGED: tap = envelope above the noise floor + this many MADs of the noise
TAP_REFRACTORY_MS = 150         # one tap = no new tap for this long (taps come every ~450 ms)
TAP_SMOOTH_MS     = 1
# sound onsets in the loopback channel: same rule as the stimuli (-40 dB re each sound's own peak)
ONSET_DB, SEG_DB, ENV_WIN_MS = -40, -30, 5

OUT_DIR = 'tapping_test'


# %% FUNCTIONS
def find_device(devices, name, index, n_in, n_out, role):                                        # CHANGED
    """The DeviceIndex of the one device matching `name` (or `index`) with at least n_in inputs / n_out outputs."""
    if index is not None:
        match = [d for d in devices if d['DeviceIndex'] == index]
    else:
        match = [d for d in devices if name.lower() in d['DeviceName'].lower()
                 and d['NrInputChannels'] >= n_in and d['NrOutputChannels'] >= n_out]
    if len(match) != 1:
        raise RuntimeError(f'{len(match)} devices match {name!r} with >= {n_in} inputs and >= {n_out} outputs '
                           f'({role}): set the DeviceIndex in the settings (see the list above)')
    print(f"-> {role}: DeviceIndex {int(match[0]['DeviceIndex'])}, {match[0]['DeviceName']}")
    return int(match[0]['DeviceIndex'])        # the device's own index (NOT its position in the list)


def make_stimulus():
    """Mono stimulus (float) and the onsets (s, from the start of the sound) of its events."""
    if STIMULUS == 'clicks':
        n = int(round(CLICK_MS / 1000 * FS))
        r = int(round(0.002 * FS))
        click = np.sin(2 * np.pi * CLICK_HZ * np.arange(n) / FS)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r))
        click[:r] *= ramp
        click[-r:] *= ramp[::-1]
        onsets = np.arange(N_CLICKS) * IOI_MS / 1000
        y = np.zeros(int(round(onsets[-1] * FS)) + n)
        for t in onsets:
            i = int(round(t * FS))
            y[i:i + n] += CLICK_LEVEL * click
        onsets = detect_sound_onsets(y, N_CLICKS)        # same onset rule as for the loopback channel
        return y, onsets, f'{N_CLICKS} clicks, IOI {IOI_MS} ms'
    design = pd.read_csv(os.path.join(STIM_DIR, 'design_stim.csv')).set_index('trial')
    events = pd.read_csv(os.path.join(STIM_DIR, 'events_stim.csv'))
    row = design.loc[TRIAL]
    stims = np.load(os.path.join(STIM_DIR, 'tapping_stimuli.npy'), mmap_mode='r')
    y = np.array(stims[row['stim_row'], :row['n_samples']], dtype=float)
    onsets = events.loc[events.trial == TRIAL].sort_values('event')['actual_onset_ms'].to_numpy() / 1000
    return y, onsets, f"trial {TRIAL} ({row['sentence_id']}, {row['jitter_pattern']})"


def envelope(x, win_ms):
    n = max(1, int(round(win_ms / 1000 * FS)))
    c = np.r_[0.0, np.cumsum(np.asarray(x, float) ** 2)]
    i = np.arange(len(x))
    lo, hi = np.clip(i - n // 2, 0, len(x)), np.clip(i + (n - 1) // 2 + 1, 0, len(x))
    return np.sqrt(np.maximum(c[hi] - c[lo], 0) / n)


def detect_taps(x):                                                                              # CHANGED
    """Tap onsets (s from the start of the recording).
    1 high-pass at TAP_HIGHPASS_HZ: removes rumble / drift, keeps the sharp click of a tap
    2 1 ms RMS envelope; threshold = noise floor (median) + TAP_THRESHOLD_MAD x MAD (robust to the taps themselves)
    3 first crossing = tap; nothing counts as a new tap for TAP_REFRACTORY_MS.
    Returns (taps, threshold, envelope)."""
    sos = butter(4, TAP_HIGHPASS_HZ, 'highpass', fs=FS, output='sos')
    env = envelope(sosfiltfilt(sos, np.asarray(x, float)), TAP_SMOOTH_MS)
    med = np.median(env)
    thr = med + TAP_THRESHOLD_MAD * np.median(np.abs(env - med))
    n_ref = TAP_REFRACTORY_MS / 1000 * FS
    taps, last = [], -np.inf
    for i in np.flatnonzero(env > thr):
        if i - last > n_ref:
            taps.append(i)
            last = i
    return np.array(taps) / FS, thr, env


def detect_sound_onsets(x, n_expected):
    """Onsets (s) of the sounds in the loopback channel, found without assuming where they are:
    1 separate sounds = envelope above (overall peak + SEG_DB)
    2 a word with a silent stretch inside gives 2+ pieces: merge the neighbouring pair with the shortest combined
      span until n_expected sounds remain (never across more than 600 ms = two different sounds)
    3 walk back to ONSET_DB of each sound's own peak (the rule used to place the sounds in the stimuli)."""
    env = envelope(x, ENV_WIN_MS)
    if env.max() <= 0:
        return np.array([])
    above = env > env.max() * 10 ** (SEG_DB / 20)
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
    starts, ends = list(edges[::2]), list(edges[1::2])
    while len(starts) > n_expected:
        spans = [ends[k + 1] - starts[k] for k in range(len(starts) - 1)]
        k = int(np.argmin(spans))
        if spans[k] / FS > 0.6:
            break
        ends[k] = ends.pop(k + 1)
        starts.pop(k + 1)
    out = []
    for s, e in zip(starts, ends):
        thr = env[s:e].max() * 10 ** (ONSET_DB / 20)
        i = s
        while i > 0 and env[i - 1] > thr:
            i -= 1
        out.append(i / FS)
    return np.array(out)


def nearest(taps, onsets):
    """For every tap: the nearest onset and the asynchrony tap - onset (s)."""
    if len(taps) == 0 or len(onsets) == 0:
        return np.full(len(taps), np.nan), np.full(len(taps), np.nan)
    k = np.abs(taps[:, None] - onsets[None, :]).argmin(axis=1)
    return onsets[k], taps - onsets[k]


def analyse(rec, file_onsets, sound_start_in_rec):
    """rec: (n, N_IN) recording. file_onsets: onsets in the stimulus file (s).
    sound_start_in_rec: where the first sample of the sound SHOULD be in the recording according to the
    PsychPortAudio timestamps (s). Returns (taps table, info dict)."""
    taps, thr, env = detect_taps(rec[:, MIC_CHANNEL])
    expected = sound_start_in_rec + file_onsets               # from timestamps: includes no measured latency
    info = dict(n_taps=len(taps), tap_threshold=round(float(thr), 5))
    ref, ref_name = expected, 'timestamps'
    if LOOPBACK_CHANNEL is not None:
        measured = detect_sound_onsets(rec[:, LOOPBACK_CHANNEL], len(file_onsets))
        info['loopback_onsets_found'] = len(measured)
        if len(measured) == len(file_onsets):
            lat = measured - expected
            info.update(loopback_minus_timestamps_ms=round(float(np.median(lat) * 1000), 3),
                        loopback_minus_timestamps_sd=round(float(np.std(lat) * 1000), 3))
            ref, ref_name = measured, 'loopback'
        else:
            print(f'  ! loopback: found {len(measured)} sound onsets, expected {len(file_onsets)} '
                  f'(cable connected? LOOPBACK_CHANNEL right?) -> asynchronies from timestamps')
    onset, asyn = nearest(taps, ref)
    # CHANGED: statistics only for taps DURING the sound (half an interval before the first onset to half an
    # interval after the last); taps before / after it are kept in the table but flagged
    half = np.median(np.diff(ref)) / 2 if len(ref) > 1 else 0.25
    during = (taps > ref[0] - half) & (taps < ref[-1] + half) if len(taps) else np.zeros(0, bool)
    a, td = asyn[during], taps[during]
    info.update(reference=ref_name, taps_during_sound=int(during.sum()), taps_outside=int((~during).sum()),
                asynchrony_ms_mean=round(float(np.mean(a) * 1000), 1) if len(a) else None,
                asynchrony_ms_sd=round(float(np.std(a) * 1000), 1) if len(a) else None,
                iti_ms_median=round(float(np.median(np.diff(td)) * 1000), 1) if len(td) > 1 else None)
    table = pd.DataFrame(dict(tap=np.arange(1, len(taps) + 1), tap_s=np.round(taps, 5), during_sound=during,
                              nearest_onset_s=np.round(onset, 5), asynchrony_ms=np.round(asyn * 1000, 2)))
    return table, info, ref, thr, env


def plot(rec, title, path):                                                                       # CHANGED
    """The raw recording, nothing drawn on top: one panel per input."""
    t = np.arange(len(rec)) / FS
    fig, axes = plt.subplots(rec.shape[1], 1, figsize=(14, 2.2 * rec.shape[1]), squeeze=False, sharex=True)
    for c in range(rec.shape[1]):
        ax = axes[c, 0]
        ax.plot(t, rec[:, c], color='k', lw=0.3)
        name = 'microphone' if c == MIC_CHANNEL else 'loopback' if c == LOOPBACK_CHANNEL else 'not used'
        ax.set(xlim=(0, t[-1]), ylabel=f'input {c + 1}\n({name})')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0, 0].set_title(f'{title} | raw recording', loc='left')
    axes[-1, 0].set_xlabel('time in recording (s)')
    plt.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# %% MAIN
os.makedirs(OUT_DIR, exist_ok=True)
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
y, file_onsets, stim_label = make_stimulus()
if np.abs(y).max() >= 1:
    raise ValueError('stimulus clips')
buffer = np.r_[np.zeros(int(LEAD_S * FS)), y, np.zeros(int(POST_S * FS))]
buffer = np.tile(buffer[:, None], (1, N_OUT)).astype(np.float32)       # same sound on both outputs
print(f'Stimulus: {stim_label}, {len(y) / FS:.1f} s (+ {LEAD_S} s before, {POST_S} s after)')

devices = audio.get_devices()                                                                    # CHANGED
print('\nAudio devices:')
for d in devices:
    print(f"  DeviceIndex {int(d['DeviceIndex']):>3} | {d['HostAudioAPIName']:<12} | in {int(d['NrInputChannels']):>2} "
          f"out {int(d['NrOutputChannels']):>2} | {d['DeviceName']}")
SEPARATE = OUTPUT_DEVICE_NAME is not None or OUTPUT_DEVICE_INDEX is not None
if SEPARATE and LOOPBACK_CHANNEL is not None:
    print('! loopback needs the sound to come out of the TASCAM: LOOPBACK_CHANNEL set to None')
    LOOPBACK_CHANNEL = None
device_index = find_device(devices, DEVICE_NAME, DEVICE_INDEX, N_IN, 0 if SEPARATE else N_OUT,
                           'recording' if SEPARATE else 'sound + recording')
if not SEPARATE:
    # one full-duplex stream: playback and capture start together on the TASCAM's clock
    play = cap = audio.Stream(freq=FS, device_id=device_index, mode=3, latency_class=LATENCY_CLASS,
                              channels=[N_OUT, N_IN])
else:
    out_index = find_device(devices, OUTPUT_DEVICE_NAME or '', OUTPUT_DEVICE_INDEX, 0, N_OUT, 'sound')
    play = audio.Stream(freq=FS, device_id=out_index, mode=1, latency_class=LATENCY_CLASS, channels=N_OUT)
    cap = audio.Stream(freq=FS, device_id=device_index, mode=2, latency_class=LATENCY_CLASS, channels=N_IN)
print()
cap.get_audio_data(secs_allocate=len(buffer) / FS + 30)       # capture buffer (must be allocated before start)

# warm-up: play 0.1 s of silence once (and record), then empty the capture buffer
PsychPortAudio('FillBuffer', play.handle, np.zeros((int(0.1 * FS), N_OUT), np.float32))
if SEPARATE:
    PsychPortAudio('Start', cap.handle, 0, 0, 1)
PsychPortAudio('Start', play.handle, 1, 0, 1)
PsychPortAudio('Stop', play.handle, 1)
if SEPARATE:
    PsychPortAudio('Stop', cap.handle)
cap.get_audio_data()

win = visual.Window([1000, 600], fullscr=False, units='height', color='grey')
kb = keyboard.Keyboard()
msg = visual.TextStim(win, height=0.04, wrapWidth=1.4)


def show(text):
    msg.text = text
    msg.draw()
    win.flip()


summary = []
for k in range(1, N_TEST_TRIALS + 1):
    show(f'Test trial {k} of {N_TEST_TRIALS}\n\n{stim_label}\n\nTap along with every sound.\n\n'
         f'Press SPACE to start (ESC to quit).')
    if kb.waitKeys(keyList=['space', 'escape'], waitRelease=True)[0].name == 'escape':
        break
    show('Tap along ...')

    cap.get_audio_data()                                      # empty leftovers before this trial    # CHANGED
    play.fill_buffer(buffer)
    if SEPARATE:
        PsychPortAudio('Start', cap.handle, 0, 0, 1)          # recording runs until stopped below
    t_start = play.start(when=GetSecs() + 0.3, wait_for_start=1)     # time the first sample of the BUFFER plays
    PsychPortAudio('Stop', play.handle, 1)                    # waits until the whole buffer has played
    if SEPARATE:
        PsychPortAudio('Stop', cap.handle)
    rec, absrecpos, overflow, cap_start = cap.get_audio_data()
    rec = np.asarray(rec, dtype=np.float64)
    if rec.ndim == 1:
        rec = rec[:, None]
    if overflow:
        print(f'  ! capture buffer overflow in trial {k}: recording incomplete')
    peaks = np.abs(rec).max(axis=0)                                                          # NEW
    print('  input peaks: ' + ', '.join(f'input {c + 1} = {p:.5f}' for c, p in enumerate(peaks)))
    if not np.any(peaks):
        print('  ! EVERY input is exactly 0 (digital silence, not even noise). On macOS this almost always means\n'
              '    the app running Python has no microphone permission: System Settings > Privacy & Security >\n'
              '    Microphone > switch on Terminal (or VS Code / PsychoPy), then quit and reopen it.')

    # where the sound should be in the recording according to the timestamps:
    # recording sample 0 = cap_start, buffer sample 0 = t_start, sound starts LEAD_S into the buffer
    sound_start_in_rec = (t_start - cap_start) + LEAD_S
    table, info, ref, thr, env = analyse(rec, file_onsets, sound_start_in_rec)
    info.update(trial=k, recording_s=round(len(rec) / FS, 2), overflow=bool(overflow),
                playback_minus_capture_start_ms=round((t_start - cap_start) * 1000, 3),
                peak_mic=round(float(np.abs(rec[:, MIC_CHANNEL]).max()), 4))
    summary.append(info)

    base = os.path.join(OUT_DIR, f'{stamp}_trial{k:02d}')
    sf.write(base + '_rec.wav', rec.astype(np.float32), FS, subtype='FLOAT')
    table.to_csv(base + '_taps.csv', index=False)
    plot(rec, f'test trial {k}: {stim_label}', base + '_plot.png')

    print(f"trial {k}: {info['taps_during_sound']} taps during the sound (+{info['taps_outside']} outside) | mic peak {info['peak_mic']} | asynchrony "
          f"{info['asynchrony_ms_mean']} ± {info['asynchrony_ms_sd']} ms vs {info['reference']}"
          + (f" | sound came out {info.get('loopback_minus_timestamps_ms')} ms (SD {info.get('loopback_minus_timestamps_sd')}) later than the timestamps say"
             if 'loopback_minus_timestamps_ms' in info else ''))
    if info['n_taps'] == 0:
        show('No taps found in the microphone channel.\n\nCheck: mic in input '
             f'{MIC_CHANNEL + 1}? input gain? +48V if needed?\n\nPress SPACE to continue.')
        kb.waitKeys(keyList=['space'], waitRelease=True)

play.close()                                                                     # CHANGED
if SEPARATE:
    cap.close()
show('Done.')
WaitSecs(0.5)
win.close()

with open(os.path.join(OUT_DIR, f'{stamp}_summary.json'), 'w') as fh:
    json.dump(dict(device_index=int(device_index), output_device=OUTPUT_DEVICE_NAME, fs=FS, latency_class=LATENCY_CLASS, stimulus=stim_label,
                   mic_channel=MIC_CHANNEL, loopback_channel=LOOPBACK_CHANNEL, trials=summary), fh, indent=2)
print('\n' + pd.DataFrame(summary).to_string(index=False))
print(f'\nRecordings, tap tables and figures in {OUT_DIR}/')
core.quit()