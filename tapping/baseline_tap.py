"""
SPONTANEOUS TAPPING: record a participant tapping at their own pace for 1 minute (no sound is played).

Flow: dialog (participant ID) -> screens 1-3 (welcome, task, how it works) -> screen 4: cross (+) while recording
      for REC_S seconds
      -> screen 5 (done) -> the RAW recording is saved as a wav + a figure of it.

Saved in DATA_DIR (nothing is ever overwritten):
    spontaneous_<pid>_<time>.wav    the microphone channel exactly as recorded (32-bit float, no filtering)
    spontaneous_<pid>_<time>.png    figure of the raw recording (whole minute + the first 10 s)
    spontaneous_<pid>_<time>.json   settings and timing (device, channel, start time, duration, overflow)

Escape stops the recording early; what was recorded up to then is still saved.
"""
import os
import json
from datetime import datetime

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')                       # the figure goes to a png file; no window that could block PsychoPy
import matplotlib.pyplot as plt
from psychtoolbox import audio, PsychPortAudio, GetSecs, WaitSecs
from psychopy import core, gui, visual
from psychopy.hardware import keyboard

#=====================
# SETTINGS
#=====================
SETUP = 'laptop'                            # 'lab' or 'laptop'

SETUPS = {
    'laptop': dict(MAIN_DIR='/Users/bastugb/Desktop/tapping_detection_sentences', DATA_SUBDIR='tapping/data_test',
                   DEVICE_NAME='US-2x2HR', DEVICE_INDEX=None, LATENCY_CLASS=3,
                   WIN_SIZE=[1000, 650], FULLSCREEN=False, SCREEN=0, TEXT_HEIGHT=0.55, WRAP_WIDTH=24.0),
    'lab':    dict(MAIN_DIR='/Users/bastugb/Desktop/tapping_detection_sentences', DATA_SUBDIR='tapping/data',   # CHECK path
                   DEVICE_NAME='US-4x4HR', DEVICE_INDEX=None, LATENCY_CLASS=1,                        # CHECK device
                   WIN_SIZE=[1920, 1080], FULLSCREEN=True, SCREEN=1, TEXT_HEIGHT=1.0, WRAP_WIDTH=40.0),
}
S = SETUPS[SETUP]
DATA_DIR = os.path.join(S['MAIN_DIR'], S['DATA_SUBDIR'])

REC_S       = 60.0                          # recording duration (s)
FS          = 44100
N_IN        = 2                             # input channels opened
MIC_CHANNEL = 1                             # 0-based: input 2 = desk microphone (same as the tapping experiment)
REC_SUBTYPE = 'FLOAT'                       # 32-bit float wav, exactly as recorded

CONTINUE_KEYS = ['1', '2', '3', '4']
QUIT_KEY      = 'escape'
QUIT_POLL_S   = 0.02
MONITOR, UNITS = 'testMonitor', 'cm'

# TEXTS (German): screens 1-3 before the recording, the cross during it, screen 5 after it
_CONTINUE = 'Drücken Sie eine beliebige Taste, um fortzufahren.'
INTRO_TEXTS = [
    # 1 welcome
    ('Willkommen zum Experiment!\n\n'
     'Wir beginnen mit einer kurzen Tippaufgabe. In diesem Teil hören Sie keine Klänge.\n\n' + _CONTINUE),
    # 2 task
    ('Auf dem Tisch sehen Sie ein kleines Mikrofon. Tippen Sie mit Ihrem Zeigefinger neben dem Mikrofon auf '
     'den Tisch.\n\n'
     'Tippen Sie in einem Tempo, das sich für Sie angenehm und natürlich anfühlt. Versuchen Sie, die ganze Zeit '
     'dasselbe Tempo beizubehalten.\n\n' + _CONTINUE),
    # 3 how it works
    ('Wenn ein Kreuz (+) auf dem Bildschirm erscheint, beginnen Sie zu tippen. Tippen Sie weiter, bis das Kreuz '
     'verschwindet. Das dauert etwa 60 Sekunden.\n\n'
     'Drücken Sie eine beliebige Taste, wenn Sie bereit sind.'),
]
TEXT_REC = '+'                              # 4: on screen while recording (disappears when the recording ends)
TEXT_END = 'Vielen Dank! Dieser Teil ist abgeschlossen.\n\n' + _CONTINUE      # 5
CROSS_SCALE = 2.0                           # the cross is this many times TEXT_HEIGHT


#=====================
# FUNCTIONS
#=====================
def find_device(name, index, n_in):
    """DeviceIndex of the one device matching `name` (or `index`) with at least n_in inputs."""
    devices = audio.get_devices()
    print('\nAudio devices:')
    for d in devices:
        print(f"  DeviceIndex {int(d['DeviceIndex']):>3} | in {int(d['NrInputChannels']):>2} "
              f"out {int(d['NrOutputChannels']):>2} | {d['DeviceName']}")
    if index is not None:
        match = [d for d in devices if d['DeviceIndex'] == index]
    else:
        match = [d for d in devices if name.lower() in d['DeviceName'].lower() and d['NrInputChannels'] >= n_in]
    if len(match) != 1:
        raise RuntimeError(f'{len(match)} devices match {name!r} with >= {n_in} inputs: set DEVICE_INDEX')
    print(f"-> recording: DeviceIndex {int(match[0]['DeviceIndex'])}, {match[0]['DeviceName']}\n")
    return int(match[0]['DeviceIndex'])


def display_text(win, text, scale=1.0):
    if text:
        visual.TextStim(win, text=text, height=S['TEXT_HEIGHT'] * scale, wrapWidth=S['WRAP_WIDTH']).draw()
    win.flip()
    return GetSecs()


def wait_key(kb):
    """Wait for a continue key (or the quit key). Returns True if the quit key was pressed."""
    keys = kb.waitKeys(keyList=CONTINUE_KEYS + [QUIT_KEY], waitRelease=True)
    return bool(keys) and keys[0].name == QUIT_KEY


def plot_recording(x, path, title):
    """The raw recording, nothing drawn on top: the whole recording and the first 10 s."""
    t = np.arange(len(x)) / FS
    fig, axes = plt.subplots(2, 1, figsize=(14, 5))
    for ax, t_max, label in ((axes[0], t[-1] if len(t) else 1, 'whole recording'), (axes[1], 10, 'first 10 s')):
        sel = t <= t_max
        ax.plot(t[sel], x[sel], color='k', lw=0.3)
        ax.set(xlim=(0, t_max), ylabel='amplitude', title=label)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_title(f'{title} | whole recording', loc='left')
    axes[-1].set_xlabel('time in recording (s)')
    plt.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


#=====================
# MAIN
#=====================
info = {'participant_id': 'P01'}
dlg = gui.DlgFromDict(dictionary=info, title=f'Spontaneous tapping ({SETUP})')
if not dlg.OK:
    core.quit()
pid = str(info['participant_id']).strip().upper()

os.makedirs(DATA_DIR, exist_ok=True)
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
base = os.path.join(DATA_DIR, f'spontaneous_{pid}_{stamp}')
for ext in ('.wav', '.png', '.json'):
    if os.path.exists(base + ext):
        raise FileExistsError(f'{base + ext} exists - not overwriting')

# audio: capture-only stream on the interface
device = find_device(S['DEVICE_NAME'], S['DEVICE_INDEX'], N_IN)
cap = audio.Stream(freq=FS, device_id=device, mode=2, latency_class=S['LATENCY_CLASS'], channels=N_IN)
cap.get_audio_data(secs_allocate=REC_S + 10)          # capture buffer (must be allocated before start)

win = visual.Window(S['WIN_SIZE'], fullscr=S['FULLSCREEN'], monitor=MONITOR, units=UNITS, screen=S['SCREEN'])
win.mouseVisible = False
kb = keyboard.Keyboard()

rec, overflow, t_start, t_stop, stopped_early = np.zeros((0, N_IN), np.float32), False, None, None, False
t_screen = t_capture_start = None
try:
    for text in INTRO_TEXTS:                                 # screens 1-3
        display_text(win, text)
        if wait_key(kb):
            raise KeyboardInterrupt
    t_screen = display_text(win, TEXT_REC, scale=CROSS_SCALE)   # screen 4: the cross

    # record REC_S seconds; the quit key stops early
    t_start = PsychPortAudio('Start', cap.handle, 0, 0, 1)   # runs until stopped; returns when capture started
    kb.clearEvents()
    while GetSecs() < t_start + REC_S + 0.05:                # + 50 ms: the file is never shorter than REC_S
        if kb.getKeys(keyList=[QUIT_KEY], waitRelease=False):
            stopped_early = True
            break
        WaitSecs(QUIT_POLL_S)
    PsychPortAudio('Stop', cap.handle)
    t_stop = GetSecs()
    rec, _absrecpos, overflow, t_capture_start = cap.get_audio_data()
    rec = np.asarray(rec, dtype=np.float32)
    if rec.ndim == 1:
        rec = rec[:, None]

    if not stopped_early:
        display_text(win, TEXT_END)
        wait_key(kb)
except KeyboardInterrupt:
    print('Quit before the recording started - nothing saved.')
finally:
    cap.close()
    win.close()

# save + plot (after the window is closed)
if t_start is not None:
    mic = rec[:, MIC_CHANNEL]
    sf.write(base + '.wav', mic, FS, subtype=REC_SUBTYPE)
    plot_recording(mic, base + '.png', f'spontaneous tapping {pid}')
    meta = dict(participant=pid, setup=SETUP, time=stamp, device_index=device, device_name=S['DEVICE_NAME'],
                fs=FS, n_in=N_IN, mic_channel=MIC_CHANNEL, rec_subtype=REC_SUBTYPE, planned_s=REC_S,
                recorded_s=round(len(mic) / FS, 4), n_samples=int(len(mic)), stopped_early=stopped_early,
                overflow=bool(overflow), t_screen_on=t_screen, t_capture_start=t_capture_start, t_stop=t_stop,
                peak=round(float(np.abs(mic).max()) if len(mic) else 0.0, 5))
    with open(base + '.json', 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nSaved {base}.wav ({meta['recorded_s']} s{', STOPPED EARLY' if stopped_early else ''}), .png, .json")
    print(f"mic peak {meta['peak']}" + ('  ! capture overflow' if overflow else ''))
    if len(mic) and not np.any(mic):
        print('! the recording is exactly 0 everywhere: on macOS this usually means the app running Python has no '
              'microphone permission (System Settings > Privacy & Security > Microphone).')
core.quit()