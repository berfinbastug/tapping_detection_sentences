"""Helper functions of the staircase pilot (2IFC).

PART 1 (this file now): the staircases (PsychoPy's StairHandler), and building / checking the sound of one trial.
    build_staircase_stimuli.py uses them to check every trial at every possible size BEFORE the session;
    the run script uses the same functions to build each trial once its size is known.
    So the checked sound and the played sound always come from exactly the same code.
PART 2: PsychoPy / Psychtoolbox: screen, keys, audio stream, one trial, practice and the session.
    run_staircase_speech.py only chooses the setup and the group and then calls run_session().

PsychoPy is only imported when a staircase is made (the stimulus builder does not need it), and repo/ is
loaded file by file (repo/__init__.py is NOT run), so no parselmouth / gTTS is needed on the lab computer.
"""
import os
import glob
import json
import importlib
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd

try:                                   # PART 2 only; the stimulus builder runs without these
    from psychtoolbox import audio, GetSecs, WaitSecs
    from psychopy import core, gui, visual
    from psychopy.hardware import keyboard
except ImportError:
    audio = GetSecs = WaitSecs = core = gui = visual = keyboard = None


class QuitExperiment(Exception):
    """Raised when the quit key is pressed."""


#=====================
# SESSION
#=====================
def get_datetime_string():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def load_params(setup):
    """'lab' -> staircase_params, 'laptop' -> staircase_params_laptop."""
    modules = {'lab': 'staircase_params', 'laptop': 'staircase_params_laptop'}
    if setup not in modules:
        raise ValueError(f'unknown SETUP {setup!r}: use {list(modules)}')
    return importlib.import_module(modules[setup])


_REPO = {}

def repo_module(params, name):
    """repo/<name>.py ('sequence' or 'audio') loaded directly from its file."""
    if name not in _REPO:
        path = os.path.join(params.REPO_DIR, 'repo', f'{name}.py')
        spec = importlib.util.spec_from_file_location(f'repo_{name}', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _REPO[name] = mod
    return _REPO[name]


#=====================
# STAIRCASE (PsychoPy's StairHandler)
#=====================
class Staircase:
    """One staircase = one PsychoPy StairHandler (psychopy.data.StairHandler), set up so that its sizes stay on
    params.SIZE_LEVELS_MS (the grid every trial was checked on by build_staircase_stimuli.py).

    StairHandler settings: startVal = START_MS, nDown / nUp, stepType 'log', applyInitialRule False
    (N_DOWN-down-N_UP-up from the first trial), minVal / maxVal = smallest / largest grid size.
    stepSizes = BIG_STEP grid steps for the first SWITCH_AFTER reversals, then SMALL_STEP grid steps
    (PsychoPy moves to the next step size at each reversal). One grid step = log10(GRID_FACTOR).

    The design table decides which staircase runs on each trial, so the 4 handlers are driven one by one
    (no MultiStairHandler). Use: shift = sc.shift_ms -> play the trial -> sc.update(correct).
    """

    def __init__(self, name, params):
        from psychopy.data import StairHandler                # imported here: the stimulus builder does not need PsychoPy
        self.name = name
        self.direction = name.split('_')[0]                    # 'delay' / 'advance'
        self.sign = params.DIRECTION_SIGN[self.direction]
        self.levels = np.asarray(params.SIZE_LEVELS_MS, dtype=float)
        if self.levels[0] != params.START_MS:
            raise ValueError('SIZE_LEVELS_MS must start at START_MS')
        self.n_discard = params.N_DISCARD
        g = np.log10(params.GRID_FACTOR)
        steps = [params.BIG_STEP * g] * params.SWITCH_AFTER + [params.SMALL_STEP * g]
        n_trials = params.N_BLOCKS * params.TRIALS_PER_BLOCK // len(params.STAIRCASES)
        self.handler = StairHandler(startVal=params.START_MS, nReversals=None, stepSizes=steps, nTrials=n_trials,
                                    nUp=params.N_UP, nDown=params.N_DOWN, applyInitialRule=False, stepType='log',
                                    minVal=float(self.levels[-1]), maxVal=float(self.levels[0]),
                                    extraInfo={'staircase': name}, name=name, autoLog=False)

    def _snap(self, value):
        """The grid size closest to a StairHandler value (they agree up to rounding of the stored grid)."""
        k = int(np.argmin(np.abs(np.log(self.levels) - np.log(value))))
        if abs(np.log(self.levels[k] / value)) > 0.01:
            raise RuntimeError(f'{self.name}: StairHandler value {value} is not on the size grid')
        return float(self.levels[k])

    @property
    def size_ms(self):
        """Size of the next trial (ms, on the grid)."""
        return self._snap(self.handler.intensity)

    @property
    def shift_ms(self):
        """Signed shift of the next trial (+ = delay, - = advance)."""
        return self.sign * self.size_ms

    @property
    def reversals(self):
        return [self._snap(v) for v in self.handler.reversalIntensities]

    @property
    def n_trials(self):
        return len(self.handler.data)

    def update(self, correct):
        """Give the handler the answer of the trial that was just played at size_ms. Returns a dict for the output."""
        size_before = self.size_ms
        n_rev = len(self.handler.reversalIntensities)
        next(self.handler)                                     # StairHandler records the intensity of this trial
        self.handler.addResponse(int(bool(correct)), intensity=size_before)
        reversal = len(self.handler.reversalIntensities) > n_rev
        after = self.size_ms
        move = 'down' if after < size_before else 'up' if after > size_before else 'stay'
        return dict(sc_size_ms=size_before, sc_move=move, sc_reversal=reversal,
                    sc_n_reversals=len(self.handler.reversalIntensities), sc_next_size_ms=after)

    def threshold(self):
        return threshold_from_reversals(self.reversals, self.n_discard)


def threshold_from_reversals(reversals, n_discard):
    """Geometric mean of the reversal sizes after the first n_discard (NaN if there are none)."""
    r = np.asarray(reversals[n_discard:], dtype=float)
    return float(np.exp(np.log(r).mean())) if r.size else float('nan')


def make_staircases(params):
    return {name: Staircase(name, params) for name in params.STAIRCASES}


def replay(staircases, results):
    """Resume after a crash: feed back the (staircase, correct) pairs of the trials already done, in order."""
    for name, correct in results:
        staircases[name].update(bool(correct))
    return staircases


#=====================
# WORD BANK (written by build_staircase_stimuli.py)
#=====================
def participant_dir(params, pid, group):
    folder = os.path.join(params.STIM_DIR, params.PARTICIPANT_FOLDER.format(pid=pid, group=group))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'no stimulus folder {folder} - run build_staircase_stimuli.py first')
    return folder


def load_word_bank(params, folder):
    """{bank_row: dict(y, on, off, word)} - every word of this participant, as float64."""
    bank = np.load(os.path.join(folder, params.WORD_NPY), mmap_mode='r')
    index = pd.read_csv(os.path.join(folder, params.WORD_INDEX))
    return {int(r.bank_row): dict(y=np.array(bank[r.bank_row, :r.n_samples], dtype=np.float64),
                                  on=int(r.on), off=int(r.off), word=r.word)
            for r in index.itertuples()}


def trial_words(words, row, params):
    """The 6 words of one design_stim row (columns w1_row ... w6_row)."""
    return [words[int(getattr(row, f'w{k}_row'))] for k in range(1, params.N_WORDS + 1)]


#=====================
# BUILD ONE TRIAL
#=====================
def shift_vector(interval, shift_ms, n_words):
    """Phase shift: words k+1 ... n move by shift_ms for interval Ik (word 1 never moves)."""
    v = np.zeros(n_words)
    if interval is not None and shift_ms != 0:
        v[int(interval[1:]):] = shift_ms
    return v


def build_presentation(ws, interval, shift_ms, params):
    """One presentation (6 words), exactly like build_one() in build_detection_stimuli.py.

    Returns (y float32, onset_ms, offset_ms); times in ms from the start of y; offsets are audible offsets.
    """
    seq = repo_module(params, 'sequence')
    events = [dict(y=w['y'], on=w['on'], off=w['off']) for w in ws]
    events[-1]['off'] = len(ws[-1]['y']) - 1                  # keep word 6's own fade-out (no cut at -40 dB)
    shifts = shift_vector(interval, shift_ms, params.N_WORDS) + params.LEAD_MS
    y, table = seq.build_sequence(events, params.FS, soa_ms=params.IOI_MS, shifts_ms=shifts, end='offset',
                                  edge_fade_ms=params.EDGE_FADE_MS)
    y = np.r_[y, np.zeros(int(round(params.TAIL_MS / 1000 * params.FS)), np.float32)]
    onset_ms = np.array([t['onset_ms'] for t in table])
    offset_ms = onset_ms + np.array([(w['off'] - w['on']) / params.FS * 1000 for w in ws])
    return y, onset_ms, offset_ms


def build_trial(ws, interval, shift_ms, shifted_presentation, params):
    """Both presentations in ONE sound (so the ISI is sample-exact).

    shifted_presentation 1: shifted + unshifted; 2: unshifted + shifted.
    Word 1 of presentation 2 starts ISI_MS after the audible offset of word 6 of presentation 1.
    Returns (y float32, onset_ms (12), offset_ms (12)).
    """
    fs = params.FS
    plain = build_presentation(ws, interval, 0, params)
    shifted = build_presentation(ws, interval, shift_ms, params)
    (a_y, a_on, a_off), (b_y, b_on, b_off) = (shifted, plain) if shifted_presentation == 1 else (plain, shifted)
    start2 = int(round((a_off[-1] + params.ISI_MS - b_on[0]) / 1000 * fs))   # sample where presentation 2 starts
    if start2 < len(a_y):
        raise ValueError(f'ISI_MS {params.ISI_MS} is too short: presentation 2 would overlap presentation 1')
    y = np.zeros(start2 + len(b_y), np.float32)
    y[:len(a_y)] = a_y
    y[start2:] = b_y
    t2 = start2 / fs * 1000
    return y, np.r_[a_on, b_on + t2], np.r_[a_off, b_off + t2]


#=====================
# CHECK ONE TRIAL (stops at the first failure)
#=====================
def check_trial(y, onset_ms, offset_ms, interval, shift_ms, shifted_presentation, params, name=''):
    """IOIs, ISI, silent gaps, onsets re-detected in the signal, no clipping. Returns the smallest gap (ms)."""
    audio = repo_module(params, 'audio')
    fs, n = params.FS, params.N_WORDS
    tol = 1000 / fs                                            # one sample
    # (a) onset-to-onset intervals per presentation: IOI everywhere, IOI + shift at the shifted interval
    for p in (1, 2):
        want = np.full(n - 1, float(params.IOI_MS))
        if p == shifted_presentation:
            want[int(interval[1:]) - 1] += shift_ms
        got = np.diff(onset_ms[(p - 1) * n:p * n])
        assert np.all(np.abs(got - want) <= tol), f'{name}: presentation {p} IOIs {got} != {want}'
    # (b) word 1 at LEAD_MS; ISI from the audible offset of word 6 (pres. 1) to the onset of word 1 (pres. 2)
    assert abs(onset_ms[0] - params.LEAD_MS) <= tol, f'{name}: word 1 at {onset_ms[0]} ms'
    isi = onset_ms[n] - offset_ms[n - 1]
    assert abs(isi - params.ISI_MS) <= 2 * tol, f'{name}: ISI {isi:.2f} ms != {params.ISI_MS}'
    # (c) silence between words inside each presentation
    gaps = np.r_[onset_ms[1:n] - offset_ms[:n - 1], onset_ms[n + 1:] - offset_ms[n:-1]]
    assert gaps.min() >= params.MIN_GAP_MS, f'{name}: gap of {gaps.min():.1f} ms < {params.MIN_GAP_MS} ms'
    # (d) every onset re-detected IN THE SIGNAL with the same rule
    next_gap = np.r_[gaps[:n - 1], 30.0, gaps[n - 1:], 30.0]   # silence after each word (ISI / tail: >= 30)
    prev_gap = np.r_[60.0, gaps[:n - 1], 60.0, gaps[n - 1:]]   # silence before each word (lead / ISI: >= 60)
    for k in range(2 * n):
        a = int((onset_ms[k] - min(30, prev_gap[k] / 2)) / 1000 * fs)   # window never reaches the previous word
        b = int((offset_ms[k] + min(15, next_gap[k] / 2)) / 1000 * fs)
        on, _ = audio.find_on_offset(y[a:b], fs, params.ONSET_DB, params.ENV_WIN_MS)
        assert abs((a + on) / fs * 1000 - onset_ms[k]) <= 0.5, f'{name}: word {k + 1} onset not where expected'
    # (e) level, with the gain used in the session
    assert np.abs(y).max() * params.STIM_GAIN < 1, f'{name}: clips with STIM_GAIN = {params.STIM_GAIN}'
    return float(gaps.min())


def stim_id(sentence_id, interval, shift_ms, shifted_presentation):
    """Name of a trial sound, e.g. S24_03_I2_m27.78_s1 (m = advance, p = delay; s1/s2 = shifted presentation)."""
    return f"{sentence_id}_{interval}_{'p' if shift_ms > 0 else 'm'}{abs(shift_ms):g}_s{int(shifted_presentation)}"


#####################################################################################################################
# PART 2: PsychoPy / Psychtoolbox
#####################################################################################################################

#=====================
# DESIGN AND WORDS FOR THE SESSION
#=====================
def load_stim_meta(params, folder, pid, group):
    meta = json.load(open(os.path.join(folder, params.STIM_META)))
    if meta['fs'] != params.FS:
        raise ValueError(f"stimuli are {meta['fs']} Hz, params.FS is {params.FS}")
    if meta['participant'] != pid or meta['group'] != group:
        raise ValueError(f"{folder}: built for {meta['participant']}/{meta['group']}, not {pid}/{group}")
    if [float(x) for x in meta['size_levels_ms']] != [float(x) for x in params.SIZE_LEVELS_MS]:
        raise ValueError(f'{folder}: stimuli were checked on another size grid - rebuild them')
    for key in ('ioi_ms', 'isi_ms', 'lead_ms', 'tail_ms', 'min_gap_ms'):
        if meta[key] != getattr(params, key.upper()):
            raise ValueError(f'{folder}: {key} = {meta[key]} in the build, {getattr(params, key.upper())} in params')
    return meta


def load_design(params, folder, pid, group):
    """design_stim.csv (design table + word bank rows), checked."""
    path = os.path.join(folder, params.STIM_DESIGN)
    d = pd.read_csv(path)
    missing = set(params.DESIGN_COLUMNS + [f'w{k}_row' for k in range(1, params.N_WORDS + 1)]) - set(d.columns)
    if missing:
        raise ValueError(f'{path}: columns missing {sorted(missing)}')
    if (d['participant'] != pid).any() or (d['group'] != group).any():
        raise ValueError(f'{path}: participant/group inside the table do not match {pid}/{group}')
    per_block = d.groupby('block').size()
    if list(per_block.index) != list(range(1, params.N_BLOCKS + 1)) or (per_block != params.TRIALS_PER_BLOCK).any():
        raise ValueError(f'{path}: trials per block {per_block.to_dict()}, expected '
                         f'{params.TRIALS_PER_BLOCK} in each of {params.N_BLOCKS} blocks')
    return d.sort_values('trial').reset_index(drop=True)


def sentence_text(words, sentence_type, params):
    if params.CAPITALIZE_FIRST_WORD and sentence_type != 'word list':
        words = words[0].upper() + words[1:]
    return words


def pre_sound_text(kind, n, row, params):
    if kind == 'sentence':
        return sentence_text(row['words'], row['sentence_type'], params)
    if kind == 'trial_number':
        return params.TRIAL_NUMBER_TEXT.format(n=n)
    return params.FIXATION_TEXT


def make_sound(words, row, size_ms, params):
    """The trial sound at this size (same function the builder checked), gain applied.
    Returns (y, shift_ms, stim_id, pres2_start_s)."""
    if size_ms not in params.SIZE_LEVELS_MS:
        raise ValueError(f'size {size_ms} ms is not on SIZE_LEVELS_MS (not checked)')
    shift = params.DIRECTION_SIGN[row['direction']] * size_ms
    y, on, _ = build_trial(trial_words(words, row, params), row['interval'], shift,
                           int(row['shifted_presentation']), params)
    y = y * params.STIM_GAIN
    if np.abs(y).max() >= 1:
        raise ValueError(f'trial {row["trial"]} clips with STIM_GAIN = {params.STIM_GAIN}')
    pres2_start_s = (on[params.N_WORDS] - params.LEAD_MS) / 1000       # start of presentation 2 in the sound
    return y, shift, stim_id(row['sentence_id'], row['interval'], shift, row['shifted_presentation']), pres2_start_s


def restore_staircases(params, staircases, pid, group, start_block):
    """Resume: replay every finished trial of blocks < start_block from this participant's earlier output files."""
    files = sorted(glob.glob(os.path.join(params.DATA_DIR, params.OUTPUT_FILE_PATTERN.format(
        pid=pid, group=group, time='*'))))
    if not files:
        raise FileNotFoundError(f'start_block {start_block}: no earlier output file for {pid}/{group} in {params.DATA_DIR}')
    done = pd.concat([pd.read_csv(f, sep='\t') for f in files])
    done = done[done['block'] < start_block].drop_duplicates('trial', keep='last').sort_values('trial')
    expected = (start_block - 1) * params.TRIALS_PER_BLOCK
    if params.MAX_TRIALS_PER_BLOCK is None and len(done) != expected:
        raise ValueError(f'resume: found {len(done)} finished trials before block {start_block}, expected {expected}')
    replay(staircases, zip(done['staircase'], done['correct'].astype(int)))
    print(f'Staircases restored from {len(done)} trials ({", ".join(os.path.basename(f) for f in files)}):',
          {n: s.size_ms for n, s in staircases.items()})


#=====================
# AUDIO
#=====================
def to_channels(y, n_channels):
    """Mono -> (n_samples, n_channels) float64 for PsychPortAudio."""
    return np.tile(np.asarray(y, dtype=np.float64)[:, None], (1, n_channels))


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
    keys = kb.waitKeys(keyList=params.CONTINUE_KEYS + [params.QUIT_KEY], waitRelease=True)
    if keys and keys[0].name == params.QUIT_KEY:
        raise QuitExperiment


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
    """Append one trial to the tab-separated output file (header written with the first row)."""
    new = not os.path.exists(path)
    pd.DataFrame([row]).reindex(columns=columns).to_csv(path, sep='\t', mode='a', header=new, index=False)


def save_thresholds(params, staircases, path):
    rows = [dict(staircase=n, direction=s.direction, n_trials=s.n_trials, n_reversals=len(s.reversals),
                 reversals_ms=' '.join(f'{r:g}' for r in s.reversals), n_discard=s.n_discard,
                 threshold_ms=s.threshold(), last_size_ms=s.size_ms) for n, s in staircases.items()]
    d = pd.DataFrame(rows)
    d.to_csv(path, index=False)
    print(d.to_string(index=False))
    return d


#=====================
# ONE TRIAL (main and practice)
#=====================
def run_trial(stream, win, kb, params, y, pres2_start_s, pre_text, pre_height, keys_reminder, next_trial):
    """pre-sound screen -> sound PRE_SOUND_S later with label '1', label '2' at the start of presentation 2
    -> key reminder at the end of the sound -> response window (RESPONSE_WINDOW_S from the end of the sound).
    Presses before the end of the sound are ignored. Returns response and timing ('t_next_base' for the ITI).
    """
    stream.fill_buffer(to_channels(y, params.N_OUT_CHANNELS))

    WaitSecs(max(0.0, next_trial - GetSecs()))
    t_pre = display_text(pre_text, win, params, height=pre_height)
    t_scheduled = t_pre + params.PRE_SOUND_S
    t_onset = stream.start(when=t_scheduled, wait_for_start=1)
    display_text(params.INTERVAL_LABELS[0], win, params, height=params.LABEL_HEIGHT)
    WaitSecs(max(0.0, t_onset + pres2_start_s - GetSecs()))
    t_label2 = display_text(params.INTERVAL_LABELS[1], win, params, height=params.LABEL_HEIGHT)
    sound_dur = len(y) / params.FS
    t_offset = t_onset + sound_dur

    WaitSecs(max(0.0, t_offset - GetSecs()))
    t_keys = display_text(keys_reminder, win, params)
    kb.clearEvents()                                  # presses before the end of the sound are discarded
    kb.clock.reset()
    t_reset = GetSecs()
    window_end = t_offset + params.RESPONSE_WINDOW_S
    keys = kb.waitKeys(maxWait=max(0.0, window_end - t_reset),
                       keyList=params.RESPONSE_KEYS + [params.QUIT_KEY], waitRelease=True)
    if keys and keys[0].name == params.QUIT_KEY:
        raise QuitExperiment
    stream.stop(wait_for_end_playback=1)

    if keys:
        key = keys[0]
        t_press = t_reset + key.rt
        key_name, response, tDown, press_dur = key.name, params.KEY_TO_PRESENTATION[key.name], key.tDown, key.duration
    else:
        t_press, key_name, response, tDown, press_dur = None, 'no_response', 'none', None, None
    return dict(key_name=key_name, response=response,
                rt_from_offset_s=None if t_press is None else t_press - t_offset,
                key_tDown=tDown, button_press_duration=press_dur, sound_dur_s=sound_dur,
                t_pre_sound_on=t_pre, t_onset_scheduled=t_scheduled, t_onset=t_onset, t_label2_on=t_label2,
                t_offset=t_offset, t_keys_on=t_keys,
                t_next_base=t_press if t_press is not None else window_end)


def score(response, row, params):
    """1 = correct, 0 = wrong; no answer -> NO_RESPONSE_CORRECT."""
    if response == 'none':
        return params.NO_RESPONSE_CORRECT
    return int(response == int(row['shifted_presentation']))


#=====================
# PRACTICE (fixed size, feedback; not part of the staircases)
#=====================
def run_practice(stream, win, kb, params, group, T, values, design, words, screen_kind, pre_height, log):
    rows = design.head(params.N_PRACTICE)
    next_trial = GetSecs() + params.FIRST_TRIAL_DELAY_S
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        y, shift, sid, p2 = make_sound(words, row, params.PRACTICE_SIZE_MS, params)
        text = pre_sound_text(screen_kind, i, row, params)
        r = run_trial(stream, win, kb, params, y, p2, text, pre_height, values['keys_reminder'], next_trial)
        correct = score(r['response'], row, params)
        if r['response'] == 'none':
            fb = fill(T['fb_slow'], values)
        else:
            fb = fill(T['fb_correct' if correct else 'fb_wrong'], values, answer=int(row['shifted_presentation']))
        display_text(fb, win, params)
        WaitSecs(params.FEEDBACK_S)
        display_text('', win, params)
        next_trial = GetSecs() + params.ITI_S
        r.pop('t_next_base')
        log(dict(practice_trial=i, sentence_type=row['sentence_type'], sentence_id=row['sentence_id'],
                 interval=row['interval'], shifted_presentation=row['shifted_presentation'], words=row['words'],
                 screen_text=text, size_ms=params.PRACTICE_SIZE_MS, shift_ms=shift, stim_id=sid,
                 pres2_start_s=p2, correct=correct, **r))
        print(f'practice {i}/{len(rows)}  {sid:24s} -> {r["response"]} ({"correct" if correct else "wrong"})')


#=====================
# THE SESSION
#=====================
def run_session(params, setup, group):
    """Dialog -> startup -> welcome / reading / task -> practice -> start -> 4 blocks with breaks -> thresholds.
    Every trial is saved immediately; thresholds are saved at the end (also after the quit key).
    """
    # ---------- dialog
    info = dict(params.DIALOG_FIELDS)
    dlg = gui.DlgFromDict(dictionary=info, title=f'Staircase pilot - {group} group ({setup})')
    if not dlg.OK:
        core.quit()
    pid = str(info['participant_id']).strip().upper()
    start_block = int(info['start_block'])

    # ---------- design, words, staircases, output files
    folder = participant_dir(params, pid, group)
    load_stim_meta(params, folder, pid, group)
    design = load_design(params, folder, pid, group)
    words = load_word_bank(params, folder)
    screen_kind = params.PRE_SOUND_SCREEN[group]
    staircases = make_staircases(params)
    print(f'{pid} ({group}): {len(design)} trials, {len(words)} words from {folder}')

    os.makedirs(params.DATA_DIR, exist_ok=True)
    if start_block > 1:
        restore_staircases(params, staircases, pid, group, start_block)
    session_time = get_datetime_string()
    fmt = dict(pid=pid, group=group, time=session_time)
    out_path = os.path.join(params.DATA_DIR, params.OUTPUT_FILE_PATTERN.format(**fmt))
    prac_path = os.path.join(params.DATA_DIR, params.PRACTICE_OUTPUT_PATTERN.format(**fmt))
    thr_path = os.path.join(params.DATA_DIR, params.THRESHOLD_FILE_PATTERN.format(**fmt))
    print('Output:', out_path)

    def log_practice(row):
        append_row(prac_path, dict(participant=pid, group=group, session_time=session_time, **row),
                   params.PRACTICE_COLUMNS)

    # ---------- audio, window, keyboard
    stream = open_stream(params, find_device(params))
    win = visual.Window(params.WIN_SIZE, fullscr=params.FULLSCREEN, monitor=params.MONITOR,
                        units=params.UNITS, screen=params.SCREEN)
    win.mouseVisible = False
    kb = keyboard.Keyboard()

    T = params.TEXTS[group]
    values = dict(n_blocks=params.N_BLOCKS, n_practice=params.N_PRACTICE, window=f'{params.RESPONSE_WINDOW_S:g}')
    values['keys_reminder'] = fill(T['keys_reminder'], values)
    pre_height = params.SENTENCE_HEIGHT if screen_kind == 'sentence' else params.TEXT_HEIGHT

    try:
        # ---------- instructions + practice (skipped when resuming)
        if start_block == 1:
            show_and_wait(fill(T['welcome'], values), win, kb, params)
            show_and_wait(fill(T['reading'], values), win, kb, params)
            show_and_wait(fill(T['task'], values), win, kb, params)
            if params.N_PRACTICE:
                show_and_wait(fill(T['practice_intro'], values), win, kb, params)
                run_practice(stream, win, kb, params, group, T, values, design, words, screen_kind, pre_height,
                             log_practice)
        show_and_wait(fill(T['start'], values), win, kb, params)

        # ---------- blocks
        for block in range(start_block, params.N_BLOCKS + 1):
            trials = design[design['block'] == block]
            if params.MAX_TRIALS_PER_BLOCK is not None:
                trials = trials.head(params.MAX_TRIALS_PER_BLOCK)
            display_text('', win, params)
            next_trial = GetSecs() + params.FIRST_TRIAL_DELAY_S

            for i, (_, row) in enumerate(trials.iterrows(), start=1):
                sc = staircases[row['staircase']]
                y, shift, sid, p2 = make_sound(words, row, sc.size_ms, params)     # built now: size known now
                text = pre_sound_text(screen_kind, i, row, params)
                r = run_trial(stream, win, kb, params, y, p2, text, pre_height, values['keys_reminder'], next_trial)
                display_text('', win, params)                                       # blank ITI
                next_trial = r.pop('t_next_base') + params.ITI_S

                correct = score(r['response'], row, params)
                size = sc.size_ms
                upd = sc.update(correct)
                out = {c: row[c] for c in params.DESIGN_COLUMNS}
                out.update(session_time=session_time, params_file=params.__name__, pre_sound_screen=screen_kind,
                           screen_text=text, size_ms=size, shift_ms=shift, stim_id=sid, pres2_start_s=p2,
                           correct=correct, **r, **upd)
                append_row(out_path, out, params.OUTPUT_COLUMNS)
                rt = r['rt_from_offset_s']
                print(f"block {block} trial {i}/{len(trials)}  {row['staircase']:9s} {sid:24s} -> "
                      f"{str(r['response']):4s} {'correct' if correct else 'wrong  '} "
                      f"{'' if rt is None else f'RT {rt:.3f} s'}  next {upd['sc_next_size_ms']:g} ms"
                      f"{'  REVERSAL' if upd['sc_reversal'] else ''}")

            if block < params.N_BLOCKS:
                show_and_wait(fill(T['break'], values, block=block), win, kb, params)
        show_and_wait(fill(T['end'], values), win, kb, params)

    except QuitExperiment:
        print('Quit key pressed - data saved up to the last completed trial:', out_path)

    finally:                       # always save thresholds, close audio and window
        if any(s.n_trials for s in staircases.values()):
            save_thresholds(params, staircases, thr_path)
            print('Thresholds:', thr_path)
        stream.close()
        win.close()