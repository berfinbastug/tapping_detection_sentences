# %% BUILD DETECTION STIMULI
"""
One stimulus per sentence x interval x shift (full grid), stored as ONE big .npy file.

    stimuli/detection_stimuli.npy    float32, shape (n_stimuli, max_samples); row r = one stimulus,
                                     zero-padded after its true length (n_samples in the index)
    stimuli/detection_stim_index.csv one row per stimulus: row, stim_id, sentence, interval, shift,
                                     n_samples, onset/offset of every word (ms from file start)
    stimuli/detection_stim_meta.json settings used
    stimuli/examples/*.wav           a few stimuli as wav, only for listening
    design_tables_stim/*.csv         every design table + stim_id + stim_row

Reading one trial in the experiment:
    stims = np.load('stimuli/detection_stimuli.npy', mmap_mode='r')     # nothing loaded yet
    y = np.array(stims[row['stim_row'], :row['n_samples']])            # reads only this row

Jitter = phase shift at interval Ik: word k+1 and every later word move by shift_ms
(+ = later, - = earlier). Word 1 never moves. Catch trials use the unshifted stimulus.
"""
import os
import sys
import glob
import json

import numpy as np
import pandas as pd
import soundfile as sf

REPO_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences/'   # folder that CONTAINS the repo package
if REPO_DIR is not None:
    sys.path.insert(0, REPO_DIR)
from repo import build_sequence, find_on_offset, load_audio

# %% SETTINGS
ITEMS_CSV    = 'sws_words/items.csv'        # from sentence_lists_to_tts.py
SWS_INFO_CSV = 'sws_words/sws_info.csv'     # from make_sws_words_trimmed.py (base -> sws_file)
DESIGN_DIR   = 'design_tables'              # detection_*.csv from the design notebook
OUT_DIR      = 'stimuli'
DESIGN_OUT   = 'design_tables_stim'

LIST_TO_TYPE = {'items_2_4': 'S24', 'items_3_3': 'S33'}    # sentence-list file name -> sentence code
N_ITEMS      = 18                                           # first N items of each list
WL_FROM      = 'S33'          # PLACEHOLDER: WL_k uses the words of S33_k until the word list exists

SIZES_MS     = [15, 30, 50]   # PLACEHOLDERS - must match JITTER_SIZES_MS in the design notebook
INTERVALS    = ['I1', 'I2', 'I3', 'I4', 'I5']
N_WORDS      = 6
IOI_MS       = 450            # onset-to-onset grid
FS           = 44100

LEAD_MS      = 50             # silence before word 1's onset
TAIL_MS      = 50             # silence after word 6 ends
EDGE_FADE_MS = 2              # file-edge fade (falls in the silence)
ONSET_DB     = -40            # onset/offset rule: same as the SWS trimming (trim_sws)
ENV_WIN_MS   = 5
MIN_GAP_MS   = 20             # smallest allowed silence between a word's offset and the next onset

N_EXAMPLE_WAVS = 8            # stimuli also written as wav (for listening); 0 = none
OVERWRITE      = True

# %% 1 SENTENCES -> WORDS
items = pd.read_csv(ITEMS_CSV)
sws_info = pd.read_csv(SWS_INFO_CSV)
items = items.merge(sws_info[['base', 'sws_file']], on='base', how='left')
if items['sws_file'].isna().any():
    missing = sorted(items.loc[items['sws_file'].isna(), 'word'].unique())
    raise FileNotFoundError(f'no SWS file for: {missing}')

sentences = {}                                         # sentence_id -> list of 6 rows (word, base, sws_file)
for lst, code in LIST_TO_TYPE.items():
    d = items[items['list'] == lst]
    if d.empty:
        raise ValueError(f'list {lst!r} not in {ITEMS_CSV} (lists there: {sorted(items["list"].unique())})')
    for k, item in enumerate(sorted(d['item'].unique())[:N_ITEMS], start=1):
        w = d[d['item'] == item].sort_values('pos')
        if len(w) != N_WORDS:
            raise ValueError(f'{lst} item {item}: {len(w)} words, expected {N_WORDS}')
        sentences[f'{code}_{k:02d}'] = w[['word', 'base', 'sws_file']].to_dict('records')
for k in range(1, N_ITEMS + 1):                          # PLACEHOLDER word list
    sentences[f'WL_{k:02d}'] = sentences[f'{WL_FROM}_{k:02d}']
print(f'{len(sentences)} sentences')

# %% 2 LOAD EVERY WORD ONCE (+ its onset / offset with the trimming rule)
words = {}
for path in sorted({w['sws_file'] for s in sentences.values() for w in s}):
    y, fs = load_audio(path)
    if fs != FS:
        raise ValueError(f'{path}: {fs} Hz, expected {FS}')
    on, off = find_on_offset(y, fs, ONSET_DB, ENV_WIN_MS)
    words[path] = dict(y=y.astype(np.float64), on=on, off=off, sound_ms=(off - on) / fs * 1000)
dur = np.array([w['sound_ms'] for w in words.values()])
print(f'{len(words)} words, audible {dur.min():.0f}-{dur.max():.0f} ms (median {np.median(dur):.0f})')
too_long = [os.path.basename(p) for p, w in words.items() if w['sound_ms'] > IOI_MS - max(SIZES_MS) - MIN_GAP_MS]
if too_long:
    print(f'WARNING longer than {IOI_MS - max(SIZES_MS) - MIN_GAP_MS} ms (may fail the gap check):', ', '.join(too_long))

# %% 3 THE STIMULUS GRID
def stim_id(sentence_id, interval, shift_ms):
    """Name of a stimulus. Catch / unshifted: interval None or shift 0."""
    if interval is None or shift_ms == 0:
        return f'{sentence_id}_none'
    return f"{sentence_id}_{interval}_{'p' if shift_ms > 0 else 'm'}{abs(int(shift_ms))}"

grid = []
for sid in sentences:
    grid.append(dict(stim_id=stim_id(sid, None, 0), sentence_id=sid, interval=None, shift_ms=0))
    for iv in INTERVALS:
        for s in SIZES_MS:
            for sign in (-1, 1):
                grid.append(dict(stim_id=stim_id(sid, iv, sign * s), sentence_id=sid, interval=iv, shift_ms=sign * s))
grid = pd.DataFrame(grid)
grid.insert(0, 'row', np.arange(len(grid)))
assert grid['stim_id'].is_unique
print(f'{len(grid)} stimuli = {len(sentences)} sentences x (1 + {len(INTERVALS)} intervals x {2 * len(SIZES_MS)} shifts)')

# %% 4 BUILD ONE STIMULUS
lead_n, tail_n = int(round(LEAD_MS / 1000 * FS)), int(round(TAIL_MS / 1000 * FS))

def shift_vector(interval, shift_ms):
    """Phase shift: words k+1 ... 6 move by shift_ms for interval Ik."""
    v = np.zeros(N_WORDS)
    if interval is not None:
        v[int(interval[1:]):] = shift_ms
    return v

def build_one(sid, interval, shift_ms):
    ws = [words[w['sws_file']] for w in sentences[sid]]
    events = [dict(y=w['y'], on=w['on'], off=w['off']) for w in ws]
    events[-1]['off'] = len(ws[-1]['y']) - 1        # keep word 6's own fade-out (no cut at -40 dB)
    shifts = shift_vector(interval, shift_ms) + LEAD_MS   # LEAD_MS on every word = silence before word 1
    y, table = build_sequence(events, FS, soa_ms=IOI_MS, shifts_ms=shifts, end='offset',
                              edge_fade_ms=EDGE_FADE_MS)
    y = np.r_[y, np.zeros(tail_n, np.float32)]
    onset_ms = np.array([t['onset_ms'] for t in table])
    offset_ms = onset_ms + np.array([(w['off'] - w['on']) / FS * 1000 for w in ws])   # audible offsets
    return y, onset_ms, offset_ms

# %% 5 CHECKS FOR ONE STIMULUS (stop at the first failure)
def check_one(sid, interval, shift_ms, y, onset_ms, offset_ms):
    name = stim_id(sid, interval, shift_ms)
    tol = 1000 / FS                                   # one sample
    # (a) onset-to-onset intervals: 450 everywhere, 450 + shift at the shifted interval
    want = np.full(N_WORDS - 1, float(IOI_MS))
    if interval is not None:
        want[int(interval[1:]) - 1] += shift_ms
    got = np.diff(onset_ms)
    assert np.all(np.abs(got - want) <= tol), f'{name}: IOIs {got} != {want}'
    # (b) word 1 at LEAD_MS
    assert abs(onset_ms[0] - LEAD_MS) <= tol, f'{name}: word 1 at {onset_ms[0]} ms'
    # (c) silence between words
    gaps = onset_ms[1:] - offset_ms[:-1]
    assert gaps.min() >= MIN_GAP_MS, f'{name}: gap of {gaps.min():.1f} ms < {MIN_GAP_MS} ms'
    # (d) onsets re-detected IN THE SIGNAL, word by word, with the same rule
    for k in range(N_WORDS):
        a = int((onset_ms[k] - 30) / 1000 * FS)
        b = int((offset_ms[k] + min(15, gaps[k] / 2 if k < N_WORDS - 1 else 15)) / 1000 * FS)
        on, _ = find_on_offset(y[a:b], FS, ONSET_DB, ENV_WIN_MS)
        assert abs((a + on) / FS * 1000 - onset_ms[k]) <= 0.5, f'{name}: word {k + 1} onset not where expected'
    # (e) level
    assert np.abs(y).max() < 1, f'{name}: clips'
    return gaps.min()

# %% 6 BUILD ALL -> ONE .npy (written straight to disk)
os.makedirs(OUT_DIR, exist_ok=True)
npy_path = os.path.join(OUT_DIR, 'detection_stimuli.npy')
if os.path.exists(npy_path) and not OVERWRITE:
    raise FileExistsError(f'{npy_path} exists (OVERWRITE = False)')

longest = max(len(words[s[-1]['sws_file']]['y']) - words[s[-1]['sws_file']]['on'] for s in sentences.values())
max_n = int(round((LEAD_MS + (N_WORDS - 1) * IOI_MS + max(SIZES_MS)) / 1000 * FS)) + longest + tail_n + 1
store = np.lib.format.open_memmap(npy_path, mode='w+', dtype=np.float32, shape=(len(grid), max_n))

rows = []
for r in grid.itertuples():
    iv = r.interval if isinstance(r.interval, str) else None          # pandas stores None as NaN
    y, onset_ms, offset_ms = build_one(r.sentence_id, iv, r.shift_ms)
    min_gap = check_one(r.sentence_id, iv, r.shift_ms, y, onset_ms, offset_ms)
    store[r.row, :len(y)] = y
    store[r.row, len(y):] = 0
    row = dict(row=r.row, stim_id=r.stim_id, sentence_id=r.sentence_id,
               sentence_type={'S24': '2|4', 'S33': '3|3', 'WL': 'word list'}[r.sentence_id.split('_')[0]],
               interval=r.interval, shift_ms=r.shift_ms, n_samples=len(y), dur_ms=round(len(y) / FS * 1000, 2),
               min_gap_ms=round(min_gap, 2), words=' '.join(w['word'] for w in sentences[r.sentence_id]))
    row |= {f'on{k + 1}_ms': round(v, 3) for k, v in enumerate(onset_ms)}
    row |= {f'off{k + 1}_ms': round(v, 3) for k, v in enumerate(offset_ms)}
    rows.append(row)
    if (r.row + 1) % 200 == 0:
        print(f'{r.row + 1}/{len(grid)}')
store.flush()
del store

index = pd.DataFrame(rows)
index.to_csv(os.path.join(OUT_DIR, 'detection_stim_index.csv'), index=False)
json.dump(dict(fs=FS, ioi_ms=IOI_MS, n_words=N_WORDS, sizes_ms=SIZES_MS, intervals=INTERVALS,
               lead_ms=LEAD_MS, tail_ms=TAIL_MS, edge_fade_ms=EDGE_FADE_MS, onset_db=ONSET_DB,
               env_win_ms=ENV_WIN_MS, min_gap_ms=MIN_GAP_MS, n_items=N_ITEMS, list_to_type=LIST_TO_TYPE,
               wl_placeholder_from=WL_FROM, n_stimuli=len(index), max_samples=max_n, dtype='float32'),
          open(os.path.join(OUT_DIR, 'detection_stim_meta.json'), 'w'), indent=2)

print(f'\n✓ {len(index)} stimuli -> {npy_path}  ({os.path.getsize(npy_path) / 1e6:.0f} MB)')
print(f'  duration {index.dur_ms.min():.0f}-{index.dur_ms.max():.0f} ms; smallest gap {index.min_gap_ms.min():.1f} ms '
      f'({index.loc[index.min_gap_ms.idxmin(), "stim_id"]})')

# %% 7 ADD stim_id + stim_row TO THE DESIGN TABLES
os.makedirs(DESIGN_OUT, exist_ok=True)
lookup = index.set_index('stim_id')
for path in sorted(glob.glob(os.path.join(DESIGN_DIR, 'detection_*.csv'))):
    d = pd.read_csv(path)
    sizes = set(d['size_ms'].dropna().astype(int))
    if not sizes <= set(SIZES_MS):
        raise ValueError(f'{path}: sizes {sorted(sizes)} not all in SIZES_MS {SIZES_MS}')
    d['stim_id'] = [stim_id(s, None if pd.isna(iv) else iv, sh)
                    for s, iv, sh in zip(d['sentence_id'], d['interval'], d['shift_ms'])]
    if not d['stim_id'].isin(lookup.index).all():
        raise ValueError(f'{path}: stimuli missing: {sorted(set(d.stim_id) - set(lookup.index))[:5]}')
    d['stim_row'] = lookup.loc[d['stim_id'], 'row'].values
    d['n_samples'] = lookup.loc[d['stim_id'], 'n_samples'].values
    # consistency: design and stimulus agree on interval and shift
    assert (lookup.loc[d['stim_id'], 'shift_ms'].values == d['shift_ms'].values).all()
    d.to_csv(os.path.join(DESIGN_OUT, os.path.basename(path)), index=False)
    print(f'✓ {os.path.basename(path)}: {len(d)} trials -> {DESIGN_OUT}/')

# %% 8 EXAMPLE WAVS (for listening only - the experiment reads the .npy)
if N_EXAMPLE_WAVS:
    ex_dir = os.path.join(OUT_DIR, 'examples')
    os.makedirs(ex_dir, exist_ok=True)
    stims = np.load(npy_path, mmap_mode='r')
    first = index[index.sentence_id == index.sentence_id.iloc[0]]
    pick = pd.concat([first[first.interval.isna()],
                      first[(first.shift_ms.abs() == max(SIZES_MS))]]).head(N_EXAMPLE_WAVS)
    for r in pick.itertuples():
        sf.write(os.path.join(ex_dir, r.stim_id + '.wav'), np.array(stims[r.row, :r.n_samples]), FS)
    print(f'{len(pick)} example wavs -> {ex_dir}/')
