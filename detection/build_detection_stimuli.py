# %% BUILD DETECTION STIMULI FOR ONE PARTICIPANT
"""
Builds the sounds of ONE participant's design table and saves everything in that participant's own folder.
Nothing is ever overwritten: if the folder exists, the script stops (delete/rename it yourself to rebuild).

    stimuli/<pid>_<group>/detection_stimuli.npy      float32 (162, max_samples); row r = trial r + 1,
                                                     zero-padded after its true length (n_samples)
    stimuli/<pid>_<group>/detection_stim_index.csv   one row per trial: design + stim_id, stim_row, n_samples,
                                                     words, onset/offset of every word (ms from file start)
    stimuli/<pid>_<group>/detection_stim_meta.json   settings, participant, source table, build time
    stimuli/<pid>_<group>/design_stim.csv            the design table + stim_id, stim_row, n_samples
                                                     (this is what the experiment reads)
    stimuli/<pid>_<group>/examples/*.wav             a few trials as wav, only for listening

Reading one trial in the experiment:
    stims = np.load('stimuli/P01_speech/detection_stimuli.npy', mmap_mode='r')
    y = np.array(stims[row['stim_row'], :row['n_samples']])

Jitter = phase shift at interval Ik: word k+1 and every later word move by shift_ms
(+ = later, - = earlier). Word 1 never moves. Catch trials use the unshifted stimulus.
"""
import os
import sys
import json
import shutil
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf

import os                                                        # (if not already imported)
PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(PROJECT_DIR)

REPO_DIR = PROJECT_DIR
if REPO_DIR is not None:
    sys.path.insert(0, REPO_DIR)
from repo import build_sequence, find_on_offset, load_audio

# %% SETTINGS
PARTICIPANT_ID = None         # e.g. 'P01'; None = ask
GROUP          = None         # 'speech' / 'acoustic'; None = ask

ITEMS_CSV    = 'sws_words/items.csv'        # from sentence_lists_to_tts.py
SWS_INFO_CSV = 'sws_words/sws_info.csv'     # from make_sws_words_trimmed.py (base -> sws_file)
DESIGN_DIR   = 'design_tables'              # detection_<pid>_<group>.csv from generate_detection_table.py
OUT_ROOT     = 'stimuli'                    # -> stimuli/<pid>_<group>/

LIST_TO_TYPE = {'items_2_4': 'S24', 'items_3_3': 'S33'}    # sentence-list file name -> sentence code
N_ITEMS      = 18                                           # first N items of each list
WL_FROM      = 'S33'          # PLACEHOLDER: WL_k uses the words of S33_k until the word list exists

N_WORDS      = 6
IOI_MS       = 450            # onset-to-onset grid
FS           = 44100

LEAD_MS      = 50             # silence before word 1's onset
TAIL_MS      = 50             # silence after word 6 ends
EDGE_FADE_MS = 2              # file-edge fade (falls in the silence)
ONSET_DB     = -40            # onset/offset rule: same as the SWS trimming (trim_sws)
ENV_WIN_MS   = 5
MIN_GAP_MS   = 20             # smallest allowed silence between a word's offset and the next onset

N_EXAMPLE_WAVS = 6            # trials also written as wav (for listening); 0 = none

# %% 0 PARTICIPANT, TABLE, OUTPUT FOLDER
pid = PARTICIPANT_ID or input('Enter participant ID: ').strip()
group = GROUP or input('Enter experimental group (speech / acoustic): ').strip()
design_path = os.path.join(DESIGN_DIR, f'detection_{pid}_{group}.csv')
if not os.path.exists(design_path):
    raise FileNotFoundError(f'no design table {design_path} - run generate_detection_table.py first')
design = pd.read_csv(design_path).sort_values('trial').reset_index(drop=True)
if (design['participant'] != pid).any() or (design['group'] != group).any():
    raise ValueError(f'{design_path}: participant/group inside the table do not match {pid}/{group}')

out_dir = os.path.join(OUT_ROOT, f'{pid}_{group}')
if os.path.exists(out_dir):
    raise FileExistsError(f'{out_dir} already exists - not overwriting. Delete or rename it to rebuild.')
tmp_dir = out_dir + '_incomplete'            # built here, renamed to out_dir only when everything passed
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)                   # leftover of an interrupted run of this script
print(f'{pid} ({group}): {len(design)} trials from {design_path}')

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
max_early = -min(0, design['shift_ms'].min())
limit = IOI_MS - max_early - MIN_GAP_MS
too_long = [os.path.basename(p) for p, w in words.items() if w['sound_ms'] > limit]
if too_long:
    print(f'WARNING longer than {limit:.0f} ms (may fail the gap check):', ', '.join(too_long))

# %% 3 STIMULUS NAMES
def stim_id(sentence_id, interval, shift_ms):
    """Name of a stimulus. Catch / unshifted: interval None or shift 0."""
    if interval is None or shift_ms == 0:
        return f'{sentence_id}_none'
    return f"{sentence_id}_{interval}_{'p' if shift_ms > 0 else 'm'}{abs(int(shift_ms))}"

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

# %% 6 BUILD EVERY TRIAL (row r = trial r + 1)
built, rows = [], []
for r, d in enumerate(design.itertuples()):
    iv = d.interval if isinstance(d.interval, str) else None          # pandas stores None as NaN
    shift = float(d.shift_ms)
    y, onset_ms, offset_ms = build_one(d.sentence_id, iv, shift)
    min_gap = check_one(d.sentence_id, iv, shift, y, onset_ms, offset_ms)
    built.append(y.astype(np.float32))
    row = dict(trial=d.trial, stim_id=stim_id(d.sentence_id, iv, shift), stim_row=r, n_samples=len(y),
               dur_ms=round(len(y) / FS * 1000, 2), min_gap_ms=round(min_gap, 2),
               words=' '.join(w['word'] for w in sentences[d.sentence_id]))
    row |= {f'on{k + 1}_ms': round(v, 3) for k, v in enumerate(onset_ms)}
    row |= {f'off{k + 1}_ms': round(v, 3) for k, v in enumerate(offset_ms)}
    rows.append(row)
    if (r + 1) % 50 == 0:
        print(f'{r + 1}/{len(design)}')

store = np.zeros((len(built), max(map(len, built))), np.float32)
for r, y in enumerate(built):
    store[r, :len(y)] = y
stim = pd.DataFrame(rows)

# %% 7 SAVE (to the temporary folder, then rename)
os.makedirs(tmp_dir)
np.save(os.path.join(tmp_dir, 'detection_stimuli.npy'), store)

design_stim = design.merge(stim[['trial', 'stim_id', 'stim_row', 'n_samples']], on='trial', validate='one_to_one')
assert (design_stim['stim_row'].values == np.arange(len(design_stim))).all()
design_stim.to_csv(os.path.join(tmp_dir, 'design_stim.csv'), index=False)
design.merge(stim, on='trial', validate='one_to_one').to_csv(os.path.join(tmp_dir, 'detection_stim_index.csv'), index=False)

shifts = sorted(design['shift_ms'].unique())
json.dump(dict(participant=pid, group=group, built=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               design_table=os.path.abspath(design_path),
               design_table_md5=hashlib.md5(open(design_path, 'rb').read()).hexdigest(),
               fs=FS, ioi_ms=IOI_MS, n_words=N_WORDS, shifts_ms=[float(s) for s in shifts],
               lead_ms=LEAD_MS, tail_ms=TAIL_MS, edge_fade_ms=EDGE_FADE_MS, onset_db=ONSET_DB,
               env_win_ms=ENV_WIN_MS, min_gap_ms=MIN_GAP_MS, n_items=N_ITEMS, list_to_type=LIST_TO_TYPE,
               wl_placeholder_from=WL_FROM, n_trials=len(stim), max_samples=store.shape[1], dtype='float32',
               words_files={w['word']: w['sws_file'] for s in sentences.values() for w in s}),
          open(os.path.join(tmp_dir, 'detection_stim_meta.json'), 'w'), indent=2)

if N_EXAMPLE_WAVS:
    ex_dir = os.path.join(tmp_dir, 'examples')
    os.makedirs(ex_dir)
    pick = pd.concat([stim[stim.stim_id.str.endswith('_none')].head(1),
                      stim[~stim.stim_id.str.endswith('_none')].head(N_EXAMPLE_WAVS - 1)])
    for r in pick.itertuples():
        sf.write(os.path.join(ex_dir, f'trial{r.trial:03d}_{r.stim_id}.wav'), store[r.stim_row, :r.n_samples], FS)

os.rename(tmp_dir, out_dir)
print(f'\n✓ {len(stim)} trials -> {out_dir}/  ({store.nbytes / 1e6:.0f} MB)')
print(f'  duration {stim.dur_ms.min():.0f}-{stim.dur_ms.max():.0f} ms; smallest gap {stim.min_gap_ms.min():.1f} ms '
      f'({stim.loc[stim.min_gap_ms.idxmin(), "stim_id"]})')