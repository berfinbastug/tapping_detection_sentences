# %% BUILD THE TAPPING TRAINING STIMULI (ONCE, THE SAME FOR EVERY PARTICIPANT)
"""
Builds every distinct sound of the tapping training (make_training_design_tapping.py) exactly like the main
stimuli (build_tapping_stimuli.py): 6 noises | 6 words | 6 noises | the same 6 words | 6 noises on the 450 ms grid,
phase shifts, the shared noise pool (same shared gain), LEAD_MS / TAIL_MS silence, the -40 dB onset rule.
Nothing is ever overwritten: if OUT_DIR exists the script stops; it builds in <OUT_DIR>_incomplete and renames the
folder only when every check has passed.

Inputs
    design_tables/tapping_training_design.csv, design_tables/tapping_training_events.csv
    sentence_lists/training_items_tapping.csv + training_items.csv     (item -> words)
    sws_words/words.csv (word -> base), sws_words/sws_info.csv (base -> SWS file)
    sws_words/noise_pool/                                               (make_noise_pool.py)
Outputs in training_stimuli_tapping/
    training_stimuli.npy     float32 (n_stimuli, max_samples); one row per distinct sound, zero-padded
    design_stim.csv          the training table + stim_row, n_samples, tokens   (what the training reads)
    events_stim.csv          events table + word / token + ACTUAL onset/offset of every sound (ms from file start)
    training_stim_meta.json  settings, md5 of all inputs, token seed, build time
    wav/<stim_id>.wav        every training sound as wav, for listening
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

PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(PROJECT_DIR)
REPO_DIR = PROJECT_DIR

if REPO_DIR is not None:
    sys.path.insert(0, REPO_DIR)
from repo import build_sequence, find_on_offset, load_audio

# %% SETTINGS
DESIGN_CSV  = 'tapping/design_tables/tapping_training_design.csv'
EVENTS_CSV  = 'tapping/design_tables/tapping_training_events.csv'
ITEM_FILES  = ['sentence_lists/training_items_tapping.csv', 'sentence_lists/training_items.csv']
WORDS_CSV   = 'sws_words/words.csv'
SWS_INFO_CSV = 'sws_words/sws_info.csv'
NOISE_DIR   = 'sws_words/noise_pool'
OUT_DIR     = 'tapping/training_stimuli_tapping'

LOWER_ITEM_INITIAL = True     # same rule as sentence_lists_to_tts.py (first word of an item lowercased)
TOKEN_SEED = 2026             # noise token order (fixed: every participant hears the same training)

N_EVENTS     = 30
N_WORDS      = 6
IOI_MS       = 450            # must match build_tapping_stimuli.py
FS           = 44100
LEAD_MS      = 50
TAIL_MS      = 50
EDGE_FADE_MS = 2
ONSET_DB     = -40
ENV_WIN_MS   = 5
MIN_GAP_MS   = 5              # smaller than the main builder's 20: a -90 ms shift leaves about 15 ms between sounds
REDETECT_TOL_MS = 0.5


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


# %% 0 TABLES, OUTPUT FOLDER
design = pd.read_csv(DESIGN_CSV)
events = pd.read_csv(EVENTS_CSV).sort_values(['stim_id', 'event']).reset_index(drop=True)
stims = design.drop_duplicates('stim_id').reset_index(drop=True)
if set(events['stim_id']) != set(stims['stim_id']):
    raise ValueError('events table and design table have different stim_ids')
for sid, e in events.groupby('stim_id'):
    if list(e['event']) != list(range(1, N_EVENTS + 1)):
        raise ValueError(f'events table: {sid} does not have events 1-{N_EVENTS}')
if os.path.exists(OUT_DIR):
    raise FileExistsError(f'{OUT_DIR} already exists - not overwriting. Delete or rename it to rebuild.')
tmp_dir = OUT_DIR + '_incomplete'
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
print(f'{len(design)} training trials, {len(stims)} distinct sounds')

# %% 1 ITEMS -> WORDS -> SWS FILES
items = pd.concat([pd.read_csv(f, dtype=str) for f in ITEM_FILES], ignore_index=True).set_index('item_id')
words_tab = pd.read_csv(WORDS_CSV)
sws_info = pd.read_csv(SWS_INFO_CSV).set_index('base')
word_to_base = dict(zip(words_tab['word'], words_tab['base']))
item_files = {}
missing = []
for item in stims['item_id'].unique():
    ws = [items.loc[item, f'slot{k}'] for k in range(1, N_WORDS + 1)]
    if LOWER_ITEM_INITIAL:
        ws[0] = ws[0].lower()
    if ' '.join(ws).lower() != ' '.join(design.loc[design.item_id == item, 'words'].iloc[0].split()).lower():
        raise ValueError(f'{item}: the words in {ITEM_FILES} differ from the design table - rerun the design script')
    files = []
    for w in ws:
        base = word_to_base.get(w)
        if base is None or base not in sws_info.index:
            missing.append(w)
            continue
        files.append(dict(word=w, sws_file=sws_info.loc[base, 'sws_file']))
    item_files[item] = files
if missing:
    raise FileNotFoundError(f'no SWS word yet for: {sorted(set(missing))} - run the three word scripts first')

# %% 2 NOISE POOL (shared with the main experiment)
with open(os.path.join(NOISE_DIR, 'noise_pool_exact.json')) as fh:
    pool = json.load(fh)
if pool['fs'] != FS or pool['onset_thresh_db'] != ONSET_DB or pool['env_win_ms'] != ENV_WIN_MS:
    raise ValueError('noise pool: different sample rate or onset rule than the settings here')
GAIN = pool['shared_gain']
pad = pool['pad_samples']
pool_on, pool_off = pool['onset_sample_padded'], pool['offset_sample_padded']
tokens, token_dev_ms = [], []
for f in pool['token_files']:
    y, fs = load_audio(os.path.join(NOISE_DIR, f))
    if fs != FS or len(y) != pool['n_samples']:
        raise ValueError(f'{f}: {fs} Hz / {len(y)} samples, expected {FS} Hz / {pool["n_samples"]}')
    y = np.r_[np.zeros(pad), y.astype(np.float64), np.zeros(pad)]
    tokens.append(y)
    token_dev_ms.append((find_on_offset(y, FS, ONSET_DB, ENV_WIN_MS)[0] - pool_on) / FS * 1000)
noise_sound_ms = (pool_off - pool_on) / FS * 1000

# %% 3 WORDS (x shared gain; onset / offset with the trimming rule)
words, changed = {}, []
for path in sorted({w['sws_file'] for fs_ in item_files.values() for w in fs_}):
    if pool['reference_words'].get(os.path.basename(path)) != md5(path):
        changed.append(os.path.basename(path))
        continue
    y, fs = load_audio(path)
    y = y.astype(np.float64) * GAIN
    on, off = find_on_offset(y, fs, ONSET_DB, ENV_WIN_MS)
    words[path] = dict(y=y, on=on, off=off, sound_ms=(off - on) / fs * 1000)
if changed:
    raise ValueError(f'{len(changed)} words are not the ones the noise pool was made from (new or changed since '
                     f'make_noise_pool.py), e.g. {changed[:5]} - rebuild the noise pool first')
lead_n, tail_n = int(round(LEAD_MS / 1000 * FS)), int(round(TAIL_MS / 1000 * FS))

# %% 4 TOKEN ORDER (fixed; 18 different tokens per sound)
rng = np.random.default_rng(TOKEN_SEED)
n_noise = int((events.loc[events.stim_id == stims.stim_id[0], 'kind'] == 'noise').sum())
if len(tokens) < n_noise:
    raise ValueError(f'the noise pool has {len(tokens)} tokens, a sound needs {n_noise} different ones')
token_order = {sid: [int(k) for k in rng.permutation(len(tokens))[:n_noise]] for sid in stims['stim_id']}


# %% 5 BUILD + CHECK ONE SOUND
def build_one(ev, item, order):
    seq, info, tok = [], [], iter(order)
    for r in ev.itertuples(index=False):
        if r.kind == 'noise':
            k = next(tok)
            seq.append(dict(y=tokens[k], on=pool_on, off=pool_off))
            info.append(dict(word='', token=k + 1, sound_ms=noise_sound_ms, det_ms=token_dev_ms[k]))
        else:
            w = item_files[item][r.pos_in_part - 1]
            wd = words[w['sws_file']]
            seq.append(dict(y=wd['y'], on=wd['on'], off=wd['off']))
            info.append(dict(word=w['word'], token=None, sound_ms=wd['sound_ms'], det_ms=0.0))
    seq[-1] = dict(seq[-1], off=len(seq[-1]['y']) - 1)       # keep sound 30's own fade-out
    shifts = ev['shift_ms'].to_numpy(float) + LEAD_MS
    y, table = build_sequence(seq, FS, soa_ms=IOI_MS, shifts_ms=shifts, end='offset', edge_fade_ms=EDGE_FADE_MS)
    y = np.r_[y, np.zeros(tail_n, np.float32)]
    onset_ms = np.array([t['onset_ms'] for t in table])
    offset_ms = onset_ms + np.array([i['sound_ms'] for i in info])
    return y, onset_ms, offset_ms, info


def check_one(sid, ev, y, onset_ms, offset_ms, info, order):
    tol = 1000 / FS + 1e-6
    want = ev['intended_onset_ms'].to_numpy(float) + LEAD_MS
    assert np.all(np.abs(onset_ms - want) <= tol), f'{sid}: onsets differ from the design'
    assert np.all(np.abs(np.diff(onset_ms) - np.diff(want)) <= tol), f'{sid}: intervals differ from the design'
    gaps = onset_ms[1:] - offset_ms[:-1]
    assert gaps.min() >= MIN_GAP_MS, f'{sid}: gap of {gaps.min():.1f} ms < {MIN_GAP_MS} ms (that shift does not fit)'
    dev = np.zeros(N_EVENTS)
    for k in range(N_EVENTS):
        a_ms = onset_ms[k] - 30 if k == 0 else max(onset_ms[k] - 30, (offset_ms[k - 1] + onset_ms[k]) / 2)
        b_ms = offset_ms[k] + (min(15, gaps[k] / 2) if k < N_EVENTS - 1 else 15)
        a, b = int(a_ms / 1000 * FS), int(b_ms / 1000 * FS)
        on, _ = find_on_offset(y[a:b], FS, ONSET_DB, ENV_WIN_MS)
        dev[k] = (a + on) / FS * 1000 - (onset_ms[k] + info[k]['det_ms'])
    assert np.abs(dev).max() <= REDETECT_TOL_MS, f'{sid}: sound {np.abs(dev).argmax() + 1} re-detected off'
    assert np.abs(y).max() < 1, f'{sid}: clips'
    assert len(set(order)) == len(order), f'{sid}: a noise token repeats'
    return gaps.min()


# %% 6 BUILD EVERY DISTINCT SOUND
built, stim_rows, event_rows = [], [], []
for r, s in enumerate(stims.itertuples(index=False)):
    ev = events[events.stim_id == s.stim_id]
    y, onset_ms, offset_ms, info = build_one(ev, s.item_id, token_order[s.stim_id])
    min_gap = check_one(s.stim_id, ev, y, onset_ms, offset_ms, info, token_order[s.stim_id])
    built.append(y.astype(np.float32))
    stim_rows.append(dict(stim_id=s.stim_id, stim_row=r, n_samples=len(y), dur_ms=round(len(y) / FS * 1000, 2),
                          min_gap_ms=round(min_gap, 2), tokens=' '.join(str(k + 1) for k in token_order[s.stim_id])))
    for k, (e, i) in enumerate(zip(ev.itertuples(index=False), info)):
        event_rows.append(dict(stim_id=s.stim_id, event=e.event, word=i['word'], token=i['token'],
                               actual_onset_ms=round(onset_ms[k], 3), actual_offset_ms=round(offset_ms[k], 3)))
    print(f'{s.stim_id:22s} {s.words:32s} {len(y) / FS:.2f} s, smallest gap {min_gap:.1f} ms')

# %% 7 SAVE (temporary folder, then rename)
os.makedirs(os.path.join(tmp_dir, 'wav'))
max_len = max(map(len, built))
store = np.zeros((len(built), max_len), np.float32)
for r, y in enumerate(built):
    store[r, :len(y)] = y
    sf.write(os.path.join(tmp_dir, 'wav', f'{stims.stim_id[r]}.wav'), y, FS)
np.save(os.path.join(tmp_dir, 'training_stimuli.npy'), store)
design.merge(pd.DataFrame(stim_rows), on='stim_id', validate='many_to_one').to_csv(
    os.path.join(tmp_dir, 'design_stim.csv'), index=False)
ev_stim = events.merge(pd.DataFrame(event_rows), on=['stim_id', 'event'], validate='one_to_one')
ev_stim['token'] = ev_stim['token'].astype('Int64')
ev_stim.to_csv(os.path.join(tmp_dir, 'events_stim.csv'), index=False)
meta = dict(built=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            design_table=os.path.abspath(DESIGN_CSV), design_table_md5=md5(DESIGN_CSV),
            events_table=os.path.abspath(EVENTS_CSV), events_table_md5=md5(EVENTS_CSV),
            noise_pool=os.path.abspath(NOISE_DIR), noise_pool_md5=md5(os.path.join(NOISE_DIR, 'noise_pool_exact.json')),
            shared_gain=GAIN, token_seed=TOKEN_SEED, fs=FS, ioi_ms=IOI_MS, n_events=N_EVENTS, lead_ms=LEAD_MS,
            tail_ms=TAIL_MS, edge_fade_ms=EDGE_FADE_MS, onset_db=ONSET_DB, env_win_ms=ENV_WIN_MS,
            min_gap_ms=MIN_GAP_MS, n_stimuli=len(built), max_samples=int(max_len), dtype='float32',
            words_files={os.path.basename(p): md5(p) for p in words})
with open(os.path.join(tmp_dir, 'training_stim_meta.json'), 'w') as fh:
    json.dump(meta, fh, indent=2)
os.rename(tmp_dir, OUT_DIR)
print(f'\n✓ {len(built)} training sounds -> {OUT_DIR}/')