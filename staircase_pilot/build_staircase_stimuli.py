# %% BUILD STAIRCASE STIMULI FOR ONE PARTICIPANT
"""
Prepares the sounds of ONE participant's staircase design table and saves everything in that participant's
own folder. Nothing is ever overwritten: if the folder exists, the script stops (delete/rename it to rebuild).

Why no ready-made trial sounds (unlike detection): the shift size of a trial is only known during the session
(the staircase sets it). So this script
    1. saves every word the participant will hear (the "word bank"),
    2. builds EVERY trial at EVERY size on the grid (params.SIZE_LEVELS_MS) with the same function the run
       script uses (staircase_functions.build_trial) and checks each one,
so any size the staircase can ever reach has been built and checked before the participant arrives.

    stimuli/<pid>_<group>/staircase_words.npy        float32 (n_words, max_samples), zero-padded
    stimuli/<pid>_<group>/staircase_words.csv        one row per word: bank_row, word, sws_file, n_samples, on, off
    stimuli/<pid>_<group>/design_stim.csv            the design table + w1_row ... w6_row + words
                                                     (this is what the run script reads)
    stimuli/<pid>_<group>/staircase_check.csv        one row per trial x size: smallest gap, duration
    stimuli/<pid>_<group>/staircase_stim_meta.json   settings, size grid, participant, source table, build time
    stimuli/<pid>_<group>/examples/*.wav             a few trials at a large and a small size, only for listening

Building one trial in the run script:
    words = sf.load_word_bank(params, folder)
    y, on, off = sf.build_trial(sf.trial_words(words, row, params), row.interval, shift_ms,
                                row.shifted_presentation, params)
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
sys.path.insert(0, os.path.join(PROJECT_DIR, 'staircase_pilot'))
import staircase_functions as stf

# %% SETTINGS
SETUP          = 'laptop'     # 'lab' or 'laptop' (only the paths differ; stimulus settings are the same)
PARTICIPANT_ID = None         # e.g. 'P01'; None = ask
GROUP          = None         # 'speech' / 'acoustic'; None = ask
N_EXAMPLE_WAVS = 4            # trials also written as wav, each at the largest and at a small size; 0 = none
EXAMPLE_SMALL_MS = 20         # the "small" example size = grid size closest to this

params = stf.load_params(SETUP)
FS, N_WORDS = params.FS, params.N_WORDS
LEVELS = list(params.SIZE_LEVELS_MS)

# %% 0 PARTICIPANT, TABLE, OUTPUT FOLDER
pid = PARTICIPANT_ID or input('Enter participant ID: ').strip().upper()
group = GROUP or input('Enter experimental group (speech / acoustic): ').strip().lower()
design_path = os.path.join(params.DESIGN_DIR, f'staircase_{pid}_{group}.csv')
if not os.path.exists(design_path):
    raise FileNotFoundError(f'no design table {design_path} - run staircase_design_table.py first')
design = pd.read_csv(design_path).sort_values('trial').reset_index(drop=True)
if (design['participant'] != pid).any() or (design['group'] != group).any():
    raise ValueError(f'{design_path}: participant/group inside the table do not match {pid}/{group}')
if sorted(design['staircase'].unique()) != sorted(params.STAIRCASES):
    raise ValueError(f'{design_path}: staircases {sorted(design["staircase"].unique())} != params.STAIRCASES')
per_block = design.groupby('block').size()
if list(per_block.index) != list(range(1, params.N_BLOCKS + 1)) or (per_block != params.TRIALS_PER_BLOCK).any():
    raise ValueError(f'{design_path}: trials per block {per_block.to_dict()}, expected '
                     f'{params.TRIALS_PER_BLOCK} in each of {params.N_BLOCKS} blocks')

out_dir = os.path.join(params.STIM_DIR, params.PARTICIPANT_FOLDER.format(pid=pid, group=group))
if os.path.exists(out_dir):
    raise FileExistsError(f'{out_dir} already exists - not overwriting. Delete or rename it to rebuild.')
tmp_dir = out_dir + '_incomplete'            # built here, renamed to out_dir only when everything passed
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)                   # leftover of an interrupted run of this script
print(f'{pid} ({group}): {len(design)} trials from {design_path}')
print(f'size grid ({len(LEVELS)} sizes): {LEVELS}')

# %% 1 SENTENCES -> WORDS (same mapping as build_detection_stimuli.py: S24_k = k-th item of items_2_4)
items = pd.read_csv(params.ITEMS_CSV)
sws_info = pd.read_csv(params.SWS_INFO_CSV)
items = items.merge(sws_info[['base', 'sws_file']], on='base', how='left')
if items['sws_file'].isna().any():
    missing = sorted(items.loc[items['sws_file'].isna(), 'word'].unique())
    raise FileNotFoundError(f'no SWS file for: {missing}')

sentences = {}                                         # sentence_id -> list of 6 rows (word, sws_file)
for lst, code in params.LIST_TO_TYPE.items():
    d = items[items['list'] == lst]
    if d.empty:
        raise ValueError(f'list {lst!r} not in {params.ITEMS_CSV} (lists there: {sorted(items["list"].unique())})')
    for k, item in enumerate(sorted(d['item'].unique()), start=1):
        w = d[d['item'] == item].sort_values('pos')
        if len(w) != N_WORDS:
            raise ValueError(f'{lst} item {item}: {len(w)} words, expected {N_WORDS}')
        sentences[f'{code}_{k:02d}'] = w[['word', 'sws_file']].to_dict('records')
for sid in [s for s in sentences if s.startswith(params.WL_FROM + '_')]:     # PLACEHOLDER word list
    sentences['WL_' + sid.split('_')[1]] = sentences[sid]

needed = sorted(design['sentence_id'].unique())
missing = [s for s in needed if s not in sentences]
if missing:
    raise ValueError(f'design table uses items that do not exist in {params.ITEMS_CSV}: {missing} '
                     f'(N_ITEMS in the design table larger than the item lists?)')
print(f'{len(needed)} items used by the design table')

# %% 2 WORD BANK: every word once (+ its onset / offset with the trimming rule)
audio = stf.repo_module(params, 'audio')
paths = sorted({w['sws_file'] for s in needed for w in sentences[s]})
bank_rows, ys = [], []
for r, path in enumerate(paths):
    y, fs = audio.load_audio(os.path.join(params.MAIN_DIR, path))
    if fs != FS:
        raise ValueError(f'{path}: {fs} Hz, expected {FS}')
    on, off = audio.find_on_offset(y, fs, params.ONSET_DB, params.ENV_WIN_MS)
    word = next(w['word'] for s in needed for w in sentences[s] if w['sws_file'] == path)
    bank_rows.append(dict(bank_row=r, word=word, sws_file=path, n_samples=len(y), on=on, off=off,
                          sound_ms=round((off - on) / fs * 1000, 2)))
    ys.append(y.astype(np.float32))
bank_index = pd.DataFrame(bank_rows)
row_of = dict(zip(bank_index['sws_file'], bank_index['bank_row']))
print(f'{len(paths)} words, audible {bank_index.sound_ms.min():.0f}-{bank_index.sound_ms.max():.0f} ms '
      f'(median {bank_index.sound_ms.median():.0f})')
limit = params.IOI_MS - params.START_MS - params.MIN_GAP_MS
too_long = bank_index.loc[bank_index.sound_ms > limit, 'word'].tolist()
if too_long:
    print(f'WARNING longer than {limit:.0f} ms (will fail the gap check at the largest advance):', ', '.join(too_long))

os.makedirs(tmp_dir)
store = np.zeros((len(ys), max(map(len, ys))), np.float32)
for r, y in enumerate(ys):
    store[r, :len(y)] = y
np.save(os.path.join(tmp_dir, params.WORD_NPY), store)
bank_index.to_csv(os.path.join(tmp_dir, params.WORD_INDEX), index=False)

# design table + the bank rows of the 6 words of each trial
design_stim = design.copy()
for k in range(N_WORDS):
    design_stim[f'w{k + 1}_row'] = [row_of[sentences[s][k]['sws_file']] for s in design['sentence_id']]
design_stim['words'] = [' '.join(w['word'] for w in sentences[s]) for s in design['sentence_id']]
design_stim.to_csv(os.path.join(tmp_dir, params.STIM_DESIGN), index=False)

# %% 3 BUILD AND CHECK EVERY TRIAL AT EVERY SIZE (read back exactly the way the run script reads it)
words = stf.load_word_bank(params, tmp_dir)
ds = pd.read_csv(os.path.join(tmp_dir, params.STIM_DESIGN))
sign = params.DIRECTION_SIGN
checks = []
for i, row in enumerate(ds.itertuples()):
    ws = stf.trial_words(words, row, params)
    for size in LEVELS:
        shift = sign[row.direction] * size
        name = stf.stim_id(row.sentence_id, row.interval, shift, row.shifted_presentation)
        y, on, off = stf.build_trial(ws, row.interval, shift, row.shifted_presentation, params)
        min_gap = stf.check_trial(y, on, off, row.interval, shift, row.shifted_presentation, params, name)
        checks.append(dict(trial=row.trial, size_ms=size, shift_ms=shift, stim_id=name, min_gap_ms=round(min_gap, 2),
                           dur_ms=round(len(y) / FS * 1000, 2), peak=round(float(np.abs(y).max()), 4)))
    if (i + 1) % 20 == 0:
        print(f'{i + 1}/{len(ds)} trials checked at all {len(LEVELS)} sizes')
checks = pd.DataFrame(checks)
checks.to_csv(os.path.join(tmp_dir, 'staircase_check.csv'), index=False)

# %% 4 META + EXAMPLES, then rename the folder
json.dump(dict(participant=pid, group=group, built=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               setup=SETUP, design_table=os.path.abspath(design_path),
               design_table_md5=hashlib.md5(open(design_path, 'rb').read()).hexdigest(),
               fs=FS, ioi_ms=params.IOI_MS, n_words=N_WORDS, isi_ms=params.ISI_MS,
               lead_ms=params.LEAD_MS, tail_ms=params.TAIL_MS, edge_fade_ms=params.EDGE_FADE_MS,
               onset_db=params.ONSET_DB, env_win_ms=params.ENV_WIN_MS, min_gap_ms=params.MIN_GAP_MS,
               size_levels_ms=LEVELS, start_ms=params.START_MS, grid_factor=params.GRID_FACTOR,
               list_to_type=params.LIST_TO_TYPE, wl_placeholder_from=params.WL_FROM,
               n_trials=len(ds), n_words_in_bank=len(paths), n_checked=len(checks),
               smallest_gap_ms=float(checks.min_gap_ms.min()), largest_peak=float(checks.peak.max()),
               words_files={w['word']: w['sws_file'] for s in needed for w in sentences[s]}),
          open(os.path.join(tmp_dir, params.STIM_META), 'w'), indent=2)

if N_EXAMPLE_WAVS:
    ex_dir = os.path.join(tmp_dir, 'examples')
    os.makedirs(ex_dir)
    small = min(LEVELS, key=lambda s: abs(s - EXAMPLE_SMALL_MS))
    for row in ds.head(N_EXAMPLE_WAVS).itertuples():
        for size in (LEVELS[0], small):
            shift = sign[row.direction] * size
            y, _, _ = stf.build_trial(stf.trial_words(words, row, params), row.interval, shift,
                                      row.shifted_presentation, params)
            sf.write(os.path.join(ex_dir, f'trial{row.trial:03d}_'
                                          f'{stf.stim_id(row.sentence_id, row.interval, shift, row.shifted_presentation)}.wav'),
                     y, FS)

os.rename(tmp_dir, out_dir)
worst = checks.loc[checks.min_gap_ms.idxmin()]
print(f'\n✓ {len(ds)} trials x {len(LEVELS)} sizes = {len(checks)} sounds built and checked -> {out_dir}/')
print(f'  word bank {store.nbytes / 1e6:.1f} MB; trial duration {checks.dur_ms.min():.0f}-{checks.dur_ms.max():.0f} ms; '
      f'smallest gap {worst.min_gap_ms:.1f} ms ({worst.stim_id}); largest peak {checks.peak.max():.3f}')