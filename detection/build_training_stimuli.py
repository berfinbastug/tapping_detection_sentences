# %% BUILD TRAINING STIMULI
"""
Stimuli for the training design table (demo + practice), built exactly like the main stimuli
(same grid, lead/tail silence, onset rule and checks as build_detection_stimuli.py).

Words are read directly from training_items.csv and looked up as SWS files:
    word -> base name (sws_words/words.csv, from sentence_lists_to_tts.py) -> SWS file (sws_words/sws_info.csv)
items.csv is NOT needed. A word that was never synthesized is reported by name; then add it to a list
(or put training_items.csv into sentence_lists/) and run the TTS -> trim -> SWS scripts.

Output (OUT_DIR = detection/training_stimuli/):
    training_stimuli.npy        float32 (n_stimuli, max_samples), zero-padded; every stimulus of the
                                table + the unshifted version of every item
    training_stim_index.csv     row, stim_id, n_samples, word onsets/offsets (ms)
    training_stim_meta.json
    wav/<stim_id>.wav           every training stimulus, for listening
    design_tables_stim/training_design.csv   the training table + stim_row + n_samples
"""
import sys
import json

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
TRAINING_CSV  = 'sentence_lists/training_items.csv'
WORDS_CSV     = 'sws_words/words.csv'         # word -> base (from sentence_lists_to_tts.py)
SWS_INFO_CSV  = 'sws_words/sws_info.csv'      # base -> SWS file (from make_sws_words_trimmed.py)
LOWER_ITEM_INITIAL = True                     # same rule as sentence_lists_to_tts.py
DESIGN_CSV    = 'detection/design_tables/training_design.csv'
DESIGN_OUT    = 'detection/design_tables_stim/training_design.csv'
OUT_DIR       = 'detection/training_stimuli'

# must match build_detection_stimuli.py
N_WORDS, IOI_MS, FS = 6, 450, 44100
LEAD_MS, TAIL_MS, EDGE_FADE_MS = 50, 50, 2
ONSET_DB, ENV_WIN_MS = -40, 5
MIN_GAP_MS = 5

# %% WORDS OF EVERY TRAINING ITEM
word_base = pd.read_csv(WORDS_CSV).set_index('word')['base']
base_file = pd.read_csv(SWS_INFO_CSV).set_index('base')['sws_file']
train = pd.read_csv(TRAINING_CSV)
slots = [f'slot{k}' for k in range(1, N_WORDS + 1)]

sentences, missing = {}, set()
for _, r in train.iterrows():
    ws = [str(r[c]).strip() for c in slots]
    if LOWER_ITEM_INITIAL:
        ws[0] = ws[0].lower()
    for w in ws:
        if w not in word_base.index or word_base[w] not in base_file.index:
            missing.add(w)
    sentences[r['item_id']] = [dict(word=w, sws_file=base_file.get(word_base.get(w))) for w in ws]
if missing:
    raise FileNotFoundError(f'no SWS word yet for: {sorted(missing)} - add these words to a sentence list '
                            f'and run sentence_lists_to_tts.py -> trim_tts_words.py -> make_sws_words_trimmed.py')
for it, ws in sentences.items():
    print(f"{it:11s} {' '.join(w['word'] for w in ws)}")

words = {}
for path in sorted({w['sws_file'] for s in sentences.values() for w in s}):
    y, fs = load_audio(path)
    assert fs == FS, f'{path}: {fs} Hz'
    on, off = find_on_offset(y, fs, ONSET_DB, ENV_WIN_MS)
    words[path] = dict(y=y.astype(np.float64), on=on, off=off)

# %% BUILD + CHECK ONE STIMULUS
def build_one(item, interval, shift_ms):
    ws = [words[w['sws_file']] for w in sentences[item]]
    events = [dict(y=w['y'], on=w['on'], off=w['off']) for w in ws]
    events[-1]['off'] = len(ws[-1]['y']) - 1                      # keep word 6's own fade-out
    shifts = np.full(N_WORDS, float(LEAD_MS))
    if isinstance(interval, str):
        shifts[int(interval[1:]):] += shift_ms                    # phase shift: words k+1 ... 6
    y, table = build_sequence(events, FS, soa_ms=IOI_MS, shifts_ms=shifts, end='offset', edge_fade_ms=EDGE_FADE_MS)
    y = np.r_[y, np.zeros(int(round(TAIL_MS / 1000 * FS)), np.float32)]
    on_ms = np.array([t['onset_ms'] for t in table])
    off_ms = on_ms + np.array([(w['off'] - w['on']) / FS * 1000 for w in ws])

    want = np.full(N_WORDS - 1, float(IOI_MS))
    if isinstance(interval, str):
        want[int(interval[1:]) - 1] += shift_ms
    assert np.all(np.abs(np.diff(on_ms) - want) <= 1000 / FS), f'{item} {interval} {shift_ms}: wrong IOIs'
    gaps = on_ms[1:] - off_ms[:-1]
    assert gaps.min() >= MIN_GAP_MS, f'{item} {interval} {shift_ms:+}: gap of {gaps.min():.1f} ms < {MIN_GAP_MS} ms'
    assert np.abs(y).max() < 1, f'{item}: clips'
    return y, on_ms, off_ms, gaps.min()

# %% BUILD ALL STIMULI OF THE TABLE
design = pd.read_csv(DESIGN_CSV)
uniq = design.drop_duplicates('stim_id')[['stim_id', 'item_id', 'interval', 'shift_ms']]
# also the unshifted version of every item (reference for plots / examples), if not in the table
extra = [dict(stim_id=f'{it}_none', item_id=it, interval=None, shift_ms=0)
         for it in sentences if f'{it}_none' not in set(uniq['stim_id'])]
uniq = pd.concat([uniq, pd.DataFrame(extra)], ignore_index=True)
built, rows = [], []
for r, d in enumerate(uniq.itertuples()):
    try:
        y, on_ms, off_ms, gap = build_one(d.item_id, d.interval, d.shift_ms)
    except ValueError as e:                                       # build_sequence: words overlap
        raise ValueError(f'{d.stim_id}: {e} - this shift does not fit with these words') from None
    built.append(y)
    row = dict(row=r, stim_id=d.stim_id, item_id=d.item_id, interval=d.interval, shift_ms=d.shift_ms,
               n_samples=len(y), dur_ms=round(len(y) / FS * 1000, 2), min_gap_ms=round(gap, 2),
               words=' '.join(w['word'] for w in sentences[d.item_id]))
    row |= {f'on{k + 1}_ms': round(v, 3) for k, v in enumerate(on_ms)}
    row |= {f'off{k + 1}_ms': round(v, 3) for k, v in enumerate(off_ms)}
    rows.append(row)

os.makedirs(os.path.join(OUT_DIR, 'wav'), exist_ok=True)
store = np.zeros((len(built), max(map(len, built))), np.float32)
for r, y in enumerate(built):
    store[r, :len(y)] = y
    sf.write(os.path.join(OUT_DIR, 'wav', rows[r]['stim_id'] + '.wav'), y, FS)
np.save(os.path.join(OUT_DIR, 'training_stimuli.npy'), store)
index = pd.DataFrame(rows)
index.to_csv(os.path.join(OUT_DIR, 'training_stim_index.csv'), index=False)
json.dump(dict(fs=FS, ioi_ms=IOI_MS, n_words=N_WORDS, lead_ms=LEAD_MS, tail_ms=TAIL_MS, onset_db=ONSET_DB,
               env_win_ms=ENV_WIN_MS, min_gap_ms=MIN_GAP_MS, n_stimuli=len(index), dtype='float32'),
          open(os.path.join(OUT_DIR, 'training_stim_meta.json'), 'w'), indent=2)

# %% DESIGN TABLE WITH stim_row
lookup = index.set_index('stim_id')
design['stim_row'] = lookup.loc[design['stim_id'], 'row'].values
design['n_samples'] = lookup.loc[design['stim_id'], 'n_samples'].values
os.makedirs(os.path.dirname(DESIGN_OUT), exist_ok=True)
design.to_csv(DESIGN_OUT, index=False)

print(f'✓ {len(index)} training stimuli -> {OUT_DIR}/ (npy + wav); smallest gap {index.min_gap_ms.min():.1f} ms')
print(index[['row', 'stim_id', 'words', 'dur_ms', 'min_gap_ms']].to_string(index=False))
print(f'✓ {DESIGN_OUT}')





# # %% CHECK THE TRAINING STIMULI BY EYE
# """
# Plots EVERY trial of the training table (demo + practice) from training_stimuli/ and checks the shifts.
# Same method as plot_detection_stimuli.py: word onsets are measured again from the signal itself
# (-40 dB re each word's peak, 5 ms RMS envelope) and compared with the regular 450 ms grid.

#     grey   = the unshifted version of the same item (the reference)
#     colour = the stimulus; shaded words = the words that should move
#     dashed = the regular grid;  dots = measured onsets
#     under each word: measured shift in ms (green = as expected, red = wrong)
# """
# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # %% SETTINGS
# DESIGN_CSV = 'detection/design_tables_stim/training_design.csv'
# STIM_DIR   = 'detection/training_stimuli'
# TOL_MS     = 0.5
# ONSET_DB, ENV_WIN_MS = -40, 5

# # %% LOAD
# meta  = json.load(open(f'{STIM_DIR}/training_stim_meta.json'))
# FS, IOI, LEAD, NW = meta['fs'], meta['ioi_ms'], meta['lead_ms'], meta['n_words']
# stims = np.load(f'{STIM_DIR}/training_stimuli.npy', mmap_mode='r')
# index = pd.read_csv(f'{STIM_DIR}/training_stim_index.csv').set_index('stim_id')
# design = pd.read_csv(DESIGN_CSV)
# design['sentence_id'] = design['item_id']             # the reference is <item_id>_none

# def get(stim_id):
#     r = index.loc[stim_id]
#     return np.array(stims[r['row'], :r['n_samples']], dtype=float)


# def env_db(y, win_ms=ENV_WIN_MS):
#     n = max(1, int(round(win_ms / 1000 * FS)))
#     return np.sqrt(np.convolve(y ** 2, np.ones(n) / n, mode='same'))


# MAX_WORD_SPAN_MS = 600   # pieces of ONE word span less than this; two neighbouring words span more
#                          # (words ~350 ms, onsets >= IOI - 50 = 400 ms apart -> two words >= ~750 ms)


# def measure_onsets(y, seg_db=-30, stim=''):
#     """Onset (ms) of each word, found WITHOUT assuming where the words should be.

#     1 find the separate sounds: envelope above (overall peak + seg_db).
#     2 a word with a silent stretch inside (voicing mask) gives 2+ pieces: repeatedly merge the
#       neighbouring pair with the SHORTEST combined span until NW sounds remain. A merge spanning
#       more than MAX_WORD_SPAN_MS would join two different words -> error.
#     3 refine each onset with the trimming rule (peak of THIS word + ONSET_DB): walk back from the
#       start of the sound until the envelope falls below that threshold.
#     """
#     env = env_db(y)
#     above = env > env.max() * 10 ** (seg_db / 20)
#     edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
#     starts, ends = list(edges[::2]), list(edges[1::2])
#     while len(starts) > NW:
#         spans = [ends[k + 1] - starts[k] for k in range(len(starts) - 1)]
#         k = int(np.argmin(spans))
#         if spans[k] / FS * 1000 > MAX_WORD_SPAN_MS:
#             raise ValueError(f'{stim}: {len(starts)} sounds and no two of them fit in one word')
#         ends[k] = ends.pop(k + 1); starts.pop(k + 1)
#     if len(starts) != NW:
#         raise ValueError(f'{stim}: found {len(starts)} separate sounds, expected {NW}')
#     out = []
#     for s, e in zip(starts, ends):
#         thr = env[s:e].max() * 10 ** (ONSET_DB / 20)
#         i = s
#         while i > 0 and env[i - 1] > thr:
#             i -= 1
#         out.append(i / FS * 1000)
#     return np.array(out)


# def expected_shifts(interval, shift_ms):
#     v = np.zeros(NW)
#     if isinstance(interval, str):
#         v[int(interval[1:]):] = shift_ms
#     return v


# grid_ms = LEAD + np.arange(NW) * IOI


# # %% FIGURE - EVERY TRAINING TRIAL
# missing_ref = sorted({f"{s}_none" for s in design.sentence_id} - set(index.index))
# if missing_ref:
#     print('No unshifted reference in the training set for:', missing_ref, '(grey trace left empty)')
# pick = design
# fig, axes = plt.subplots(len(pick), 1, figsize=(13, 2.3 * len(pick)), squeeze=False)
# for ax, (_, row) in zip(axes[:, 0], pick.iterrows()):
#     y   = get(row['stim_id'])
#     ref = get(f"{row['sentence_id']}_none") if f"{row['sentence_id']}_none" in index.index else np.zeros(1)
#     t, t_ref = np.arange(len(y)) / FS * 1000, np.arange(len(ref)) / FS * 1000
#     on = measure_onsets(y, stim=row['stim_id'])
#     meas, want = on - grid_ms, expected_shifts(row['interval'], row['shift_ms'])
#     ok = np.abs(meas - want) <= TOL_MS

#     ax.plot(t_ref, ref - 1.1, color='0.65', lw=0.4)                     # reference, below
#     ax.plot(t, y + 1.1, color='tab:blue', lw=0.4)                        # stimulus, above
#     for k in range(NW):
#         if want[k] != 0:
#             ax.axvspan(on[k], index.loc[row['stim_id'], f'off{k + 1}_ms'], ymin=0.5, ymax=1,
#                        color='tab:orange', alpha=0.15, lw=0)
#         ax.axvline(grid_ms[k], color='k', ls='--', lw=0.7)
#         ax.plot(on[k], 1.1, 'o', ms=4, color='tab:red' if not ok[k] else 'tab:green', zorder=5)
#         ax.text(grid_ms[k] + 175, -2.45, f'{meas[k]:+.1f}', ha='center', fontsize=8,
#                 color='tab:green' if ok[k] else 'tab:red', fontweight='bold')
#     words = index.loc[row['stim_id'], 'words'].split()
#     for k, w in enumerate(words):
#         ax.text(grid_ms[k] + 175, 2.25, w, ha='center', fontsize=8, color='0.3')
#     status = 'OK' if ok.all() else 'WRONG'
#     ax.set_title(f"{row['phase']} {row['trial']} | {row['stim_id']} | {row['role']} | expected: "
#                  + (f"{row['shift_ms']:+.0f} ms from word {int(row['interval'][1:]) + 1} on"
#                     if isinstance(row['interval'], str) else 'no shift')
#                  + f"  ->  {status}", loc='left', fontsize=9,
#                  color='tab:green' if ok.all() else 'tab:red')
#     ax.set(ylim=(-2.7, 2.6), xlim=(0, t.max() + 20), yticks=[-1.1, 1.1])
#     ax.set_yticklabels(['unshifted', 'stimulus'], fontsize=8)
#     for s in ('top', 'right'):
#         ax.spines[s].set_visible(False)
# axes[-1, 0].set_xlabel('Time (ms)   [dashed = regular grid, numbers = measured shift of each word in ms]')
# plt.tight_layout()
# plt.show()


# # %% SUMMARY
# rows = []
# for _, row in design.iterrows():
#     meas = measure_onsets(get(row['stim_id']), stim=row['stim_id']) - grid_ms
#     want = expected_shifts(row['interval'], row['shift_ms'])
#     rows.append(dict(phase=row['phase'], trial=row['trial'], stim_id=row['stim_id'],
#                      expected=' '.join(f'{v:+.0f}' for v in want), measured=' '.join(f'{v:+.1f}' for v in meas),
#                      max_error_ms=round(float(np.abs(meas - want).max()), 3),
#                      min_gap_ms=index.loc[row['stim_id'], 'min_gap_ms']))
# chk = pd.DataFrame(rows)
# print(chk.to_string(index=False))
# print('✓ every word of every training trial is where it should be' if (chk.max_error_ms <= TOL_MS).all()
#       else f'✗ {(chk.max_error_ms > TOL_MS).sum()} trials with a wrong shift')