# %%
# settings
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from IPython.display import Audio, display

REPO_DIR = "Users/bastugb/Desktop/tapping_detection_sentences/"    # folder that CONTAINS the sws_stimuli repo
if REPO_DIR is not None:
    sys.path.insert(0, REPO_DIR)
from repo import (make_noise_pool, save_token_pool, build_sequence, draw_token_order,
                         load_audio, save_audio, find_on_offset, rms_envelope)



# %%
LIST_NAME   = 'items_2_4'                  # the sentence list (file name without .csv)
ITEMS_CSV   = 'sws_words/items.csv'        # word -> file table from sentence_lists_to_tts.ipynb
SWS_DIR     = 'sws_words/3_sws'
WORD_SUFFIX = '_sws50ms_dur350ms.wav'      # which SWS version of the words
NOISE_DIR   = 'sws_words/noise_pool'
OUT_DIR     = 'trials'

REPO_DIR = None    # folder that CONTAINS the sws_stimuli repo; None if installed with  pip install -e .

# ----- trial
SOA_MS    = 450                            # onset-to-onset interval of ALL events
STRUCTURE = ['noise', 'sentence', 'noise', 'sentence', 'noise']
N_NOISE   = 3                              # bursts per noise block
ITEM      = None                           # None = random sentence; or an item number, e.g. 7
SEED      = None                           # None = different every run; a number = same sentence and tokens every run
ONSET_THRESH_DB = -40                      # an event's onset = envelope first above (peak + this); this point goes on the grid
ENV_WIN_MS      = 5
EDGE_FADE_MS    = 1

# ----- noise pool (same values as sentence_to_sws_noise_blocks_v2.ipynb)
CARRIER          = 'noise'  # 'noise' or 'harmonic' (a steady buzz with one pitch)
F0_HZ            = 150         # pitch of the harmonic tone (only for 'harmonic')
PHASES           = 'schroeder' # 'schroeder' = flat envelope, low peaks (only for 'harmonic')
NOISE_DUR_MS     = None        # None = mean audible duration of the words
NOISE_SPECTRUM   = 'shaped'    # 'shaped' or 'white'
NOISE_SMOOTH_OCT = 3
NOISE_BAND_HZ    = (150, 5000)
NOISE_FIR_TAPS   = 2049
NOISE_ENV        = 'word'      # 'word' or 'ramp'
NOISE_ENV_SMOOTH_MS = 5
NOISE_RAMP_MS    = 3           # only used if MATCH_RISE = False
MATCH_RISE       = True
RISE_DB          = -6
RISE_WIN_MS      = 5
RAMP_SEARCH_MS   = (0.5, 25)
RAMP_STEP_MS     = 0.25
NOISE_LEVEL      = 'match'
NOISE_LEVEL_DB   = 0.0
NOISE_MAX_PEAK   = 0.95
N_TOKENS         = 20
NOISE_SEED       = 1

# %%
# noise pool
pool = make_noise_pool(os.path.join(SWS_DIR, '*' + WORD_SUFFIX),
                       fs=44100, onset_thresh_db=ONSET_THRESH_DB, env_win_ms=ENV_WIN_MS,
                       carrier=CARRIER, f0_hz=F0_HZ if CARRIER == 'harmonic' else None,
                       phases=PHASES if CARRIER == 'harmonic' else None,
                       duration_ms=NOISE_DUR_MS, spectrum=NOISE_SPECTRUM, smooth_oct=NOISE_SMOOTH_OCT,
                       band_hz=NOISE_BAND_HZ, fir_taps=NOISE_FIR_TAPS, envelope=NOISE_ENV,
                       env_smooth_ms=NOISE_ENV_SMOOTH_MS, ramp_ms=NOISE_RAMP_MS, match_rise=MATCH_RISE,
                       rise_db=RISE_DB, rise_win_ms=RISE_WIN_MS, ramp_search_ms=RAMP_SEARCH_MS,
                       ramp_step_ms=RAMP_STEP_MS, level=NOISE_LEVEL, level_db=NOISE_LEVEL_DB,
                       max_peak=NOISE_MAX_PEAK, n_tokens=N_TOKENS, seed=NOISE_SEED)
save_token_pool(pool, NOISE_DIR)

info = pool['info']
print(f"{CARRIER} from {info['n_reference_words']} words: {info['n_tokens']} tokens of {info['duration_ms']:.0f} ms")
print(f"rise to {RISE_DB} dB: words {info['word_rise_ms_mean']:.1f} ms, noise {info['noise_rise_ms']:.1f} ms "
      f"(ramp {info['ramp_ms']} ms)  {info['rise_note']}")
print(f"shared gain {info['shared_gain_db']:+.1f} dB, noise RMS - word RMS = {info['rms_noise_minus_words_db']:+.2f} dB")
print(f'saved in {NOISE_DIR}/')
display(Audio(pool['tokens'][0], rate=44100, normalize=False))


# %%
# pick a sentence
rng = np.random.default_rng(SEED)

# --- the sentence
items = pd.read_csv(ITEMS_CSV)
items = items[items['list'] == LIST_NAME]
if items.empty:
    raise ValueError(f'no items of list {LIST_NAME} in {ITEMS_CSV}')
item = int(rng.choice(items['item'].unique())) if ITEM is None else int(ITEM)
sent = items[items['item'] == item].sort_values('pos')
if sent.empty:
    raise ValueError(f'item {item} is not in {LIST_NAME}')
print(f"{LIST_NAME}, item {item}: {' '.join(sent['word'])}")

# --- its words, with the shared gain, and their onsets
fs = pool['info']['fs']
words = []
for _, r in sent.iterrows():
    path = os.path.join(SWS_DIR, r['base'] + WORD_SUFFIX)
    y, fs_w = load_audio(path)
    if fs_w != fs:
        raise ValueError(f'{path}: {fs_w} Hz, noise pool is {fs} Hz')
    on, off = find_on_offset(y, fs, ONSET_THRESH_DB, ENV_WIN_MS)
    words.append(dict(y=y * pool['gain'], on=on, off=off, word=r['word'], file=os.path.basename(path)))

# --- the events, block by block
n_noise_total = STRUCTURE.count('noise') * N_NOISE
token_order = iter(draw_token_order(n_noise_total, len(pool['tokens']), rng))
events, counts = [], {'noise': 0, 'sentence': 0}
for b, kind in enumerate(STRUCTURE, start=1):
    counts[kind] += 1
    label = f'{kind}{counts[kind]}'
    if kind == 'noise':
        for j in range(N_NOISE):
            k = next(token_order)
            events.append(dict(y=pool['tokens'][k], on=pool['onset'], off=pool['offset'], kind='noise',
                               word='', token=k + 1, block=b, block_label=label, pos_in_block=j + 1))
    else:
        for j, w in enumerate(words):
            events.append(dict(w, kind='word', token=None, block=b, block_label=label, pos_in_block=j + 1))

# --- on the grid
seq, table = build_sequence(events, fs, soa_ms=SOA_MS, shifts_ms=None, end='offset', edge_fade_ms=EDGE_FADE_MS)
seq_tab = pd.DataFrame(table)

os.makedirs(OUT_DIR, exist_ok=True)
name = f'{LIST_NAME}_item{item:02d}'
save_audio(os.path.join(OUT_DIR, name + '.wav'), seq, fs)
seq_tab.to_csv(os.path.join(OUT_DIR, name + '_events.csv'), index=False)

print(f"{len(seq_tab)} events ({(seq_tab['kind'] == 'word').sum()} words, {(seq_tab['kind'] == 'noise').sum()} noise), "
      f"{len(seq) / fs:.2f} s, peak {np.abs(seq).max():.2f} -> {OUT_DIR}/{name}.wav")
display(Audio(seq, rate=fs, normalize=False))


# %%
# check
t = np.arange(len(seq)) / fs
fig, axes = plt.subplots(3, 1, figsize=(13, 8), gridspec_kw=dict(height_ratios=[1, 1.3, 0.9]))

ax = axes[0]
ax.plot(t, seq, lw=0.2, color='k')
for label, blk in seq_tab.groupby('block_label', sort=False):
    is_noise = label.startswith('noise')
    ax.axvspan(blk['onset_ms'].min() / 1000, blk['offset_ms'].max() / 1000,
               color='0.8' if is_noise else 'tab:blue', alpha=0.35 if is_noise else 0.15, lw=0)
    ax.text(blk['onset_ms'].mean() / 1000, 1.02, label, transform=ax.get_xaxis_transform(), ha='center', fontsize=9)
for _, r in seq_tab.iterrows():
    ax.axvline(r['onset_ms'] / 1000, color='tab:red', lw=0.5, alpha=0.6)
    if r['kind'] == 'word':
        ax.text(r['onset_ms'] / 1000, 0.92, r['word'], transform=ax.get_xaxis_transform(), fontsize=7)
ax.set(xlim=(0, t[-1]), yticks=[])
ax.set_title(f"{name}: {' '.join(sent['word'])}  |  SOA {SOA_MS} ms", loc='left', pad=18)

ax = axes[1]
nper = int(0.02 * fs); hop = int(0.005 * fs)
f, ts, Z = stft(seq, fs=fs, window='hann', nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
S = 20 * np.log10(np.abs(Z) + 1e-12); S -= S.max()
ax.pcolormesh(ts, f[f <= 5000], S[f <= 5000], shading='auto', cmap='magma', vmin=-70, vmax=0)
ax.set(xlim=(0, t[-1]), ylabel='Freq (Hz)', xlabel='Time (s)')

# independent check: rising threshold crossings in the saved file, at least 50 ms after the last fall
env = rms_envelope(seq, fs, ENV_WIN_MS)
above = env > env.max() * 10 ** (ONSET_THRESH_DB / 20)
rises = np.flatnonzero(above & ~np.r_[False, above[:-1]])
falls = np.flatnonzero(~above & np.r_[False, above[:-1]])
measured = np.array([r for r in rises if r == rises[0] or r - falls[falls < r].max() >= 0.05 * fs]) / fs * 1000
iois = np.diff(measured)

ax = axes[2]
if len(measured) == len(seq_tab):
    kinds = seq_tab['kind'].tolist()
    trans = np.array([f'{a}->{b}' for a, b in zip(kinds[:-1], kinds[1:])])
    for tr, c in {'noise->noise': '0.5', 'word->word': 'tab:blue', 'noise->word': 'tab:green', 'word->noise': 'tab:orange'}.items():
        sel = trans == tr
        ax.plot(np.flatnonzero(sel) + 1, iois[sel], 'o', ms=4, color=c, label=tr)
    print(f'Check: {len(measured)} onsets re-detected, onset-to-onset {iois.min():.1f}-{iois.max():.1f} ms')
else:
    ax.plot(np.arange(1, len(iois) + 1), iois, 'o', ms=3, color='k')
    print(f'WARNING: re-detected {len(measured)} onsets but placed {len(seq_tab)} (a word with an inner gap?)')
ax.axhline(SOA_MS, color='tab:red', lw=1)
ax.set(xlabel='Interval number', ylabel='Onset-to-onset (ms)', ylim=(SOA_MS - 10, SOA_MS + 10))
ax.legend(frameon=False, ncol=4, fontsize=8, loc='upper right')
plt.tight_layout()
plt.show()
seq_tab[['position', 'block_label', 'kind', 'word', 'token', 'onset_ms', 'offset_ms']]
# %%
