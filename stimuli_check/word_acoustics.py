"""
check_03_word_acoustics.py
Step 3: per-word acoustics of ALL SWS words in sws_words/3_sws/ (whole pool, used or not)
plus the noise tokens in sws_words/noise_pool/ as a reference.

Measures per file (times in ms from FILE START, because the stimuli are placed on the
grid by file start):
  lead_sil_ms, trail_sil_ms   silence before / after the sound (threshold SIL_DB re peak)
  sound_ms                    duration of the sounding part
  t10_ms, t50_ms, t90_ms      first time the envelope reaches 10 / 50 / 90 % of its maximum
                              (t50 = -6 dB = the "rise" criterion of make_noise_pool.py)
  rise_10_90_ms               t90 - t10
  t_peak_ms                   time of the envelope maximum
  max_slope_ms                time of the steepest envelope rise in the first ONSET_WIN_MS
  max_slope                   that slope (fraction of max per ms)
  rms_db, peak_db             level of the sounding part (dBFS), crest_db = peak - rms
  onset_class                 from spelling: vowel / sonorant / voiced_obstr / voiceless
Merged from sws_info.csv: stretch, natural_sound_ms, voiced_frac, sws_onset_in_file_ms.

Outputs (OUT_DIR): 03_word_acoustics.csv, 03_onset_envelopes.png, printed summary.
Run (conda env `psychopy`):  python check_03_word_acoustics.py
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import hilbert

# ------------------------------------------------------------------ CONFIG
BASE_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(BASE_DIR)

SWS_DIR   = os.path.join(BASE_DIR, 'sws_words', '3_sws')
NOISE_DIR = os.path.join(BASE_DIR, 'sws_words', 'noise_pool')
SWS_INFO  = os.path.join(BASE_DIR, 'sws_words', 'sws_info.csv')
WORDS_CSV = os.path.join(BASE_DIR, 'sws_words', 'words.csv')
CSV_DIR =os.path.join(BASE_DIR, 'sentence_lists')

OUT_DIR   = os.path.join(BASE_DIR, 'checks')

ENV_SMOOTH_MS = 5      # moving-average smoothing of the Hilbert envelope (same as noise pool: env_smooth_ms = 5)
SIL_DB        = -40    # silence threshold re file peak, for lead / trail silence
ONSET_WIN_MS  = 80     # window after t10 in which the steepest rise is searched
PLOT_MS       = 150    # envelope plot window
# -------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


def read_mono(path):
    fs, raw = wavfile.read(path)
    x = raw.astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        x /= np.iinfo(raw.dtype).max
    if x.ndim > 1:
        x = x.mean(axis=1)
    return fs, x


def envelope(x, fs):
    env = np.abs(hilbert(x))
    n = max(1, int(round(ENV_SMOOTH_MS * fs / 1000)))
    return np.convolve(env, np.ones(n) / n, mode='same')


def first_cross(env, level, start=0):
    idx = np.nonzero(env[start:] >= level)[0]
    return start + idx[0] if len(idx) else np.nan


def onset_class(word):
    w = word.lower()
    if w[0] in 'aeiouäöü':
        return 'vowel'
    if w[0] in 'mnlrj':
        return 'sonorant'
    if w.startswith(('sch', 'sp', 'st', 'ch', 'pf', 'z')) or w[0] in 'ptkfhv':
        return 'voiceless'          # v = /f/ (vom, vor, vier), z = /ts/, ch = /k/ (Chor)
    if w[0] in 'bdgw' or (w[0] == 's' and len(w) > 1 and w[1] in 'aeiouäöü'):
        return 'voiced_obstr'       # w = /v/, s + vowel = /z/ (sein, Sohn, Saal)
    return 'other'


def measure(path):
    fs, x = read_mono(path)
    ms = 1000.0 / fs
    env = envelope(x, fs)
    peak = np.max(np.abs(x))
    above = np.nonzero(np.abs(x) >= peak * 10 ** (SIL_DB / 20))[0]
    on, off = (above[0], above[-1]) if len(above) else (0, len(x) - 1)
    seg = x[on:off + 1]
    emax = env.max()
    i10 = first_cross(env, 0.1 * emax)
    i50 = first_cross(env, 0.5 * emax)
    i90 = first_cross(env, 0.9 * emax)
    # steepest rise in the onset window
    w0 = int(i10) if not np.isnan(i10) else on
    w1 = min(len(env) - 1, w0 + int(ONSET_WIN_MS / ms))
    d = np.diff(env[w0:w1 + 1]) / emax / ms
    j = int(np.argmax(d)) if len(d) else 0
    rms = np.sqrt(np.mean(seg ** 2))
    return dict(
        fs=fs, file_ms=len(x) * ms,
        lead_sil_ms=on * ms, trail_sil_ms=(len(x) - 1 - off) * ms, sound_ms=(off - on + 1) * ms,
        t10_ms=i10 * ms, t50_ms=i50 * ms, t90_ms=i90 * ms, rise_10_90_ms=(i90 - i10) * ms,
        t_peak_ms=int(np.argmax(env)) * ms,
        max_slope_ms=(w0 + j) * ms, max_slope=d[j] if len(d) else np.nan,
        rms_db=20 * np.log10(rms + 1e-12), peak_db=20 * np.log10(peak + 1e-12),
        crest_db=20 * np.log10(peak / (rms + 1e-12) + 1e-12),
    ), env, fs


# ---------------------------------------------------------------- word -> base map
base2word = {}
if os.path.exists(WORDS_CSV):
    wdf = pd.read_csv(WORDS_CSV, dtype=str)
    for w, b in zip(wdf['word'], wdf['base']):
        base2word.setdefault(b.strip(), w.strip())

# ---------------------------------------------------------------- measure words + noise
rows, envs = [], []
for path in sorted(glob.glob(os.path.join(SWS_DIR, '*.wav'))):
    base = os.path.basename(path)[:-4].split('_sws')[0]
    word = base2word.get(base, base)
    r, env, fs = measure(path)
    r.update(kind='word', base=base, word=word, onset_class=onset_class(word), file=os.path.basename(path))
    rows.append(r)
    envs.append((r['onset_class'], env / env.max(), fs))
for path in sorted(glob.glob(os.path.join(NOISE_DIR, '*.wav'))):
    r, env, fs = measure(path)
    r.update(kind='noise', base=os.path.basename(path)[:-4], word='', onset_class='noise',
             file=os.path.basename(path))
    rows.append(r)
    envs.append(('noise', env / env.max(), fs))

df = pd.DataFrame(rows)
if os.path.exists(SWS_INFO):
    info = pd.read_csv(SWS_INFO)
    keep = [c for c in ['base', 'stretch', 'natural_sound_ms', 'voiced_frac', 'sws_onset_in_file_ms', 'capped']
            if c in info.columns]
    df = df.merge(info[keep], on='base', how='left')

first = ['kind', 'word', 'base', 'onset_class']
df = df[first + [c for c in df.columns if c not in first]]
df.to_csv(os.path.join(OUT_DIR, '03_word_acoustics.csv'), index=False, float_format='%.3f')

# ---------------------------------------------------------------- summary
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)
W = df[df.kind == 'word']
N = df[df.kind == 'noise']
cols = ['t10_ms', 't50_ms', 't90_ms', 'rise_10_90_ms', 't_peak_ms', 'rms_db', 'peak_db', 'lead_sil_ms', 'sound_ms']

print(f'Words: {len(W)}   noise tokens: {len(N)}   fs: {sorted(df.fs.unique())}')
print('\n--- Words: distribution ---')
print(W[cols].describe(percentiles=[.1, .5, .9]).T[['mean', 'std', 'min', '10%', '50%', '90%', 'max']].round(1))
print('\n--- Noise tokens: distribution ---')
print(N[cols].describe().T[['mean', 'std', 'min', 'max']].round(1) if len(N) else 'no noise files found')

print('\n--- Words by onset class (mean / SD) ---')
g = W.groupby('onset_class')[['t10_ms', 't50_ms', 't90_ms', 'rise_10_90_ms', 'rms_db']]
print(pd.concat([g.size().rename('n'), g.mean().round(1), g.std().round(1).add_suffix('_sd')], axis=1))

print('\n--- Slowest / fastest words by t50 (time to -6 dB) ---')
s = W.sort_values('t50_ms')[['word', 'onset_class', 't10_ms', 't50_ms', 't90_ms', 'rms_db', 'stretch']]
print('fastest 10:\n', s.head(10).to_string(index=False))
print('slowest 10:\n', s.tail(10).to_string(index=False))

if 'stretch' in W.columns:
    print('\n--- Correlation with time-stretch factor (Pearson r) ---')
    for c in ['t10_ms', 't50_ms', 't90_ms', 'rise_10_90_ms', 'rms_db']:
        print(f'   stretch ~ {c:14s} r = {W["stretch"].corr(W[c]):+.2f}')

print(f'\nLevel spread across words: RMS {W.rms_db.min():.1f} to {W.rms_db.max():.1f} dBFS '
      f'(range {W.rms_db.max() - W.rms_db.min():.1f} dB, SD {W.rms_db.std():.2f}); '
      f'peak range {W.peak_db.max() - W.peak_db.min():.1f} dB')
bad = W[(W.lead_sil_ms > 10)]
print(f'Words with > 10 ms leading silence (alignment): {bad.word.tolist() if len(bad) else "none"}')

# ---------------------------------------------------------------- plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    classes = ['vowel', 'sonorant', 'voiced_obstr', 'voiceless', 'noise']
    fig, axes = plt.subplots(1, len(classes), figsize=(18, 3.6), sharey=True)
    for ax, cl in zip(axes, classes):
        sub = [(e, fs) for c, e, fs in envs if c == cl]
        for e, fs in sub:
            n = int(PLOT_MS * fs / 1000)
            ax.plot(np.arange(n) * 1000 / fs, e[:n], lw=0.6, alpha=0.5)
        ax.axhline(0.5, color='k', ls=':', lw=0.8)
        ax.set_title(f'{cl} (n={len(sub)})')
        ax.set_xlabel('ms from file start')
    axes[0].set_ylabel('envelope (norm.)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '03_onset_envelopes.png'), dpi=120)
    print(f'\nPlot: {os.path.join(OUT_DIR, "03_onset_envelopes.png")}')
except Exception as e:
    print(f'[plot skipped: {e}]')

print(f'CSV : {os.path.join(OUT_DIR, "03_word_acoustics.csv")}')