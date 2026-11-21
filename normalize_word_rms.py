"""
Equal-RMS normalization of the frozen SWS words (replaces the old peak normalization).

Why: norm_audio() did RMS normalization and then peak normalization, so the peak step undid the
RMS step (all words peaked at 0.95, RMS varied: SD 0.56 dB, range 2.6 dB).
What: one gain per word so every word has the same RMS. Pure gain: no regeneration, waveform shape,
t50 and onset classes are unchanged. Common target = highest RMS at which the peakiest word still
peaks at PEAK (0.95) -> no clipping.
RMS window: the audible part (same rule as build_tapping_stimuli.py: 5-ms RMS envelope, -40 dB
re max), so the builder's `levels` check shows SD ~0.

Run from the folder that contains sws_words/:
    python normalize_word_rms.py
Then: rebuild the noise pool (make_noise_pool.py) and all participant stimuli.
"""
import glob
import json
import os
import shutil

import numpy as np
import soundfile as sf

SWS_DIR = 'sws_words/3_sws_no_rms'
BACKUP_DIR = 'sws_words/3_sws_peaknorm_backup'   # original files, untouched
PEAK = 0.95
WIN_MS = 5
THRESH_DB = -40


def audible_span(x, fs):
    n = max(1, int(round(WIN_MS / 1000 * fs)))
    env = np.sqrt(np.convolve(x ** 2, np.ones(n) / n, mode='same'))
    above = np.where(env >= env.max() * 10 ** (THRESH_DB / 20))[0]
    return above[0], above[-1]


# 1 backup (once) -- always normalize FROM the backup, so rerunning is safe
if not os.path.isdir(BACKUP_DIR):
    shutil.copytree(SWS_DIR, BACKUP_DIR)
    print(f'backup: {BACKUP_DIR}')
files = sorted(glob.glob(os.path.join(BACKUP_DIR, '*.wav')))
assert files, f'no wav files in {BACKUP_DIR}'

# 2 measure
data = {}
for f in files:
    x, fs = sf.read(f, dtype='float64')
    info = sf.info(f)
    assert x.ndim == 1, f'{f} is not mono'
    on, off = audible_span(x, fs)
    rms = np.sqrt(np.mean(x[on:off + 1] ** 2))
    data[os.path.basename(f)] = dict(x=x, fs=fs, subtype=info.subtype, rms=rms, peak=np.abs(x).max())

# 3 common target RMS: the peakiest word (highest peak/RMS) lands exactly at PEAK
crest = {k: v['peak'] / v['rms'] for k, v in data.items()}
target_rms = PEAK / max(crest.values())

# 4 apply and write
log = {}
for k, v in data.items():
    g = target_rms / v['rms']
    y = v['x'] * g
    assert np.abs(y).max() <= PEAK + 1e-9, k
    sf.write(os.path.join(SWS_DIR, k), y, v['fs'], subtype=v['subtype'])
    log[k] = dict(gain_db=round(20 * np.log10(g), 3), rms_db_before=round(20 * np.log10(v['rms']), 3),
                  peak_db_after=round(20 * np.log10(np.abs(y).max()), 3))

# 5 verify from disk
after = []
for k in data:
    y, fs = sf.read(os.path.join(SWS_DIR, k), dtype='float64')
    on, off = audible_span(y, fs)
    after.append(20 * np.log10(np.sqrt(np.mean(y[on:off + 1] ** 2))))
after = np.array(after)
before = np.array([20 * np.log10(v['rms']) for v in data.values()])
gains = np.array([v['gain_db'] for v in log.values()])

summary = dict(n_words=len(data), target_rms=target_rms, target_rms_db=round(20 * np.log10(target_rms), 3),
               peak_limit=PEAK, limiting_word=max(crest, key=crest.get),
               rms_db_before=dict(min=round(before.min(), 2), max=round(before.max(), 2), sd=round(before.std(), 3)),
               rms_db_after=dict(min=round(after.min(), 2), max=round(after.max(), 2), sd=round(after.std(), 3)),
               gain_db=dict(min=round(gains.min(), 2), max=round(gains.max(), 2)), per_word=log)
with open('sws_words/rms_normalization.json', 'w') as fh:
    json.dump(summary, fh, indent=2)

print(f"{len(data)} words -> common RMS {summary['target_rms_db']:.2f} dBFS (limited by {summary['limiting_word']})")
print(f"RMS before: {before.min():.2f} to {before.max():.2f} dB (SD {before.std():.2f})")
print(f"RMS after:  {after.min():.2f} to {after.max():.2f} dB (SD {after.std():.3f})")
print(f"gains: {gains.min():+.2f} to {gains.max():+.2f} dB; max peak {max(v['peak_db_after'] for v in log.values()):.2f} dBFS")
print('NEXT: rebuild noise pool (make_noise_pool.py), then all participant stimuli.')