# %% SET THE ENVIRONMENT
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import stft
from IPython.display import Audio, display
import soundfile as sf

PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'   # the project folder (repo/, sentence_lists/, sws_words/)
os.chdir(PROJECT_DIR)        # every relative path below ('sws_words/...', 'sentence_lists') is inside the project
                             # folder, wherever this script is started from (it lives in stimuli_generator_codes/)
REPO_DIR = PROJECT_DIR       # folder that CONTAINS the repo package
HELPERS_DIR = os.path.join(PROJECT_DIR, 'stimuli_generator_codes')   # folder with stim_helpers.py

if REPO_DIR is not None:
    sys.path.insert(0, REPO_DIR)
if HELPERS_DIR is not None:
    sys.path.insert(0, HELPERS_DIR)
import stim_helpers                      # fails here if stim_helpers.py can't be found
from repo import (extract_tracks, drop_short_voicing, natural_tracks, stretch_tracks, sounding_ms,
                         make_sws, trim_sws, finish_sws, save_audio, load_audio, find_on_offset)
# %% OTHER SETTINGS
IN_DIR    = 'sws_words/2_trimmed'
OUT_DIR   = 'sws_words/3_sws'
TABLE_DIR = 'sws_words'            # sws_info.csv goes here

# ----- tracks (stim_helpers.extract_formants_and_masks)
TIME_STEP          = 0.0125
WIN_LENGTH         = 0.025
MAX_FORMANTS       = 3
MALE_FEMALE        = 1        # 0 male, 1 female (the German gTTS voice is female)
PITCH_FLOOR        = 75
PITCH_CEILING      = 500
F0_SMOOTH_SIGMA    = 1.0
MASK_EDGE_SIGMA_MS = 5

# ----- equalization
TARGET_MS      = 350     # sounding duration of every word, e.g. 400; None = natural duration (no equalization)
STRETCH_ALPHA  = 1.0      # 1 = every word exactly TARGET_MS; 0.5 = halfway (log scale)
MAX_STRETCH    = 1.8      # stretch factor is capped to 1/MAX_STRETCH ... MAX_STRETCH
MASK_THRESHOLD = 0.01     # voicing-mask value that defines the voiced part
MIN_VOICED_MS  = 30       # voiced islands with less than this much clear voicing are removed; None = off

# ----- discrete SWS
WIN_TIME = 0.05           # 50 ms steps

# ----- trimming the SWS
SWS_TRIM_THRESH_DB = -40  # sounding part = envelope above (peak + this)
SWS_ENV_WIN_MS     = 5
SWS_TRIM_PAD_MS    = 10

# ----- finishing
FADE_MS    = 10
TARGET_RMS = 0.06
PEAK       = 0.95
OUT_SR     = 44100



# %%
# make sws words
in_paths = sorted(glob.glob(os.path.join(IN_DIR, '*.wav')))
if not in_paths:
    raise FileNotFoundError(f'no .wav files in {os.path.abspath(IN_DIR)}')
os.makedirs(OUT_DIR, exist_ok=True)

DUR_TAG = '' if TARGET_MS is None else f'_dur{TARGET_MS:.0f}ms'
if TARGET_MS is None:
    print('No equalization: natural word durations')
else:
    print(f'Equalizing to {TARGET_MS:.0f} ms (alpha = {STRETCH_ALPHA}, max stretch = {MAX_STRETCH})')

rows, failed, kept = [], [], {}
for n_done, path in enumerate(in_paths, start=1):
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        # 1 tracks
        tracks = extract_tracks(path, time_step=TIME_STEP, win_length=WIN_LENGTH, max_formants=MAX_FORMANTS,
                                male_female=MALE_FEMALE, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                                f0_smooth_sigma=F0_SMOOTH_SIGMA, mask_edge_sigma_ms=MASK_EDGE_SIGMA_MS)
        fs = int(tracks['fs'])

        # remove stray voiced islands (e.g. the blip before 'wein')
        removed = []
        if MIN_VOICED_MS is not None:
            tracks, removed = drop_short_voicing(tracks, threshold=MASK_THRESHOLD, min_voiced_ms=MIN_VOICED_MS)

        # 2 equalize
        if TARGET_MS is None:
            drive, factor = natural_tracks(tracks), 1.0
        else:
            drive, factor = stretch_tracks(tracks, TARGET_MS, alpha=STRETCH_ALPHA,
                                           max_stretch=MAX_STRETCH, mask_threshold=MASK_THRESHOLD)

        # 3 discrete SWS
        sig, sws_tracks = make_sws(drive, fs, max_formants=MAX_FORMANTS, win_time=WIN_TIME)

        # 4 trim
        sig_trim, cut_start, cut_end = trim_sws(sig, fs, thresh_db=SWS_TRIM_THRESH_DB,
                                                env_win_ms=SWS_ENV_WIN_MS, pad_ms=SWS_TRIM_PAD_MS)

        # 5 fade, normalize, resample
        out, fs_out = finish_sws(sig_trim, fs, fade_ms=FADE_MS, target_rms=TARGET_RMS, peak=PEAK, out_sr=OUT_SR)
    except Exception as e:
        failed.append(base)
        print(f'FAILED: {base} ({type(e).__name__}: {e})')
        continue

    out_path = os.path.join(OUT_DIR, f'{base}_sws{int(round(WIN_TIME * 1000))}ms{DUR_TAG}.wav')
    save_audio(out_path, out, fs_out)

    on, off = find_on_offset(out, fs_out, SWS_TRIM_THRESH_DB, SWS_ENV_WIN_MS)
    capped = TARGET_MS is not None and (factor >= MAX_STRETCH - 1e-6 or factor <= 1 / MAX_STRETCH + 1e-6)
    rows.append(dict(base=base, trim_file=path, sws_file=out_path,
                     natural_sound_ms=round(sounding_ms(tracks, MASK_THRESHOLD)),
                     islands_removed=str(removed) if removed else '',
                     stretch=round(factor, 3), capped=capped,
                     sws_file_ms=round(len(out) / fs_out * 1000),             # incl. the pads
                     sws_sound_ms=round((off - on) / fs_out * 1000),          # audible part
                     sws_onset_in_file_ms=round(on / fs_out * 1000, 1),
                     voiced_frac=round(float(np.mean(drive['voiced_mask'] > 0.5)), 2)))
    kept[base] = dict(tracks=tracks, drive=drive, sws_tracks=sws_tracks, fs=fs, out=out, fs_out=fs_out)

    if n_done % 10 == 0:
        print(f'{n_done}/{len(in_paths)}')

sws_info = pd.DataFrame(rows)
sws_info.to_csv(os.path.join(TABLE_DIR, 'sws_info.csv'), index=False)

print(f'\n{len(sws_info)}/{len(in_paths)} SWS words -> {OUT_DIR}/')
print(f"Audible duration: {sws_info['sws_sound_ms'].min()}-{sws_info['sws_sound_ms'].max()} ms "
      f"(median {sws_info['sws_sound_ms'].median():.0f} ms)")
if failed:
    print('Failed:', ', '.join(failed))
if sws_info['capped'].any():
    print('Hit MAX_STRETCH, did NOT reach TARGET_MS:', ', '.join(sws_info.loc[sws_info['capped'], 'base']))
cleaned = sws_info[sws_info['islands_removed'] != '']
if len(cleaned):
    print('Stray voiced islands removed (ms in the trimmed TTS word):')
    print(cleaned[['base', 'islands_removed']].to_string(index=False))
low = sws_info[sws_info['voiced_frac'] < 0.3]
if len(low):
    print('Little voicing (< 30 %; listen to these):', ', '.join(low['base']))
sws_info.sort_values('sws_sound_ms')


# %%
# check
CHECK = list(sws_info['base'][:6])     # or e.g. ['licht', 'schnee', 'im']

def spec_db(y, fs, win_ms=20, hop_ms=2):
    nper = int(round(win_ms / 1000 * fs)); hop = max(1, int(round(hop_ms / 1000 * fs)))
    f, t, Z = stft(y, fs=fs, window='hann', nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    S = 20 * np.log10(np.abs(Z) + 1e-12)
    return f, t, S - S.max()

sel = [b for b in CHECK if b in kept]
fig, axes = plt.subplots(len(sel), 3, figsize=(14, 2.6 * len(sel)), squeeze=False)
for r, base in enumerate(sel):
    k = kept[base]
    tr, drive, st, fs = k['tracks'], k['drive'], k['sws_tracks'], k['fs']

    ax = axes[r, 0]
    f, t, S = spec_db(np.asarray(tr['y'], float), fs, win_ms=10, hop_ms=1)
    ax.pcolormesh(t, f[f <= 5000], S[f <= 5000], shading='auto', cmap='magma', vmin=-70, vmax=0)
    t0 = np.arange(int(tr['n_samples'])) / fs
    for j in range(MAX_FORMANTS):
        ax.plot(t0, np.where(tr['voiced_mask'] > 0.5, tr['formants_up'][:, j], np.nan), color='w', lw=0.8)
    ax.set(xlim=(0, t0[-1]), ylim=(0, 5000)); ax.set_ylabel(f'{base}\nFreq (Hz)')

    ax = axes[r, 1]
    t1 = np.arange(drive['n_samples']) / fs
    voiced = drive['voiced_mask'] > 0.5
    ax.fill_between(t1, 0, 1, where=~voiced, transform=ax.get_xaxis_transform(), color='0.9', step='mid')
    for x in np.arange(WIN_TIME, t1[-1], WIN_TIME):
        ax.axvline(x, color='0.85', lw=0.5, zorder=0)
    for j, c in enumerate(['tab:blue', 'tab:orange', 'tab:green'][:MAX_FORMANTS]):
        if j in st and st[j]['f_smooth_original'] is not None:
            ax.plot(t1, np.where(voiced, st[j]['f_smooth_original'], np.nan), lw=2, color=c, label=f'F{j + 1}')
    ax.set(xlim=(0, t1[-1]), ylim=(0, 4000))
    row = sws_info.set_index('base').loc[base]
    if row['stretch'] != 1:
        ax.text(0.01, 0.95, f"x{row['stretch']:.2f}", transform=ax.transAxes, va='top', fontsize=8)

    ax = axes[r, 2]
    f, t, S = spec_db(k['out'], k['fs_out'])
    ax.pcolormesh(t, f[f <= 5000], S[f <= 5000], shading='auto', cmap='magma', vmin=-70, vmax=0)
    ax.set(xlim=(0, len(k['out']) / k['fs_out']), ylim=(0, 5000))

axes[0, 0].set_title('Trimmed TTS + formants')
axes[0, 1].set_title(f'Discrete tracks ({int(WIN_TIME * 1000)} ms)')
axes[0, 2].set_title('Saved SWS word')
axes[0, 1].legend(loc='upper right', frameon=False, ncol=3, fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel('Time (s)')
plt.tight_layout()
plt.show()

for base in sel:
    print(base)
    display(Audio(kept[base]['out'], rate=kept[base]['fs_out'], normalize=False))


# %%
# # plot
# SWS_DIR = 'sws_words/3_sws_normal'
# PATTERN = '*_sws50ms_dur350ms.wav'        # '*_dur400ms.wav' for the equalized files

# file_ms, sound_ms = [], []
# for path in sorted(glob.glob(os.path.join(SWS_DIR, PATTERN))):
#     y, fs = load_audio(path)
#     on, off = find_on_offset(y, fs, -40, 5)              # same rule as the SWS trim
#     file_ms.append(len(y) / fs * 1000)                   # whole file, incl. the 10 ms pads
#     sound_ms.append((off - on) / fs * 1000)              # audible part

# fig, ax = plt.subplots(figsize=(8, 3.5))
# bins = np.arange(min(sound_ms) // 10 * 10, max(file_ms) + 20, 10)
# ax.hist(sound_ms, bins=bins, color='tab:blue', alpha=0.8, label='audible part')
# ax.hist(file_ms, bins=bins, color='0.6', alpha=0.5, label='whole file')
# ax.set(xlabel='Duration (ms)', ylabel='Words', title=f'{len(sound_ms)} SWS words')
# ax.legend(frameon=False)
# plt.tight_layout(); plt.show()

# print(f'audible: {min(sound_ms):.0f}–{max(sound_ms):.0f} ms, median {np.median(sound_ms):.0f}, SD {np.std(sound_ms):.1f}')

