# %%
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio, display

PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'   # the project folder (repo/, sentence_lists/, sws_words/)
os.chdir(PROJECT_DIR)        # every relative path below ('sws_words/...', 'sentence_lists') is inside the project
                             # folder, wherever this script is started from (it lives in stimuli_generator_codes/)
REPO_DIR = PROJECT_DIR       # folder that CONTAINS the repo package

if REPO_DIR is not None:
    sys.path.insert(0, REPO_DIR)
from repo import load_audio, save_audio, trim_silence

# %%
# SETTINGS
IN_DIR    = 'sws_words/1_tts'
OUT_DIR   = 'sws_words/2_trimmed'
TABLE_DIR = 'sws_words'             

TRIM_THRESH_REL_DB = -40
TRIM_CHUNK_MS      = 5
TRIM_PAD_MS        = 10

# %%
# TRIM AND SAVE
in_paths = sorted(glob.glob(os.path.join(IN_DIR, '*.wav')))
if not in_paths:
    raise FileNotFoundError(f'no .wav files in {os.path.abspath(IN_DIR)}')
os.makedirs(OUT_DIR, exist_ok=True)

rows = []
for path in in_paths:
    name = os.path.basename(path)
    y, fs = load_audio(path)
    y_trim, start, end = trim_silence(y, fs, thresh_rel_db=TRIM_THRESH_REL_DB,
                                      chunk_ms=TRIM_CHUNK_MS, pad_ms=TRIM_PAD_MS)
    out_path = os.path.join(OUT_DIR, name)
    save_audio(out_path, y_trim, fs, subtype='PCM_16')
    rows.append(dict(file=name, base=os.path.splitext(name)[0], tts_file=path, trim_file=out_path, fs=fs,
                     tts_ms=round(len(y) / fs * 1000),
                     trim_start_ms=round(start / fs * 1000), trim_end_ms=round(end / fs * 1000),
                     trimmed_ms=round((end - start) / fs * 1000)))

trim_info = pd.DataFrame(rows)
trim_info.to_csv(os.path.join(TABLE_DIR, 'trim_info.csv'), index=False)

print(f'{len(trim_info)} words trimmed -> {OUT_DIR}/')
print(f"Trimmed duration: {trim_info['trimmed_ms'].min()}-{trim_info['trimmed_ms'].max()} ms "
      f"(median {trim_info['trimmed_ms'].median():.0f} ms)")
nothing_cut = trim_info[trim_info['trimmed_ms'] == trim_info['tts_ms']]
if len(nothing_cut):
    print('Nothing was cut from (check these):', ', '.join(nothing_cut['file']))
trim_info.sort_values('trimmed_ms')


# %%
# Check 
# Each word's TTS waveform; the shaded part is what was kept. Pick words in `CHECK`.
CHECK = list(trim_info['base'][:8])     # or e.g. ['licht', 'schnee', 'weg', 'weg_2']

sel = trim_info[trim_info['base'].isin(CHECK)]
ncol = min(4, len(sel))
nrow = int(np.ceil(len(sel) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.3 * nrow), squeeze=False)
for ax, (_, row) in zip(axes.flat, sel.iterrows()):
    y, fs = load_audio(row['tts_file'])
    t = np.arange(len(y)) / fs
    ax.plot(t, y, lw=0.4, color='k')
    ax.axvspan(row['trim_start_ms'] / 1000, row['trim_end_ms'] / 1000, color='tab:green', alpha=0.15)
    ax.set_title(f"{row['base']}  ({row['trimmed_ms']} ms kept)", fontsize=10)
    ax.set(xlim=(0, t[-1]), ylim=(-1, 1), yticks=[])
for ax in axes.flat[len(sel):]:
    ax.axis('off')
for ax in axes[-1]:
    ax.set_xlabel('Time (s)')
plt.tight_layout()
plt.show()

# for _, row in sel.iterrows():
#     print(row['base'])
#     display(Audio(row['trim_file']))
# %%
