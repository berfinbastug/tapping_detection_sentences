# %% MAP TRIMMED TTS -> SWS
"""
Shows which part of the trimmed TTS word ends up where in the saved SWS word.

The word is rebuilt with the SAME functions and settings as make_sws_words_trimmed.py (nothing is saved).
Mapping rule: the SWS is made only from the voiced part of the TTS word (voiced_mask > 0.5); unvoiced parts
(frication, closures, bursts, silence) produce no tone. stretch_tracks stretches the voiced time to TARGET_MS,
so the n-th % of voiced TTS time maps to the n-th % of voiced time in the stretched tracks. trim_sws then cuts
the edges; the cut is found by matching the saved SWS against the untrimmed SWS (cross-correlation).

Figure:  top    = trimmed TTS spectrogram + formant tracks; dark bands = unvoiced = NOT in the SWS
         bottom = saved SWS spectrogram
         lines  = the same moment in both (one line every STEP_MS of voiced TTS time; same colour = same moment)
Audio:   1 trimmed TTS   2 TTS with the unvoiced parts muted (= the material the SWS is made of)   3 SWS
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy.signal import stft, correlate, correlation_lags, resample_poly
from IPython.display import Audio, display

PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'stimuli_generator_codes'))
from repo import extract_tracks, drop_short_voicing, natural_tracks, stretch_tracks, make_sws, trim_sws, finish_sws
# %%
WORDS   = ['blitz']          # bases, as in sws_words/2_trimmed/<base>.wav
STEP_MS = 20                # one connecting line every STEP_MS of voiced TTS time
FMAX    = 5000              # spectrogram upper limit (Hz)

# ----- settings: SAME as make_sws_words_trimmed.py (change them there AND here)
IN_DIR             = 'sws_words/2_trimmed'
TIME_STEP          = 0.0125
WIN_LENGTH         = 0.025
MAX_FORMANTS       = 3
MALE_FEMALE        = 1
PITCH_FLOOR        = 75
PITCH_CEILING      = 500
F0_SMOOTH_SIGMA    = 1.0
MASK_EDGE_SIGMA_MS = 5
TARGET_MS          = 350
STRETCH_ALPHA      = 1.0
MAX_STRETCH        = 1.8
MASK_THRESHOLD     = 0.01
MIN_VOICED_MS      = 30
WIN_TIME           = 0.05
SWS_TRIM_THRESH_DB = -40
SWS_ENV_WIN_MS     = 5
SWS_TRIM_PAD_MS    = 10
FADE_MS            = 10
TARGET_RMS         = 0.06
PEAK               = 0.95
OUT_SR             = 44100

VOICED = 0.5                # voiced_mask above this = voiced (same as the plots in make_sws_words_trimmed.py)


def rebuild(base):
    """Same steps as make_sws_words_trimmed.py; returns everything needed for the mapping."""
    path = os.path.join(IN_DIR, base + '.wav')
    tracks = extract_tracks(path, time_step=TIME_STEP, win_length=WIN_LENGTH, max_formants=MAX_FORMANTS,
                            male_female=MALE_FEMALE, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                            f0_smooth_sigma=F0_SMOOTH_SIGMA, mask_edge_sigma_ms=MASK_EDGE_SIGMA_MS)
    fs = int(tracks['fs'])
    if MIN_VOICED_MS is not None:
        tracks, _ = drop_short_voicing(tracks, threshold=MASK_THRESHOLD, min_voiced_ms=MIN_VOICED_MS)
    if TARGET_MS is None:
        drive, factor = natural_tracks(tracks), 1.0
    else:
        drive, factor = stretch_tracks(tracks, TARGET_MS, alpha=STRETCH_ALPHA,
                                       max_stretch=MAX_STRETCH, mask_threshold=MASK_THRESHOLD)
    sig, _ = make_sws(drive, fs, max_formants=MAX_FORMANTS, win_time=WIN_TIME)
    sig_trim, _, _ = trim_sws(sig, fs, thresh_db=SWS_TRIM_THRESH_DB, env_win_ms=SWS_ENV_WIN_MS,
                              pad_ms=SWS_TRIM_PAD_MS)
    out, fs_out = finish_sws(sig_trim, fs, fade_ms=FADE_MS, target_rms=TARGET_RMS, peak=PEAK, out_sr=OUT_SR)
    return dict(tracks=tracks, drive=drive, factor=factor, sig=np.asarray(sig, float),
                out=np.asarray(out, float), fs=fs, fs_out=int(fs_out))


def make_mapping(k):
    """Returns tts_to_sws(t_ms) -> ms in the saved SWS file (NaN where the TTS is unvoiced)."""
    fs, fs_out = k['fs'], k['fs_out']
    v_tts = np.asarray(k['tracks']['voiced_mask']) > VOICED
    v_drv = np.asarray(k['drive']['voiced_mask']) > VOICED
    c_tts, c_drv = np.cumsum(v_tts), np.cumsum(v_drv)
    if c_tts[-1] == 0 or c_drv[-1] == 0:
        raise ValueError('no voiced samples')

    # where the saved SWS starts inside the untrimmed SWS
    sig = k['sig'] if fs_out == fs else resample_poly(k['sig'], fs_out, fs)
    xc = correlate(sig, k['out'], mode='full', method='fft')
    lags = correlation_lags(len(sig), len(k['out']), mode='full')
    best = int(np.argmax(xc))
    shift = int(lags[best])                       # saved sample 0 = untrimmed sample `shift` (fs_out samples)
    a = sig[max(shift, 0): shift + len(k['out'])]
    b = k['out'][max(-shift, 0): max(-shift, 0) + len(a)]
    r = np.corrcoef(a, b)[0, 1]
    if r < 0.99:
        print(f'  WARNING: rebuilt SWS matches the trimmed/finished SWS only with r = {r:.3f}; mapping may be off')

    def tts_to_sws(t_ms):
        i = np.clip(np.round(np.asarray(t_ms, float) / 1000 * fs).astype(int), 0, len(v_tts) - 1)
        target = c_tts[i] * c_drv[-1] / c_tts[-1]            # same fraction of voiced time
        j = np.clip(np.searchsorted(c_drv, target), 0, len(v_drv) - 1)
        t = j / fs * 1000 - shift / fs_out * 1000
        return np.where(v_tts[i], t, np.nan)

    return tts_to_sws, v_tts


def runs(mask):
    """(start, end) sample index of every True run."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def spec_db(y, fs, win_ms=10, hop_ms=1):
    nper = int(round(win_ms / 1000 * fs)); hop = max(1, int(round(hop_ms / 1000 * fs)))
    f, t, Z = stft(y, fs=fs, window='hann', nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    S = 20 * np.log10(np.abs(Z) + 1e-12)
    keep = f <= FMAX
    return f[keep], t, (S - S.max())[keep]



for base in WORDS:
    k = rebuild(base)
    tts_to_sws, v_tts = make_mapping(k)
    tr, fs, fs_out, out = k['tracks'], k['fs'], k['fs_out'], k['out']
    y_tts = np.asarray(tr['y'], float)
    t_tts = np.arange(len(y_tts)) / fs

    # ----- table: each TTS stretch -> where it is in the SWS
    print(f'\n{base}   stretch x{k["factor"]:.2f}   TTS {len(y_tts) / fs * 1000:.0f} ms -> SWS {len(out) / fs_out * 1000:.0f} ms')
    edges = sorted({0, len(v_tts)} | {s for r in runs(v_tts) for s in r})
    for s, e in zip(edges[:-1], edges[1:]):
        a_ms, b_ms = s / fs * 1000, e / fs * 1000
        if v_tts[s]:
            sa, sb = tts_to_sws([a_ms, b_ms - 1000 / fs])
            print(f'  TTS {a_ms:6.0f}-{b_ms:4.0f} ms  voiced    -> SWS {sa:6.0f}-{sb:4.0f} ms')
        else:
            print(f'  TTS {a_ms:6.0f}-{b_ms:4.0f} ms  unvoiced  -> not in the SWS')

    # ----- figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), gridspec_kw=dict(hspace=0.5))

    f, t, S = spec_db(y_tts, fs)
    ax1.pcolormesh(t, f, S, shading='auto', cmap='magma', vmin=-70, vmax=0)
    for j in range(MAX_FORMANTS):
        ax1.plot(t_tts, np.where(v_tts, tr['formants_up'][:, j], np.nan), color='w', lw=0.8)
    for s, e in runs(~v_tts):
        ax1.axvspan(s / fs, e / fs, color='0.15', alpha=0.6, lw=0)
        if (e - s) / fs > 0.04:
            ax1.text((s + e) / 2 / fs, FMAX * 0.93, 'unvoiced\nnot in SWS', color='w', ha='center', va='top', fontsize=8)
    ax1.set(xlim=(0, t_tts[-1]), ylim=(0, FMAX), ylabel='Freq (Hz)', xlabel='Time in trimmed TTS (s)',
            title=f'{base}: trimmed TTS (white = formant tracks the SWS follows)')

    f, t, S = spec_db(out, fs_out)
    ax2.pcolormesh(t, f, S, shading='auto', cmap='magma', vmin=-70, vmax=0)
    ax2.set(xlim=(0, len(out) / fs_out), ylim=(0, FMAX), ylabel='Freq (Hz)', xlabel='Time in saved SWS (s)',
            title=f'{base}: saved SWS (voiced part stretched x{k["factor"]:.2f})')

    grid = np.arange(0, len(y_tts) / fs * 1000, STEP_MS)
    grid = grid[v_tts[np.clip((grid / 1000 * fs).astype(int), 0, len(v_tts) - 1)]]
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(grid), 1)))
    for g, s_ms, c in zip(grid, tts_to_sws(grid), colors):
        ax1.axvline(g / 1000, color=c, lw=1.2, ymax=0.08)
        ax2.axvline(s_ms / 1000, color=c, lw=1.2, ymin=0.92)
        fig.add_artist(ConnectionPatch(xyA=(g / 1000, 0), coordsA=ax1.transData,
                                       xyB=(s_ms / 1000, FMAX), coordsB=ax2.transData, color=c, lw=1))
    plt.show()

    # ----- listen
    muted = y_tts * np.asarray(tr['voiced_mask'], float)
    print('1 trimmed TTS');                          display(Audio(y_tts, rate=fs))
    print('2 TTS, unvoiced parts muted (= what the SWS is made of)'); display(Audio(muted, rate=fs))
    print('3 SWS');                                  display(Audio(out, rate=fs_out, normalize=False))
# %%
