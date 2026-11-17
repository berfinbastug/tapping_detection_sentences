# %% MAKE THE NOISE POOL (ONCE, SHARED BY ALL PARTICIPANTS)
"""
Builds the pool of noise tokens that the tapping stimuli use, ONCE for the whole experiment.
Every participant hears the same tokens with the same shared gain; only the ORDER of the tokens
differs between participants (set in build_tapping_stimuli.py).

Nothing is ever overwritten: if NOISE_DIR exists, the script stops. Rebuild the pool (delete the folder)
whenever the SWS words change (new sentence lists, re-made words): the spectrum, envelope, level and the
shared gain all come from the words. build_tapping_stimuli.py checks this for you (word md5s below).

    sws_words/noise_pool/pool_<carrier>_tokNN_<dur>ms.wav   the tokens, already x shared gain (save_token_pool)
    sws_words/noise_pool/noise_settings.json                settings + measured values (save_token_pool)
    sws_words/noise_pool/noise_pool_exact.json              what the builder needs, unrounded:
                                                            shared gain, token onset/offset in samples,
                                                            token files, md5 of every reference word
"""
import os
import sys
import glob
import json
import shutil
import hashlib
from datetime import datetime

import numpy as np

PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(PROJECT_DIR)        # all relative paths below are inside the project folder
REPO_DIR = PROJECT_DIR

if REPO_DIR is not None:
    sys.path.insert(0, REPO_DIR)
from repo import make_noise_pool, save_token_pool, find_on_offset

# %% SETTINGS (same values as make_trial.py)
SWS_DIR     = 'sws_words/3_sws'
WORD_SUFFIX = '_sws50ms_dur350ms.wav'      # reference words = every SWS word of this version
NOISE_DIR   = 'sws_words/noise_pool'
FS          = 44100

ONSET_THRESH_DB = -40                      # onset/offset rule: same as the SWS trimming and the builders
ENV_WIN_MS      = 5

CARRIER          = 'noise'     # 'noise' or 'harmonic'
F0_HZ            = 150         # only for 'harmonic'
PHASES           = 'schroeder' # only for 'harmonic'
NOISE_DUR_MS     = None        # None = mean audible duration of the words
NOISE_SPECTRUM   = 'shaped'
NOISE_SMOOTH_OCT = 3
NOISE_BAND_HZ    = (150, 5000)
NOISE_FIR_TAPS   = 2049
NOISE_ENV        = 'word'
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
N_TOKENS         = 20          # >= 18: every trial draws its 18 bursts without replacement
NOISE_SEED       = 1

PAD_MS = 10        # silence put around every token by the builder (the SWS words have 10 ms pads, too)


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


# %% 0 OUTPUT FOLDER
if os.path.exists(NOISE_DIR):
    raise FileExistsError(f'{NOISE_DIR} already exists - not overwriting. Delete or rename it to rebuild the pool '
                          f'(and then rebuild every participant\'s tapping stimuli).')
tmp_dir = NOISE_DIR + '_incomplete'
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)

reference = sorted(glob.glob(os.path.join(SWS_DIR, '*' + WORD_SUFFIX)))
if not reference:
    raise FileNotFoundError(f'no reference words {os.path.join(SWS_DIR, "*" + WORD_SUFFIX)}')
print(f'{len(reference)} reference words')

# %% 1 THE POOL
pool = make_noise_pool(reference,
                       fs=FS, onset_thresh_db=ONSET_THRESH_DB, env_win_ms=ENV_WIN_MS,
                       carrier=CARRIER, f0_hz=F0_HZ if CARRIER == 'harmonic' else None,
                       phases=PHASES if CARRIER == 'harmonic' else None,
                       duration_ms=NOISE_DUR_MS, spectrum=NOISE_SPECTRUM, smooth_oct=NOISE_SMOOTH_OCT,
                       band_hz=NOISE_BAND_HZ, fir_taps=NOISE_FIR_TAPS, envelope=NOISE_ENV,
                       env_smooth_ms=NOISE_ENV_SMOOTH_MS, ramp_ms=NOISE_RAMP_MS, match_rise=MATCH_RISE,
                       rise_db=RISE_DB, rise_win_ms=RISE_WIN_MS, ramp_search_ms=RAMP_SEARCH_MS,
                       ramp_step_ms=RAMP_STEP_MS, level=NOISE_LEVEL, level_db=NOISE_LEVEL_DB,
                       max_peak=NOISE_MAX_PEAK, n_tokens=N_TOKENS, seed=NOISE_SEED)

# %% 2 ONSET / OFFSET OF A PADDED TOKEN
# make_noise_pool measures the onset on the token's envelope WITHOUT silence before it. The 5 ms RMS window
# then cannot "see" the zeros before sample 0, so the -40 dB point is cut off at sample 0 (and the offset at the
# last sample). In a trial there IS silence before the token, and there the same rule finds the onset slightly
# earlier. The words are measured with 10 ms of silence around them, so measure the noise the same way:
# pad the expected envelope by PAD_MS and apply the same rule. The builder pads every token by PAD_MS too.
pad = int(round(PAD_MS / 1000 * FS))
env_padded = np.r_[np.zeros(pad), pool['envelope'], np.zeros(pad)]
on_p, off_p = find_on_offset(env_padded, FS, ONSET_THRESH_DB, ENV_WIN_MS)

# %% 3 SAVE (temporary folder, then rename)
paths = save_token_pool(pool, tmp_dir)
exact = dict(
    built=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    fs=int(pool['info']['fs']),
    shared_gain=float(pool['gain']),                       # unrounded: the builder multiplies the words by it
    pad_samples=pad,                                       # the builder puts this much silence around every token
    onset_sample_padded=int(on_p), offset_sample_padded=int(off_p),   # USED: sample index in the PADDED token
    onset_sample_unpadded=int(pool['onset']), offset_sample_unpadded=int(pool['offset']),   # repo values, for info
    n_samples=int(len(pool['tokens'][0])),
    onset_thresh_db=ONSET_THRESH_DB, env_win_ms=ENV_WIN_MS,
    token_files=[os.path.basename(p) for p in paths],
    reference_words={os.path.basename(p): md5(p) for p in reference},
)
with open(os.path.join(tmp_dir, 'noise_pool_exact.json'), 'w') as fh:
    json.dump(exact, fh, indent=2)
os.rename(tmp_dir, NOISE_DIR)

info = pool['info']
print(f"✓ {CARRIER}: {info['n_tokens']} tokens of {info['duration_ms']:.0f} ms -> {NOISE_DIR}/")
print(f"  rise to {RISE_DB} dB: words {info['word_rise_ms_mean']:.1f} ms, noise {info['noise_rise_ms']:.1f} ms "
      f"(ramp {info['ramp_ms']} ms)  {info['rise_note']}")
print(f"  shared gain {info['shared_gain_db']:+.2f} dB, noise RMS - word RMS = {info['rms_noise_minus_words_db']:+.2f} dB")
print(f"  token onset {(on_p - pad) / FS * 1000:+.2f} ms, offset {(off_p - pad) / FS * 1000:.2f} ms relative to the "
      f"token's first sample (unpadded measurement: {pool['onset'] / FS * 1000:+.2f} / {pool['offset'] / FS * 1000:.2f} ms)")