"""Word-shaped noise (or a word-shaped harmonic tone): a pool of tokens matched to a set of reference words.

carrier = 'noise'    : Gaussian noise (the original).
carrier = 'harmonic' : a harmonic complex tone - one steady pitch (f0_hz), all harmonics up to the band's upper
                       edge, Schroeder phases (flat envelope, low peaks). Clearly not speech, and periodic like
                       the words. Every token is the same sound.
Both get exactly the same spectrum filter, envelope, rise matching and RMS.

Port of step 6 of sentence_to_sws_noise_blocks_v2.ipynb. Every burst is matched to the reference words
(e.g. all SWS words of the experiment, sws_words/3_sws/*_dur400ms.wav), so the noise marks where a
sentence starts and ends by its timbre and does nothing else:

    spectrum  : long-term spectrum of the words, smoothed in 1/N octave (no formant peaks), band-limited
    duration  : mean audible duration of the words
    envelope  : mean amplitude envelope of the words (or a flat burst)
    rise      : on-ramp tuned so a burst rises as fast as the words do on average
    level     : same RMS (over the audible part) as the words on average
    gain      : one shared gain for words AND noise keeps all peaks below a limit

Order of use:
    words = load_reference_words(paths, ...)
    f, p = long_term_spectrum(words); fir = shaping_filter(f, target_spectrum(f, p, ...), ...)
    template = word_envelope_template(words, n, ...)
    ramp_ms, rise = match_ramp(template, fs, mean(word_rise_ms(words, ...)), ...)
    envelope = burst_envelope(template, fs, ramp_ms)
    tokens = make_token_pool(n_tokens, fir, envelope, rms, seed)
    gain = shared_gain(tokens + [w['y'] for w in words], max_peak)   -> apply to tokens AND words
or all of it at once with make_noise_pool(...).
"""
import glob
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, firwin2, welch

from .audio import find_on_offset, resample, rms_envelope


# ---------------------------------------------------------------- reference words

def load_reference_words(paths, *, fs, onset_thresh_db, env_win_ms):
    """Load the reference words (mono, resampled to fs) and find their audible part.

    paths: a list of files or a glob pattern ('sws_words/3_sws/*_dur350ms.wav').
    Audible part = RMS envelope (env_win_ms) above (peak + onset_thresh_db) dB.
    Returns a list of dicts: name, y, fs, on, off (samples; the audible part is y[on:off + 1]).
    """
    if isinstance(paths, str):
        paths = sorted(glob.glob(paths))
    words = []
    for p in paths:
        y, fs_in = sf.read(p, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        y, fs_y = resample(y, fs_in, fs)
        on, off = find_on_offset(y, fs_y, onset_thresh_db, env_win_ms)
        words.append(dict(name=os.path.basename(p), y=y, fs=fs_y, on=on, off=off))
    if not words:
        raise FileNotFoundError(f'no reference words found ({paths})')
    return words


def sound_part(word):
    """The audible part of a word from load_reference_words, as float."""
    return word['y'][word['on']:word['off'] + 1].astype(float)


def mean_sound_ms(words):
    """Mean audible duration (ms) of the reference words."""
    return float(np.mean([(w['off'] - w['on'] + 1) / w['fs'] * 1000 for w in words]))


def mean_sound_rms(words):
    """Mean RMS of the reference words over their audible parts."""
    return float(np.mean([np.sqrt(np.mean(sound_part(w) ** 2)) for w in words]))


# ---------------------------------------------------------------- spectrum

def long_term_spectrum(words):
    """Long-term power spectrum of the audible parts (Welch; words weighted by their length).

    Returns (f, p).
    """
    psds, lens = [], []
    for w in words:
        s = sound_part(w)
        f, p = welch(s, fs=w['fs'], nperseg=min(4096, len(s)))
        psds.append(p)
        lens.append(len(s))
    return f, np.average(psds, axis=0, weights=lens)


def smooth_octave(f, p, frac):
    """Average the power within +-1/(2*frac) octave around each frequency (frac=3 -> 1/3 octave)."""
    k = 2 ** (1 / (2 * frac))
    out = np.empty_like(p)
    for i, fi in enumerate(f):
        sel = (f >= fi / k) & (f <= fi * k) if fi > 0 else (f == 0)
        out[i] = p[sel].mean()
    return out


def target_spectrum(f, p, *, spectrum, smooth_oct, band_hz):
    """The noise's target power spectrum.

    spectrum : 'shaped' = smooth_octave(f, p, smooth_oct); 'white' = flat.
    band_hz  : (low, high) - power outside is set to 0; None = keep everything.
    """
    if spectrum == 'shaped':
        p_target = smooth_octave(f, p, smooth_oct)
    elif spectrum == 'white':
        p_target = np.ones_like(p)
    else:
        raise ValueError("spectrum must be 'shaped' or 'white'")
    if band_hz is not None:
        p_target = p_target.copy()
        p_target[(f < band_hz[0]) | (f > band_hz[1])] = 0
    return p_target


def shaping_filter(f, p_target, fs, *, n_taps):
    """Linear-phase FIR (n_taps, odd) whose magnitude follows sqrt(p_target), peak 1."""
    if n_taps % 2 != 1:
        raise ValueError('n_taps must be odd')
    g = np.sqrt(p_target)
    return firwin2(n_taps, f / (fs / 2), g / g.max())


def make_noise(n, fir, rng):
    """n samples of Gaussian noise through the shaping filter (no filter edge effects), unit SD."""
    x = fftconvolve(rng.standard_normal(n + len(fir) - 1), fir, mode='valid')
    return x / x.std()


def make_harmonic(n, fir, fs, *, f0_hz, f_max_hz, phases):
    """n samples of a harmonic complex tone through the shaping filter, unit SD.

    Harmonics k * f0_hz for k = 1, 2, ... up to f_max_hz, equal amplitude before the filter
    (the filter then gives them the words' spectral shape).
    phases : 'schroeder' (phi_k = pi k (k-1) / K: flat envelope, low peak factor) or
             'sine' (all in sine phase: a pulse train - peaky, mainly for comparison).
    """
    t = np.arange(n + len(fir) - 1) / fs
    K = int(np.floor(f_max_hz / f0_hz))
    if K < 1:
        raise ValueError('f0_hz is above f_max_hz')
    k = np.arange(1, K + 1)
    if phases == 'schroeder':
        phi = np.pi * k * (k - 1) / K
    elif phases == 'sine':
        phi = np.zeros(K)
    else:
        raise ValueError("phases must be 'schroeder' or 'sine'")
    x = np.zeros(len(t))
    for kk, ph in zip(k, phi):
        x += np.sin(2 * np.pi * kk * f0_hz * t + ph)
    x = fftconvolve(x, fir, mode='valid')
    return x / x.std()


# ---------------------------------------------------------------- envelope and rise

def cosine_ramps(n, fs, ms):
    """Flat envelope of n samples with ms cosine on/off ramps."""
    env = np.ones(n)
    r = min(int(round(ms / 1000 * fs)), n // 2)
    if r > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r))
        env[:r] = ramp
        env[-r:] = ramp[::-1]
    return env


def word_envelope_template(words, n, *, smooth_ms):
    """Mean amplitude envelope of the words' audible parts, each resampled to n samples, peak 1.

    smooth_ms: RMS window - short enough not to smear the rise, long enough to smooth the beating
    between the sine waves.
    """
    from .equalize import resample_rows
    envs = []
    for w in words:
        e = rms_envelope(w['y'], w['fs'], smooth_ms)[w['on']:w['off'] + 1]
        envs.append(resample_rows(e / e.max(), n))
    env = np.mean(envs, axis=0)
    return env / env.max()


def burst_envelope(template, fs, ramp_ms):
    """Envelope of one burst: template x cosine on/off ramps of ramp_ms."""
    return cosine_ramps(len(template), fs, ramp_ms) * template


def rise_ms(env, fs, *, rise_db, onset_db):
    """Time (ms) from the onset (env > peak + onset_db) to the first point at peak + rise_db."""
    peak = env.max()
    on = int(np.argmax(env > peak * 10 ** (onset_db / 20)))
    return (int(np.argmax(env >= peak * 10 ** (rise_db / 20))) - on) / fs * 1000


def word_rise_ms(words, *, rise_db, onset_db, win_ms):
    """Rise time (ms) of every reference word, measured on its RMS envelope (win_ms)."""
    return np.array([rise_ms(rms_envelope(w['y'], w['fs'], win_ms), w['fs'], rise_db=rise_db, onset_db=onset_db)
                     for w in words])


def expected_noise_rise(template, fs, ramp_ms, *, rise_db, onset_db, win_ms):
    """Rise time (ms) a burst with this ramp has on average (the RMS envelope of the deterministic
    envelope = the expected RMS envelope of unit noise x envelope)."""
    pad = np.zeros(int(0.05 * fs))
    env = np.r_[pad, burst_envelope(template, fs, ramp_ms), pad]
    return rise_ms(rms_envelope(env, fs, win_ms), fs, rise_db=rise_db, onset_db=onset_db)


def match_ramp(template, fs, target_rise_ms, *, search_ms, step_ms, rise_db, onset_db, win_ms):
    """Search the ramp (search_ms = (min, max), in step_ms steps) whose expected rise is closest to
    target_rise_ms.

    Returns (ramp_ms, noise_rise_ms, note): note is '' or says why the target cannot be reached
    (the envelope itself rises too slowly -> smaller smooth_ms; or the search range is too short).
    """
    grid = np.arange(search_ms[0], search_ms[1] + 1e-9, step_ms)
    rises = np.array([expected_noise_rise(template, fs, r, rise_db=rise_db, onset_db=onset_db, win_ms=win_ms)
                      for r in grid])
    ramp = float(grid[np.argmin(np.abs(rises - target_rise_ms))])
    note = ''
    if rises.min() > target_rise_ms + 1:
        note = (f'even the shortest ramp rises in {rises.min():.1f} ms (words: {target_rise_ms:.1f} ms): '
                f'the envelope itself is too slow, try a smaller smooth_ms')
    elif rises.max() < target_rise_ms - 1:
        note = (f'even the longest ramp rises in {rises.max():.1f} ms (words: {target_rise_ms:.1f} ms): '
                f'widen search_ms')
    return ramp, expected_noise_rise(template, fs, ramp, rise_db=rise_db, onset_db=onset_db, win_ms=win_ms), note


# ---------------------------------------------------------------- token pool and level

def make_token_pool(n_tokens, fir, envelope, rms, seed, *, carrier, fs, f0_hz, f_max_hz, phases):
    """n_tokens bursts: carrier x envelope, each scaled to exactly `rms`.

    carrier 'noise'    : make_noise, a different token each time (same seed -> same pool).
    carrier 'harmonic' : make_harmonic (f0_hz, f_max_hz, phases); all tokens are the same sound.
    Returns a list of float arrays (len(envelope) samples each).
    """
    rng = np.random.default_rng(seed)
    tokens = []
    for _ in range(n_tokens):
        if carrier == 'noise':
            x = make_noise(len(envelope), fir, rng)
        elif carrier == 'harmonic':
            x = make_harmonic(len(envelope), fir, fs, f0_hz=f0_hz, f_max_hz=f_max_hz, phases=phases)
        else:
            raise ValueError("carrier must be 'noise' or 'harmonic'")
        y = x * envelope
        tokens.append(y * rms / np.sqrt(np.mean(y ** 2)))
    return tokens


def shared_gain(signals, max_peak):
    """Gain (<= 1) that keeps the largest peak of all signals at or below max_peak.

    Pass the noise tokens AND the words, and apply the gain to both, so their RMS match is kept.
    """
    peak = max(float(np.abs(s).max()) for s in signals)
    return min(1.0, max_peak / peak)


# ---------------------------------------------------------------- all at once

def make_noise_pool(reference, *, fs, onset_thresh_db, env_win_ms, carrier, f0_hz, phases,
                    duration_ms, spectrum, smooth_oct, band_hz, fir_taps,
                    envelope, env_smooth_ms, ramp_ms, match_rise, rise_db, rise_win_ms,
                    ramp_search_ms, ramp_step_ms,
                    level, level_db, max_peak, n_tokens, seed):
    """Everything above in one call.

    reference   : list of word files or a glob pattern
    carrier     : 'noise' or 'harmonic'; f0_hz and phases are used only for 'harmonic' (None otherwise).
                  The harmonics go up to band_hz[1] (or fs/2 if band_hz is None).
    duration_ms : burst duration; None = mean audible duration of the words
    envelope    : 'word' (mean word envelope x ramps) or 'ramp' (flat burst x ramps)
    ramp_ms     : on/off ramp; with match_rise=True it is searched instead (ramp_search_ms, ramp_step_ms)
    level       : 'match' = mean RMS of the words, or a number (RMS); plus level_db
    max_peak    : words AND noise are turned down by one shared gain if anything would peak above it

    Returns a dict:
      tokens   : list of float32 arrays, already multiplied by `gain`
      gain     : the shared gain - multiply the WORDS by it too when you build sequences
      onset, offset : onset/offset sample of every token (same for all: taken from the expected envelope,
                      with the words' rule, so random fluctuations can't move a token on the grid)
      envelope : the burst envelope
      info     : settings and measured values (for a json file)
    """
    words = load_reference_words(reference, fs=fs, onset_thresh_db=onset_thresh_db, env_win_ms=env_win_ms)
    noise_ms = mean_sound_ms(words) if duration_ms is None else float(duration_ms)
    n_dur = int(round(noise_ms / 1000 * fs))
    rms = (mean_sound_rms(words) if level == 'match' else float(level)) * 10 ** (level_db / 20)

    f, p = long_term_spectrum(words)
    fir = shaping_filter(f, target_spectrum(f, p, spectrum=spectrum, smooth_oct=smooth_oct, band_hz=band_hz),
                         fs, n_taps=fir_taps)

    if envelope == 'word':
        template = word_envelope_template(words, n_dur, smooth_ms=env_smooth_ms)
    elif envelope == 'ramp':
        template = np.ones(n_dur)
    else:
        raise ValueError("envelope must be 'word' or 'ramp'")

    w_rise = word_rise_ms(words, rise_db=rise_db, onset_db=onset_thresh_db, win_ms=rise_win_ms)
    note = ''
    if match_rise:
        ramp_ms, n_rise, note = match_ramp(template, fs, float(w_rise.mean()), search_ms=ramp_search_ms,
                                           step_ms=ramp_step_ms, rise_db=rise_db, onset_db=onset_thresh_db,
                                           win_ms=rise_win_ms)
    else:
        n_rise = expected_noise_rise(template, fs, ramp_ms, rise_db=rise_db, onset_db=onset_thresh_db,
                                     win_ms=rise_win_ms)
    env = burst_envelope(template, fs, ramp_ms)

    f_max = band_hz[1] if band_hz is not None else fs / 2 - 1
    raw = make_token_pool(n_tokens, fir, env, rms, seed, carrier=carrier, fs=fs, f0_hz=f0_hz,
                          f_max_hz=f_max, phases=phases)
    gain = shared_gain(raw + [w['y'] for w in words], max_peak)
    tokens = [(y * gain).astype(np.float32) for y in raw]
    onset, offset = find_on_offset(env, fs, onset_thresh_db, env_win_ms)

    word_rms = np.array([np.sqrt(np.mean(sound_part(w) ** 2)) for w in words]) * gain
    tok_rms = np.array([np.sqrt(np.mean(t[onset:offset + 1].astype(float) ** 2)) for t in tokens])
    info = dict(
        n_reference_words=len(words), carrier=carrier, f0_hz=f0_hz, phases=phases,
        n_tokens=n_tokens, seed=seed, fs=fs, duration_ms=round(noise_ms, 1),
        spectrum=spectrum, smooth_oct=f'1/{smooth_oct}', band_hz=band_hz, fir_taps=fir_taps,
        envelope=envelope, env_smooth_ms=env_smooth_ms, ramp_ms=ramp_ms, rise_db=rise_db,
        word_rise_ms_mean=round(float(w_rise.mean()), 2),
        word_rise_ms_range=[round(float(w_rise.min()), 2), round(float(w_rise.max()), 2)],
        noise_rise_ms=round(float(n_rise), 2), rise_note=note,
        rms_target=round(rms, 5), level_db=level_db, shared_gain=round(gain, 5),
        shared_gain_db=round(20 * np.log10(gain), 2),
        rms_noise_minus_words_db=round(float(20 * np.log10(tok_rms.mean() / word_rms.mean())), 2),
        peak_factor_tokens_db=round(float(np.mean([20 * np.log10(np.abs(t).max() / r) for t, r in zip(tokens, tok_rms)])), 2),
        onset_ms=round(onset / fs * 1000, 2), offset_ms=round(offset / fs * 1000, 2),
    )
    return dict(tokens=tokens, gain=gain, onset=onset, offset=offset, envelope=env, info=info,
                words=words, word_rise_ms=w_rise, spectrum=(f, p), fir=fir)


def save_token_pool(pool, out_dir):
    """Write the tokens of make_noise_pool as pool_<carrier>_tokNN_<dur>ms.wav plus noise_settings.json.

    Returns the list of token file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    fs, dur = pool['info']['fs'], pool['info']['duration_ms']
    paths = []
    for k, y in enumerate(pool['tokens'], start=1):
        path = os.path.join(out_dir, f"pool_{pool['info']['carrier']}_tok{k:02d}_{dur:.0f}ms.wav")
        sf.write(path, y, fs)
        paths.append(path)
    with open(os.path.join(out_dir, 'noise_settings.json'), 'w') as fh:
        json.dump(pool['info'], fh, indent=2)
    return paths
