"""Steps 2 and 4 - trimming.

trim_silence : step 2, cuts the silence around the raw TTS word.
trim_sws     : step 4, cuts the SWS word to its sounding part. Needed because the voicing mask
               silences unvoiced sounds (/ʃ/ of 'Schnee', /t/ of 'Licht'), so the SWS signal can
               start or end with silence although the TTS word was already trimmed.
"""
import numpy as np

from .audio import find_on_offset


def trim_silence(y, fs, *, thresh_rel_db, chunk_ms, pad_ms):
    """Cut leading and trailing silence from a TTS word.

    Silence = chunk_ms chunks whose RMS is below (the word's peak + thresh_rel_db) dB.
    pad_ms of the cut part is kept on each side so onsets/offsets aren't clipped.

    Returns (y_trimmed, start_sample, end_sample) with y_trimmed = y[start:end].
    """
    y = np.asarray(y)
    chunk = max(1, int(round(chunk_ms / 1000 * fs)))
    n = int(np.ceil(len(y) / chunk))
    yp = np.r_[y, np.zeros(n * chunk - len(y))].astype(float)
    rms = np.sqrt(np.mean(yp.reshape(n, chunk) ** 2, axis=1))
    loud = 20 * np.log10(rms / (np.abs(y).max() + 1e-12) + 1e-12) >= thresh_rel_db
    if not loud.any():
        return y, 0, len(y)
    first = int(np.argmax(loud)) * chunk
    last = min(len(y), (n - int(np.argmax(loud[::-1]))) * chunk)
    pad = int(round(pad_ms / 1000 * fs))
    start, end = max(0, first - pad), min(len(y), last + pad)
    return y[start:end], start, end


def trim_sws(y, fs, *, thresh_db, env_win_ms, pad_ms):
    """Cut an SWS signal to its sounding part.

    Sounding part = RMS envelope (env_win_ms window) above (envelope peak + thresh_db) dB.
    pad_ms is kept on each side. Do this BEFORE fading and normalizing (see sws.finish_sws),
    so the RMS is computed over the sounding part.

    Returns (y_trimmed, start_sample, end_sample) with y_trimmed = y[start:end].
    """
    on, off = find_on_offset(y, fs, thresh_db, env_win_ms)
    pad = int(round(pad_ms / 1000 * fs))
    start, end = max(0, on - pad), min(len(y), off + 1 + pad)
    return y[start:end], start, end
