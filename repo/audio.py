"""Small audio utilities shared by the other modules."""
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path):
    """Read a sound file as mono float32. MP3 goes through pydub if soundfile cannot read it.

    Returns (y, fs).
    """
    try:
        y, fs = sf.read(path, dtype='float32')
    except Exception:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(path).set_channels(1)
        y = np.array(seg.get_array_of_samples(), dtype=np.float32) / float(1 << (8 * seg.sample_width - 1))
        fs = seg.frame_rate
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, int(fs)


def save_audio(path, y, fs, subtype=None):
    """Write y to path (subtype e.g. 'PCM_16'; None = soundfile's default for the format)."""
    sf.write(path, y, int(fs), subtype=subtype)
    return path


def resample(y, fs_in, fs_out):
    """Polyphase resampling from fs_in to fs_out. Returns (y, fs_out)."""
    y = np.asarray(y, dtype=np.float32).squeeze()
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_in == fs_out:
        return y, fs_in
    g = math.gcd(fs_in, fs_out)
    return resample_poly(y, fs_out // g, fs_in // g).astype(np.float32), fs_out


def rms_envelope(y, fs, win_ms):
    """Moving-RMS envelope with a win_ms rectangular window (same length as y)."""
    n = max(1, int(round(win_ms / 1000 * fs)))
    return np.sqrt(np.convolve(np.asarray(y, float) ** 2, np.ones(n) / n, mode='same'))


def find_on_offset(y, fs, thresh_db, env_win_ms):
    """First and last sample where the RMS envelope exceeds (envelope peak + thresh_db).

    Returns (onset_sample, offset_sample).
    """
    env = rms_envelope(y, fs, env_win_ms)
    above = np.flatnonzero(env > env.max() * 10 ** (thresh_db / 20))
    if above.size == 0:
        raise ValueError('signal is silent')
    return int(above[0]), int(above[-1])
