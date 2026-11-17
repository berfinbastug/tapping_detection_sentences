"""Step 5 - equalize word durations.

Equalization works on the TRACKS (formants, amplitudes, voicing mask), not on the audio:
the tracks are cut to the sounding (voiced) region and linearly resampled to the target duration,
and only then turned into discrete SWS. So the 50 ms steps stay 50 ms long - a stretched word
just gets more steps. Stretching the finished SWS audio instead would stretch the steps too.

Order in the pipeline:  extract tracks -> stretch_tracks (or natural_tracks) -> SWS -> trim_sws.
"""
import numpy as np


def sounding_region(voiced_mask, threshold):
    """First and last+1 sample where the voicing mask is above threshold. Returns (a, b)."""
    idx = np.flatnonzero(np.asarray(voiced_mask) > threshold)
    if idx.size == 0:
        raise ValueError('no voiced frames: SWS output would be silent')
    return int(idx[0]), int(idx[-1]) + 1


def sounding_ms(tracks, threshold):
    """Duration (ms) of the sounding region of tracks from stim_helpers.extract_formants_and_masks."""
    a, b = sounding_region(tracks['voiced_mask'], threshold)
    return (b - a) / int(tracks['fs']) * 1000


def fill_nan(a):
    """Linear interpolation over NaNs, column by column (formant gaps). All-NaN columns -> 0."""
    a = np.array(a, dtype=float)
    cols = a.reshape(len(a), -1)
    x = np.arange(len(a))
    for k in range(cols.shape[1]):
        bad = np.isnan(cols[:, k])
        if bad.all():
            cols[:, k] = 0.0
        elif bad.any():
            cols[bad, k] = np.interp(x[bad], x[~bad], cols[~bad, k])
    return cols.reshape(a.shape)


def resample_rows(a, n_new):
    """Linearly resample an (n,) or (n, k) array to n_new rows."""
    a = np.asarray(a, dtype=float)
    x_old = np.linspace(0, 1, len(a))
    x_new = np.linspace(0, 1, n_new)
    if a.ndim == 1:
        return np.interp(x_new, x_old, a)
    return np.column_stack([np.interp(x_new, x_old, a[:, k]) for k in range(a.shape[1])])


def drop_short_voicing(tracks, *, threshold, min_voiced_ms):
    """Remove short, isolated voiced islands from the voicing mask.

    The pitch tracker sometimes marks a stray frame as voiced (a click or breath before the word).
    Such an island is separated from the word by silence (mask <= threshold), so stretch_tracks
    would count it as the start of the word, and the SWS gets a blip + gap before the word.

    An island = a run of mask > threshold. It is removed (mask set to 0) if it contains less than
    min_voiced_ms of clearly voiced signal (mask > 0.5). Real voicing in a word is far longer.

    Returns (new_tracks, removed): new_tracks is a copy with the cleaned voiced_mask;
    removed is a list of (start_ms, end_ms) of the islands that were removed (empty = unchanged).
    """
    fs = int(tracks['fs'])
    mask = np.asarray(tracks['voiced_mask'], dtype=float)
    above = mask > threshold
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]                      # island k = mask[starts[k]:ends[k]]
    new_mask = mask.copy()
    removed = []
    for a, b in zip(starts, ends):
        voiced_ms = np.count_nonzero(mask[a:b] > 0.5) / fs * 1000
        if voiced_ms < min_voiced_ms:
            new_mask[a:b] = 0.0
            removed.append((round(float(a) / fs * 1000, 1), round(float(b) / fs * 1000, 1)))
    if removed and not np.any(new_mask > threshold):
        raise ValueError('all voicing would be removed: min_voiced_ms is too large for this word')
    new_tracks = dict(tracks)
    new_tracks['voiced_mask'] = new_mask
    return new_tracks, removed


def natural_tracks(tracks):
    """The tracks as extracted, in the form stretch_tracks returns (no equalization)."""
    return dict(formants_up=tracks['formants_up'], amps_up=tracks['amps_up'],
                voiced_mask=tracks['voiced_mask'], n_samples=int(tracks['n_samples']))


def stretch_tracks(tracks, target_ms, *, alpha, max_stretch, mask_threshold):
    """Cut the tracks to their sounding region and resample that region to target_ms.

    alpha        : 1 = exactly target_ms; 0.5 = halfway on a log scale; 0 = no change.
    max_stretch  : the stretch factor is clipped to [1/max_stretch, max_stretch]; words that hit the
                   cap do NOT reach target_ms (check the returned factor).
    mask_threshold : voicing-mask value that defines the sounding region.

    Returns (drive, factor): drive has formants_up, amps_up, voiced_mask, n_samples and goes
    into sws.make_sws.
    """
    fs = int(tracks['fs'])
    a, b = sounding_region(tracks['voiced_mask'], mask_threshold)
    factor = (target_ms / ((b - a) / fs * 1000)) ** alpha
    factor = float(np.clip(factor, 1 / max_stretch, max_stretch))
    n_new = int(round((b - a) * factor))
    drive = dict(
        formants_up=resample_rows(fill_nan(tracks['formants_up'][a:b]), n_new),
        amps_up=np.clip(resample_rows(fill_nan(tracks['amps_up'][a:b]), n_new), 0, None),
        voiced_mask=np.clip(resample_rows(tracks['voiced_mask'][a:b], n_new), 0, 1),
        n_samples=n_new,
    )
    return drive, factor
