"""Step 3 - discrete sine-wave speech.

Thin wrappers only: all SWS work is done by YOUR stim_helpers.py (imported unchanged, it must be
importable, e.g. in the same folder as the notebook or on the Python path). Nothing here
re-implements or alters it.
"""
import numpy as np

from .audio import resample


def _helpers():
    # imported only when an SWS function is called, so TTS / trimming work without stim_helpers.py
    import stim_helpers
    return stim_helpers


def extract_tracks(wav_path, *, time_step, win_length, max_formants, male_female,
                   pitch_floor, pitch_ceiling, f0_smooth_sigma, mask_edge_sigma_ms):
    """Formants, per-formant amplitudes and voicing mask of a (trimmed) word.

    Calls stim_helpers.extract_formants_and_masks. Returns its dict
    (y, fs, n_samples, formants_up, amps_up, voiced_mask, ...).
    """
    return _helpers().extract_formants_and_masks(
        wav_path,
        time_step=time_step, win_length=win_length, max_formants=max_formants,
        male_female=male_female, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling,
        f0_smooth_sigma=f0_smooth_sigma, mask_edge_sigma_ms=mask_edge_sigma_ms,
        fs_override=None,
    )


def make_sws(drive, fs, *, max_formants, win_time):
    """Raw discrete SWS from tracks (equalize.stretch_tracks or equalize.natural_tracks).

    Calls stim_helpers.discrete_sws (cue_type=1, no reversal).
    Returns (signal, tracks) - tracks = discrete_sws's per-formant dict, for figures.
    The signal is NOT trimmed, faded or normalized yet: next are trim.trim_sws and finish_sws.
    """
    sig, tracks = _helpers().discrete_sws(
        formants_up=drive['formants_up'], amps_up=drive['amps_up'], voiced_mask=drive['voiced_mask'],
        n_samples=drive['n_samples'], max_formants=max_formants, win_time=win_time, fs=fs,
        cue_type=1, reverse_track=None, return_tracks=list(range(max_formants)), return_waveforms=True,
    )
    if not np.any(np.abs(sig) > 0):
        raise ValueError('no voiced frames: SWS output would be silent')
    return sig, tracks


def finish_sws(y, fs, *, fade_ms, target_rms, peak, out_sr):
    """Cosine fade (stim_helpers.apply_cosine_fade), RMS then peak normalization
    (stim_helpers.norm_audio), resampling to out_sr. Returns (y, out_sr) as float32."""
    helpers = _helpers()
    y = helpers.apply_cosine_fade(y, fs=fs, fade_ms=fade_ms)
    y = helpers.norm_audio(y, target_rms, peak).astype(np.float32)
    return resample(y, fs, out_sr)
