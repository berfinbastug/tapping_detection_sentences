import os
import glob
import numpy as np
import soundfile as sf
import parselmouth
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from scipy.signal import stft



def bandpass_filter(signal, fs, low, high, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)



def amplitude_envelope(signal):
    return np.abs(hilbert(signal))


def smooth_envelope(env, fs, sigma_ms=10):
    sigma_samples = int((sigma_ms / 1000) * fs)
    return gaussian_filter1d(env, sigma_samples)


def rms_normalize(x, target_rms=0.06, eps=1e-12):
    x = np.asarray(x, dtype=float)
    rms = np.sqrt(np.mean(x**2) + eps)
    return x * (target_rms / (rms + eps))


def peak_normalize(x, peak=0.95, eps=1e-12):
    x = np.asarray(x, dtype=float)
    m = np.max(np.abs(x)) + eps
    return x * (peak / m)


def apply_cosine_fade(x, fs, fade_ms=10):
    x = np.asarray(x, float).copy()
    n = len(x)
    m = int(round(fs * fade_ms / 1000))
    if m <= 1 or 2*m > n:
        return x

    ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, m)))
    x[:m] *= ramp
    x[-m:] *= ramp[::-1]
    return x


def norm_audio(x, target_rms=0.06, peak=0.95):
    x = rms_normalize(x, target_rms=target_rms)
    x = peak_normalize(x, peak=peak)
    return x



def upsample_amp_tracks(spec_times, amps_frames, t_samps, voiced_mask):
    n_samples = len(t_samps)
    max_formants = amps_frames.shape[1]
    amps_up = np.zeros((n_samples, max_formants), dtype=float)

    for k in range(max_formants):
        valid = ~np.isnan(amps_frames[:, k])
        if valid.sum() < 2:
            continue

        xp = spec_times[valid]
        fp = amps_frames[valid, k]

        a = np.interp(t_samps, xp, fp, left=0.0, right=0.0)

        # normalize each track
        if np.max(a) > 0:
            a = a / np.max(a)

        # optional compression
        a = np.sqrt(a)

        amps_up[:, k] = voiced_mask * a

    return amps_up




def upsample_track(frame_times, frame_values, sample_times, fill_edges=False):
    """
    Upsample a 1D track with NaNs allowed in frame_values.
    If fill_edges=False, outside valid support stays NaN.
    If fill_edges=True, edge values are held.
    UPSAMPLE PITCH AND FORMANT
    """
    out = np.full(sample_times.shape, np.nan, dtype=float)

    valid = ~np.isnan(frame_values)
    if valid.sum() < 2:
        return out

    xp = frame_times[valid]
    fp = frame_values[valid]

    out = np.interp(sample_times, xp, fp, left=fp[0], right=fp[-1])

    if not fill_edges:
        out[sample_times < xp[0]] = np.nan
        out[sample_times > xp[-1]] = np.nan

    return out



def extract_formant_amplitudes_from_spectrogram(
    y,
    fs,
    formant_times,
    formants,          # shape: [n_frames, max_formants]
    nperseg=512,
    noverlap=384,
    band_halfwidth=80, # Hz around each formant
):
    """
    Estimate per-formant amplitude tracks from spectrogram energy near each formant.
    Returns:
        spec_times: STFT frame times
        amps_frames: shape [n_spec_frames, max_formants]
    """
    f, t_spec, Zxx = stft(
        y,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        padded=False,
        boundary=None,
    )

    S = np.abs(Zxx)  # magnitude spectrogram
    max_formants = formants.shape[1]
    amps_frames = np.full((len(t_spec), max_formants), np.nan, dtype=float)

    # interpolate formant tracks from Praat frame times to STFT frame times
    formants_spec = np.full((len(t_spec), max_formants), np.nan, dtype=float)
    for k in range(max_formants):
        valid = ~np.isnan(formants[:, k])
        if valid.sum() >= 2:
            formants_spec[:, k] = np.interp(
                t_spec,
                formant_times[valid],
                formants[valid, k],
                left=np.nan,
                right=np.nan,
            )

    # measure energy around each formant
    for i in range(len(t_spec)):
        for k in range(max_formants):
            fk = formants_spec[i, k]
            if np.isnan(fk) or fk <= 0:
                continue

            band = (f >= fk - band_halfwidth) & (f <= fk + band_halfwidth)
            if not np.any(band):
                continue

            amps_frames[i, k] = np.mean(S[band, i])

    # smooth each amplitude track
    for k in range(max_formants):
        valid = ~np.isnan(amps_frames[:, k])
        if valid.sum() > 2:
            amps_frames[valid, k] = gaussian_filter1d(amps_frames[valid, k], sigma=1.0)

    return t_spec, amps_frames




def extract_formants_and_masks(
    wav_path,
    time_step=0.0125,
    win_length=0.025,
    max_formants=3,
    male_female=1,  # 0 male, 1 female
    pitch_floor = 75,
    pitch_ceiling = 500,
    f0_smooth_sigma = 1,
    mask_edge_sigma_ms = 5,
    fs_override=None,
):

    # -----------------------------
    # Load audio
    # -----------------------------
    y, fs = sf.read(wav_path)
    if y.ndim > 1:
        y = y[:, 0]

    # Optional sanity check: ensure the loaded audio sample rate matches the expected fs.
    if fs_override is not None and fs != fs_override:
        raise ValueError(f"Expected fs={fs_override}, got fs={fs} for {wav_path}")

    snd = parselmouth.Sound(y, fs)
    duration = snd.get_total_duration()

    # Use the actual audio length to define n_samples (more stable than int(duration*fs))
    n_samples = int(len(y))
    duration_samps = n_samples / fs  # can differ slightly from Praat duration

    # -----------------------------
    # Formants
    # -----------------------------
    maximum_formant_freq = 5000 if male_female == 0 else 5500
    formant_obj = snd.to_formant_burg(
        time_step=time_step,
        max_number_of_formants=5,
        window_length=win_length,
        maximum_formant=maximum_formant_freq,
    )

    # Build a times grid that won't overshoot duration used by sample grid
    duration_use = min(duration, duration_samps)
    n_frames = int(np.floor(duration_use / time_step)) + 1
    times = np.linspace(0, duration_use, n_frames)

    formants = np.full((len(times), max_formants), np.nan)
    for i, t in enumerate(times):
        for k in range(max_formants):
            f = formant_obj.get_value_at_time(k + 1, t)
            if f is not None and (not np.isnan(f)) and f > 0:
                formants[i, k] = f

    # Smooth (NaN-safe)
    formants_smooth = np.full_like(formants, np.nan)
    for k in range(max_formants):
        fk = formants[:, k]
        valid = ~np.isnan(fk)
        if valid.sum() > 2:
            formants_smooth[valid, k] = gaussian_filter1d(fk[valid], sigma=1.2)

    # Upsample to sample grid
    t_samps = np.arange(n_samples) / fs
    formants_up = np.full((n_samples, max_formants), np.nan, dtype=float)
    for k in range(max_formants):
        formants_up[:, k] = upsample_track(times, formants_smooth[:, k], t_samps, fill_edges=True)


    # -----------------------------
    # Pitch / F0
    # -----------------------------
    pitch = snd.to_pitch(
        time_step=time_step,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling,
        )
    
    pitch_times = pitch.xs()
    f0_frames = pitch.selected_array["frequency"].astype(float)
    f0_frames[f0_frames <= 0] = np.nan

    f0_smooth = np.full_like(f0_frames, np.nan)
    valid_f0 = ~np.isnan(f0_frames)
    if valid_f0.sum() > 2:
        f0_smooth[valid_f0] = gaussian_filter1d(f0_frames[valid_f0], sigma=f0_smooth_sigma)
    else:
        f0_smooth = f0_frames.copy()

    
    # voiced / unvoiced mask from pitch
    voiced_frames = (~np.isnan(f0_smooth)).astype(float)

    voiced_mask = np.interp(
        t_samps,
        pitch_times,
        voiced_frames,
        left=0.0,
        right=0.0
    )

    sigma = max(1, int((mask_edge_sigma_ms / 1000.0) * fs))
    voiced_mask = gaussian_filter1d(voiced_mask, sigma=sigma)
    voiced_mask = np.clip(voiced_mask, 0.0, 1.0)

    # -----------------------------
    # Per-formant amplitudes from spectrogram
    # -----------------------------
    spec_times, amps_frames = extract_formant_amplitudes_from_spectrogram(
        y=y,
        fs=fs,
        formant_times=times,
        formants=formants_smooth,
        nperseg=512,
        noverlap=384,
        band_halfwidth=80,
    )

    amps_up = upsample_amp_tracks(
        spec_times=spec_times,
        amps_frames=amps_frames,
        t_samps=t_samps,
        voiced_mask=voiced_mask)

    amps_up_smooth = amps_up.copy()
    for k in range(max_formants):
        amps_up_smooth[:, k] = gaussian_filter1d(amps_up_smooth[:, k], sigma=int(0.03 * fs))

    return {
        "wav_path": wav_path,
        "y": y,
        "fs": fs,
        "duration": duration_use,
        "n_samples": n_samples,
        "formants_up": formants_up,
        "amps_up": amps_up_smooth,
        "voiced_mask": voiced_mask,
    }




def reverse_blocks(
    f_step,
    fs,
    chunk_ms=50,
    block_ms=300,
    a_step=None,
    mask=None
):
    f_step = np.asarray(f_step, dtype=float)
    n = len(f_step)

    a_out = None
    m_out = None

    if a_step is not None:
        a_step = np.asarray(a_step, dtype=float)
        if len(a_step) != n:
            raise ValueError("a_step must have the same length as f_step")
        a_out = np.empty_like(a_step)

    if mask is not None:
        mask = np.asarray(mask, dtype=float)
        if len(mask) != n:
            raise ValueError("mask must have the same length as f_step")
        m_out = np.empty_like(mask)

    chunk = max(1, int(round(chunk_ms * fs / 1000)))
    block = max(chunk, int(round(block_ms * fs / 1000)))

    f_out = np.empty_like(f_step)
    boundary = np.zeros(n, dtype=bool)

    for b0 in range(0, n, block):
        b1 = min(n, b0 + block)

        if b0 > 0:
            boundary[b0] = True

        chunks = []
        for c0 in range(b0, b1, chunk):
            c1 = min(b1, c0 + chunk)
            chunks.append((c0, c1))

        chunks = chunks[::-1]

        write_pos = b0
        for i, (c0, c1) in enumerate(chunks):
            L = c1 - c0

            f_out[write_pos:write_pos + L] = f_step[c0:c1]
            if a_out is not None:
                a_out[write_pos:write_pos + L] = a_step[c0:c1]
            if m_out is not None:
                m_out[write_pos:write_pos + L] = mask[c0:c1]

            if i > 0:
                boundary[write_pos] = True

            write_pos += L

    outs = [f_out, boundary]
    if a_out is not None:
        outs.append(a_out)
    if m_out is not None:
        outs.append(m_out)
    return tuple(outs)



def render_track_from_steps(
    f_step,
    fs,
    mask,
    boundary_hint,        # joins from shuffling (output-time indices)
    a_step = None,
    a_smooth = None,
    transition_ms=10,       # MUST match discrete_sws
    start_phase_at_zero = True,
):
    """
    Render from stepped (per-sample expanded) f_step and a_step, reproducing
    the same 10 ms transition smoothing used in discrete_sws.

    This is the main reason your “edges” come back: without this,
    rendering from f_step directly creates discontinuities.
    """

    f_step = np.asarray(f_step, dtype=float)
    mask = np.asarray(mask, dtype=float)

    n = len(f_step)
    if len(mask) != n:
        raise ValueError("f_step and mask must have same length")

    # DECIDE WHICH AMPLITUDE TO USE
    if a_step is not None:
        a_step = np.asarray(a_step, dtype = float)
        if len(a_step) != n:
            raise ValueError("a_step must have same length as f_step")
        a_used = a_step.copy()
        smooth_a = True
    else:
        if a_smooth is None:
            raise ValueError("provie either a_step or a_smooth")
        a_smooth = np.asarray(a_smooth, dtype=float)
        if len(a_smooth) != n:
            raise ValueError("a_smooth must have same length as f_step")
        a_used = a_smooth
        smooth_a = False

    # prepare join indices
    bh = np.asarray(boundary_hint, dtype=bool)
    if len(bh) != n:
        raise ValueError("boundary_hint must have same length as f_step")

    join_idx = np.where(bh)[0]

    trans_samp = max(1, int(round(transition_ms * fs / 1000)))

    # Cosine ramp 0->1 (same shape as your discrete_sws)
    ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, trans_samp, endpoint=False)))

    f_smooth = f_step.copy()
    
    for j in join_idx:
        start = j
        end = min(start + trans_samp, n)
        L = end - start
        if L <= 2:
            continue
        rr = ramp[:L]
        pre = max(0, start - 1)
        f0, f1 = f_step[pre], f_step[start] 
        f_smooth[start:end] = f0 + rr * (f1 - f0)
        
        if smooth_a:
            a0, a1 = a_step[pre], a_step[start]
            a_used[start:end] = a0 + rr * (a1 - a0)


    # Render tone
    dphi = 2 * np.pi * f_smooth / fs
    phase_track = np.cumsum(dphi)

    if start_phase_at_zero:
        # shift so phase[0] = 0 tone[0] = 0 (helps onset click)
        # whatever, i don't understand
        phase_track = np.concatenate([[0.0], phase_track[:-1]])

    tone = a_used * np.sin(phase_track)
    
    tone *= mask

    return tone



def discrete_sws(
    formants_up,
    amps_up,
    voiced_mask,
    n_samples,
    max_formants = 3,
    win_time=0.05,
    fs=16000,
    cue_type = 1,
    reverse_track = None,
    return_tracks=None,
    return_waveforms=True,
):

    if return_tracks is not None:
        return_tracks = list(return_tracks)
        for k in return_tracks:
            if not (0 <= k < max_formants):
                raise ValueError(f"Invalid track index {k}; must be 0..{max_formants-1}")

    win_samples = int(round(win_time * fs))
    win_samples = max(1, win_samples)

    transition_ms = 10
    transition_samples = int(round(fs * transition_ms / 1000))
    transition_samples = max(1, transition_samples)

    ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, transition_samples)))

    sig_mix = np.zeros(n_samples, dtype=float)

    tracks = {}
    if return_tracks is not None:
        for kk in return_tracks:
            tracks[kk] = {
                "f_step": None, # final one being used
                "f_step_original": None, # original unreversed
                "a_step": None,
                "f_smooth": None, # final smooth if applicable smooth corresponds to transitions
                "f_smooth_original": None, # original unreversed smooth
                "a_smooth": None,  # smooth corresponds to transitions
                "mask": None,
                "boundary_hint": None,
                "was_reversed": False,
            }
            if return_waveforms:
                tracks[kk]["waveform"] = None
                tracks[kk]["waveform_original"] = None


    for k in range(max_formants):
        fk = np.asarray(formants_up[:, k], dtype=float)
        ak = np.asarray(amps_up[:, k], dtype=float)
        mask_k = np.asarray(voiced_mask, dtype=float)

        # Ensure mask length matches
        if len(mask_k) >= n_samples:
            mask_k = mask_k[:n_samples]
        else:
            mask_k = np.pad(mask_k, (0, n_samples - len(mask_k)))

        # If this track is never active, skip
        if np.max(ak) <= 0:
            continue

        f_stepped = np.zeros(n_samples, dtype=float)
        a_stepped = np.zeros(n_samples, dtype=float)

        # Initialize prev_f from first valid frequency if present
        pos = fk[fk > 0]
        prev_f = float(pos[0]) if pos.size else 0.0

        for start in range(0, n_samples, win_samples):
            end = min(start + win_samples, n_samples)

            seg_f = fk[start:end]
            seg_a = ak[start:end]

            valid_f = seg_f[seg_f > 0]
            if valid_f.size > 0:
                avg_f = float(np.mean(valid_f))
                prev_f = avg_f
            else:
                avg_f = prev_f  # hold-last instead of 0

            # If amplitude is all zeros in this window, keep avg_a at 0
            avg_a = float(np.mean(seg_a)) if np.any(seg_a > 0) else 0.0

            f_stepped[start:end] = avg_f
            a_stepped[start:end] = avg_a

        # BUILD THE ORIGINAL FIRST            
        # Smooth transitions between windows
        f_smooth = f_stepped.copy()
        a_smooth = a_stepped.copy()

        for start in range(win_samples, n_samples, win_samples):
            end = min(start + transition_samples, n_samples)
            if end <= start:
                continue
            f_smooth[start:end] = (
                f_stepped[start - 1]
                + ramp[: end - start] * (f_stepped[start] - f_stepped[start - 1])
            )
            a_smooth[start:end] = (
                a_stepped[start - 1]
                + ramp[: end - start] * (a_stepped[start] - a_stepped[start - 1])
            )
        
        # save originals before any reversal
        original_f_step = f_stepped.copy()
        original_f_smooth = f_smooth.copy()
        boundary_hint = np.zeros(n_samples, dtype = bool)
        was_reversed = False

        phase_track_original = np.cumsum(2 * np.pi * original_f_smooth / fs)
        tone_original = a_smooth * np.sin(phase_track_original)
        tone_original *= mask_k
        
        # --------------------------------------------------
        # REVERSED TARGET TRACK: reverse only frequency steps,
        # then render using original mask + original a_smooth
        # --------------------------------------------------
        if cue_type == 0 and k == reverse_track:
            block_ms = (n_samples/fs) * 1000.0
            reversed_f_step, boundary_hint = reverse_blocks(
                f_stepped,
                fs,
                chunk_ms = win_time * 1000.0,
                block_ms= block_ms
            )
            tone = render_track_from_steps(
                f_step = reversed_f_step,
                fs = fs,
                mask = mask_k,
                boundary_hint=boundary_hint,
                a_smooth=a_smooth,
                transition_ms=transition_ms,
                start_phase_at_zero=False
            )
            final_f_step = reversed_f_step
            final_f_smooth = None
            was_reversed = True
        else:
            tone = tone_original.copy()
            final_f_step = f_stepped
            final_f_smooth = f_smooth

        sig_mix += tone

        # Return requested track info
        if return_tracks is not None and k in return_tracks:
            tracks[k]["f_step"] = final_f_step.copy()
            tracks[k]["f_step_original"] = original_f_step.copy()
            tracks[k]["a_step"] = a_stepped.copy()
            tracks[k]["f_smooth"] = None if final_f_smooth is None else final_f_smooth.copy()
            tracks[k]["f_smooth_original"] = original_f_smooth.copy()
            tracks[k]["a_smooth"] = a_smooth.copy()
            tracks[k]["mask"] = mask_k.copy()
            tracks[k]["boundary_hint"] = boundary_hint.copy()
            tracks[k]["was_reversed"] = was_reversed
            if return_waveforms:
                tracks[k]["waveform"] = tone.copy() # if it is reversed this is the reversed version 
                tracks[k]["waveform_original"] = tone_original.copy()  # if it is reversed then this is the unreversed version
    if return_tracks is None:
        return sig_mix
    return sig_mix, tracks

