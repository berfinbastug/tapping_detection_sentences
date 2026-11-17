"""All five steps for one word, chained.

    1 synthesize_word -> 2 trim_silence -> extract_tracks -> 5 stretch_tracks / natural_tracks
      -> 3 make_sws -> 4 trim_sws -> finish_sws

Equalization (5) sits before SWS synthesis (3) on purpose: see equalize.py.
Every setting is a required keyword argument, so you can keep them in one dict and call
make_sws_word(text, out_dir, base, **settings).
"""
import os

import numpy as np

from .audio import load_audio, save_audio
from .equalize import natural_tracks, sounding_ms, stretch_tracks
from .sws import extract_tracks, finish_sws, make_sws
from .trim import trim_silence, trim_sws
from .tts import synthesize_word


def make_sws_word(
    text, out_dir, base, *,
    # 1 TTS
    tts_lang, tts_tld, tts_slow, tts_retries, tts_pause_s, overwrite_tts, sr,
    # 2 trim TTS
    trim_thresh_rel_db, trim_chunk_ms, trim_pad_ms,
    # track extraction (stim_helpers)
    time_step, win_length, max_formants, male_female, pitch_floor, pitch_ceiling,
    f0_smooth_sigma, mask_edge_sigma_ms,
    # 5 equalize (target_ms=None -> natural duration)
    target_ms, stretch_alpha, max_stretch, mask_threshold,
    # 3 SWS
    win_time,
    # 4 trim SWS
    sws_trim_thresh_db, sws_env_win_ms, sws_trim_pad_ms,
    # finishing
    fade_ms, target_rms, peak,
):
    """Make one SWS word file from text.

    Files: out_dir/1_tts/<base>.wav, out_dir/2_trimmed/<base>.wav,
           out_dir/3_sws/<base>_sws<win>ms[_dur<target>ms].wav

    Returns a dict with the file paths, durations, the stretch factor and the voiced fraction;
    one dict per word -> pandas.DataFrame gives a words table.
    """
    dirs = {k: os.path.join(out_dir, k) for k in ('1_tts', '2_trimmed', '3_sws')}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 1 TTS
    tts_path = os.path.join(dirs['1_tts'], base + '.wav')
    if overwrite_tts or not os.path.exists(tts_path):
        synthesize_word(text, tts_path, lang=tts_lang, tld=tts_tld, slow=tts_slow,
                        sr=sr, retries=tts_retries, pause_s=tts_pause_s)

    # 2 trim TTS
    y, fs = load_audio(tts_path)
    y_trim, t0, t1 = trim_silence(y, fs, thresh_rel_db=trim_thresh_rel_db,
                                  chunk_ms=trim_chunk_ms, pad_ms=trim_pad_ms)
    trim_path = save_audio(os.path.join(dirs['2_trimmed'], base + '.wav'), y_trim, fs, subtype='PCM_16')

    # tracks
    tracks = extract_tracks(trim_path, time_step=time_step, win_length=win_length,
                            max_formants=max_formants, male_female=male_female,
                            pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling,
                            f0_smooth_sigma=f0_smooth_sigma, mask_edge_sigma_ms=mask_edge_sigma_ms)
    fs_tr = int(tracks['fs'])

    # 5 equalize
    if target_ms is None:
        drive, factor, dur_tag = natural_tracks(tracks), 1.0, ''
    else:
        drive, factor = stretch_tracks(tracks, target_ms, alpha=stretch_alpha,
                                       max_stretch=max_stretch, mask_threshold=mask_threshold)
        dur_tag = f'_dur{target_ms:.0f}ms'

    # 3 SWS
    sig, _ = make_sws(drive, fs_tr, max_formants=max_formants, win_time=win_time)

    # 4 trim SWS, then fade / normalize / resample
    sig_trim, s0, s1 = trim_sws(sig, fs_tr, thresh_db=sws_trim_thresh_db,
                                env_win_ms=sws_env_win_ms, pad_ms=sws_trim_pad_ms)
    out, fs_out = finish_sws(sig_trim, fs_tr, fade_ms=fade_ms, target_rms=target_rms, peak=peak, out_sr=sr)
    sws_path = save_audio(os.path.join(dirs['3_sws'], f'{base}_sws{int(round(win_time * 1000))}ms{dur_tag}.wav'),
                          out, fs_out)

    return dict(
        word=text, base=base, tts_file=tts_path, trim_file=trim_path, sws_file=sws_path,
        tts_ms=round(len(y) / fs * 1000), trimmed_ms=round(len(y_trim) / fs * 1000),
        natural_sound_ms=round(sounding_ms(tracks, mask_threshold)),
        stretch=round(factor, 3), stretch_capped=bool(target_ms is not None and
                                                      (factor >= max_stretch - 1e-6 or factor <= 1 / max_stretch + 1e-6)),
        driven_ms=round(drive['n_samples'] / fs_tr * 1000),
        sws_cut_start_ms=round(s0 / fs_tr * 1000), sws_ms=round(len(out) / fs_out * 1000),
        voiced_frac=round(float(np.mean(drive['voiced_mask'] > 0.5)), 2),
    )
