"""Word -> gTTS -> trim -> (equalize) -> discrete SWS (via stim_helpers) -> trim; word-shaped noise.

Functions only; every setting is passed in by the caller.
"""
from .audio import find_on_offset, load_audio, resample, rms_envelope, save_audio
from .equalize import (drop_short_voicing, fill_nan, natural_tracks, resample_rows, sounding_ms, sounding_region,
                       stretch_tracks)
from .noise import make_noise_pool, save_token_pool
from .pipeline import make_sws_word
from .sws import extract_tracks, finish_sws, make_sws
from .trim import trim_silence, trim_sws
from .tts import synthesize_word, synthesize_word_edge
from .sequence import build_sequence, draw_token_order
