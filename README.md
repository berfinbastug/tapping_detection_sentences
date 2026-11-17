# sws_stimuli

Functions that turn a word into a discrete sine-wave-speech (SWS) WAV file. The package has no
settings of its own. Every value is a required keyword argument that you pass in.

All SWS synthesis comes from **your `stim_helpers.py`**, which the package imports unchanged.
It must be importable, for example in the same folder as your notebook.

## Order of the steps

| Step | Function | Module |
|---|---|---|
| 1. TTS | `synthesize_word(text, wav_path, ...)` | `tts.py` |
| 2. Trim TTS | `trim_silence(y, fs, ...)` | `trim.py` |
| Extract tracks | `extract_tracks(wav_path, ...)` calls `stim_helpers.extract_formants_and_masks` | `sws.py` |
| 5. Equalise | `stretch_tracks(tracks, target_ms, ...)`, or `natural_tracks(tracks)` | `equalize.py` |
| 3. SWS | `make_sws(drive, fs, ...)` calls `stim_helpers.discrete_sws` | `sws.py` |
| 4. Trim SWS | `trim_sws(y, fs, ...)` | `trim.py` |
| Finish | `finish_sws(y, fs, ...)`: fade, normalise and resample, via `stim_helpers` | `sws.py` |

Equalisation (step 5) runs **before** SWS synthesis (step 3). It stretches the formant, amplitude
and voicing tracks rather than the audio, so the 50 ms steps stay 50 ms long.

`make_sws_word(text, out_dir, base, **settings)` in `pipeline.py` chains all of the above for
one word, writes `1_tts/`, `2_trimmed/` and `3_sws/`, and returns one row of a words table.

## Word-shaped noise (`noise.py`)

`make_noise_pool(reference, ...)` makes a pool of noise tokens matched to a set of reference words, for
example `'sws_words/3_sws/*_dur400ms.wav'`. The tokens match the words' long-term spectrum (smoothed
to 1/N octave and band-limited), their mean audible duration and envelope, their mean rise time and their
mean RMS. `save_token_pool(pool, out_dir)` writes the tokens and `noise_settings.json`. Each step is
also a separate function: `load_reference_words`, `long_term_spectrum`, `target_spectrum`,
`shaping_filter`, `word_envelope_template`, `match_ramp`, `make_token_pool` and `shared_gain`.

`pool['gain']` is one gain shared by the noise and the words. The tokens already have it applied.
**Multiply the words by it too** when you build the sequences, or the noise–word RMS match is lost.

## Install

```
pip install -e .        # from this folder
```
