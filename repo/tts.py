"""Step 1 - synthesize one word: gTTS (female German voice only) or edge-tts (Microsoft voices, male or female)."""
import os
import subprocess
import sys
import time

from .audio import load_audio, resample, save_audio


def synthesize_word(text, wav_path, *, lang, tld, slow, sr, retries, pause_s):
    """Synthesize `text` with gTTS and save it as a mono 16-bit WAV at `sr` Hz.

    gTTS delivers MP3; it is decoded, resampled to `sr` and the MP3 is deleted.
    On failure the request is retried up to `retries` times, waiting pause_s * 2**attempt
    between attempts; after a successful request it waits `pause_s` (avoids rate limiting).

    Returns wav_path. Raises the last error if every attempt failed.
    """
    from gtts import gTTS        # imported here, so the rest of the repo works without gTTS installed
    mp3_path = os.path.splitext(wav_path)[0] + '.mp3'
    for attempt in range(1, retries + 1):
        try:
            gTTS(text=text, lang=lang, tld=tld, slow=slow).save(mp3_path)
            y, fs = load_audio(mp3_path)
            y, fs = resample(y, fs, sr)
            save_audio(wav_path, y, fs, subtype='PCM_16')
            os.remove(mp3_path)
            time.sleep(pause_s)
            return wav_path
        except Exception:
            if attempt == retries:
                raise
            time.sleep(pause_s * 2 ** attempt)


def synthesize_word_edge(text, wav_path, *, voice, rate, sr, retries, pause_s):
    """Synthesize `text` with edge-tts (Microsoft neural voices) and save it as a mono 16-bit WAV at `sr` Hz.

    voice : e.g. 'de-DE-ConradNeural', 'de-DE-KillianNeural', 'de-DE-KlausNeural', 'de-DE-RalfNeural' (male),
            'de-DE-KatjaNeural', 'de-DE-AmalaNeural' (female). List them all:  edge-tts --list-voices
    rate  : speaking rate relative to normal, e.g. '+0%' (normal) or '-30%' (slower).
    Needs  pip install edge-tts  and an internet connection. Runs the edge-tts command-line tool,
    so it also works inside Jupyter.

    Same retries / pause_s behaviour as synthesize_word. Returns wav_path.
    """
    mp3_path = os.path.splitext(wav_path)[0] + '.mp3'
    cmd = [sys.executable, '-m', 'edge_tts', '--voice', voice, f'--rate={rate}', '--text', text,
           '--write-media', mp3_path]
    for attempt in range(1, retries + 1):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            y, fs = load_audio(mp3_path)
            y, fs = resample(y, fs, sr)
            save_audio(wav_path, y, fs, subtype='PCM_16')
            os.remove(mp3_path)
            time.sleep(pause_s)
            return wav_path
        except subprocess.CalledProcessError as e:
            if attempt == retries:
                raise RuntimeError(f'edge-tts failed for {text!r}: {e.stderr.strip()[-300:]}') from None
            time.sleep(pause_s * 2 ** attempt)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(pause_s * 2 ** attempt)
