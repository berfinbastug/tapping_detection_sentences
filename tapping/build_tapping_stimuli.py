# %% BUILD TAPPING STIMULI FOR ONE PARTICIPANT
"""
Builds the 216 tapping trials of ONE participant and saves everything in that participant's own folder.
Same rules as build_detection_stimuli.py: nothing is ever overwritten (if the folder exists the script stops),
everything is built in <folder>_incomplete and renamed only when every check has passed.

Inputs
    design_tables/tapping_<pid>_<group>.csv          trial level: which sentence (generate_tapping_design_table.py)
    design_tables/tapping_events_<pid>_<group>.csv   event level: WHEN each of the 30 sounds plays (shift_ms is
                                                     cumulative = phase shift; used unchanged)
    sws_words/items.csv, sws_words/sws_info.csv      sentence -> words -> SWS files
    sws_words/noise_pool/                            the shared noise tokens (make_noise_pool.py, built once)

Outputs in stimuli_tapping/<pid>_<group>/
    tapping_stimuli.npy       float32 (216, max_samples); row r = trial r + 1, zero-padded after n_samples
    design_stim.csv           design table + stim_row, n_samples, ...  (this is what the experiment reads)
    events_stim.csv           events table + word, file, token, ACTUAL onset/offset of every sound
                              (ms from file start) -> the reference for the tap analysis
    tapping_stim_meta.json    settings, md5 of all inputs, token seed, level summary, build time
    examples/*.wav            a few trials as wav, only for listening

Reading one trial in the experiment:
    stims = np.load('tapping/stimuli_tapping/P01_speech/tapping_stimuli.npy', mmap_mode='r')
    y = np.array(stims[row['stim_row'], :row['n_samples']])

Trial: 6 noises (lead-in) | 6 words (S1) | 6 noises (A) | the same 6 words (S2) | 6 noises (B).
Sound k's onset (the -40 dB point, same rule for words and noise) is placed at LEAD_MS + intended_onset_ms.
"""
import os
import sys
import json
import zlib
import shutil
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(PROJECT_DIR)        # relative paths below are inside the project folder
REPO_DIR = PROJECT_DIR       # folder that CONTAINS the repo package


if REPO_DIR is not None:
    sys.path.insert(0, REPO_DIR)
from repo import build_sequence, find_on_offset, load_audio                                  # CHANGED (no draw_token_order)

# %% SETTINGS
PARTICIPANT_ID = None         # e.g. 'P01'; None = ask
GROUP          = None         # 'speech' / 'acoustic'; None = ask
VALID_GROUPS   = ['speech', 'acoustic']

ITEMS_CSV    = 'sws_words/items.csv'        # from sentence_lists_to_tts.py
SWS_INFO_CSV = 'sws_words/sws_info.csv'     # from make_sws_words_trimmed.py (base -> sws_file)
DESIGN_DIR   = 'tapping/design_tables'            
NOISE_DIR    = 'sws_words/noise_pool'       # from make_noise_pool.py
OUT_ROOT     = 'tapping/stimuli_tapping'


LIST_TO_TYPE = {'items_2_4': 'S24', 'items_3_3': 'S33'}    # sentence-list file name -> sentence code
N_ITEMS      = 18                                           # first N items of each list
WL_FROM      = 'S33'          # PLACEHOLDER: WL_k uses the words of S33_k until the word list exists

N_EVENTS     = 30
N_WORDS      = 6
IOI_MS       = 450            # onset-to-onset grid
FS           = 44100

LEAD_MS      = 50             # silence before sound 1's onset
TAIL_MS      = 50             # silence after sound 30 ends
EDGE_FADE_MS = 2              # file-edge fade (falls in the silence)
ONSET_DB     = -40            # onset/offset rule: same as the SWS trimming and the noise pool
ENV_WIN_MS   = 5
MIN_GAP_MS   = 20             # smallest allowed silence between a sound's offset and the next onset
REDETECT_TOL_MS = 0.5         # onsets re-detected in the finished signal must be this close

N_EXAMPLE_WAVS = 6            # trials also written as wav (for listening); 0 = none


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


# %% 0 PARTICIPANT, TABLES, OUTPUT FOLDER
pid = (PARTICIPANT_ID or input('Enter participant ID: ')).strip().upper()          # same rule as the generators
group = (GROUP or input(f"Enter experimental group ({' / '.join(VALID_GROUPS)}): ")).strip().lower()
if group not in VALID_GROUPS:
    raise ValueError(f'group must be one of {VALID_GROUPS}, got {group!r}')

design_path = os.path.join(DESIGN_DIR, f'tapping_{pid}_{group}.csv')
events_path = os.path.join(DESIGN_DIR, f'tapping_events_{pid}_{group}.csv')
for p in (design_path, events_path):
    if not os.path.exists(p):
        raise FileNotFoundError(f'no {p} - run generate_tapping_design_table.py first')
design = pd.read_csv(design_path).sort_values('trial').reset_index(drop=True)
events = pd.read_csv(events_path).sort_values(['trial', 'event']).reset_index(drop=True)

for name, t in (('design', design), ('events', events)):
    if (t['participant'] != pid).any() or (t['group'] != group).any():
        raise ValueError(f'{name} table: participant/group inside the table do not match {pid}/{group}')
n_trials = len(design)
if len(events) != n_trials * N_EVENTS or set(events['trial']) != set(design['trial']):
    raise ValueError(f'events table does not match the design table ({len(events)} rows for {n_trials} trials)')
if not (events['event'].to_numpy().reshape(n_trials, N_EVENTS) == np.arange(1, N_EVENTS + 1)).all():
    raise ValueError('events table: every trial must have events 1-30 in order')
chk = events.groupby('trial')[['sentence_id', 'block', 'role']].first().reset_index().merge(
    design[['trial', 'sentence_id', 'block', 'role']], on='trial', suffixes=('_ev', ''))
if not ((chk.sentence_id_ev == chk.sentence_id) & (chk.block_ev == chk.block) & (chk.role_ev == chk.role)).all():
    raise ValueError('events table and design table disagree on sentence / block / role of some trials')

out_dir = os.path.join(OUT_ROOT, f'{pid}_{group}')
if os.path.exists(out_dir):
    raise FileExistsError(f'{out_dir} already exists - not overwriting. Delete or rename it to rebuild.')
tmp_dir = out_dir + '_incomplete'            # built here, renamed to out_dir only when everything passed
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)                   # leftover of an interrupted run of this script
print(f'{pid} ({group}): {n_trials} trials, {len(events)} sounds from {design_path} + {events_path}')

# %% 1 SENTENCES -> WORDS
items = pd.read_csv(ITEMS_CSV)
sws_info = pd.read_csv(SWS_INFO_CSV)
items = items.merge(sws_info[['base', 'sws_file']], on='base', how='left')
if items['sws_file'].isna().any():
    missing = sorted(items.loc[items['sws_file'].isna(), 'word'].unique())
    raise FileNotFoundError(f'no SWS file for: {missing}')

sentences = {}                                         # sentence_id -> list of 6 rows (word, base, sws_file)
for lst, code in LIST_TO_TYPE.items():
    d = items[items['list'] == lst]
    if d.empty:
        raise ValueError(f'list {lst!r} not in {ITEMS_CSV} (lists there: {sorted(items["list"].unique())})')
    for k, item in enumerate(sorted(d['item'].unique())[:N_ITEMS], start=1):
        w = d[d['item'] == item].sort_values('pos')
        if len(w) != N_WORDS:
            raise ValueError(f'{lst} item {item}: {len(w)} words, expected {N_WORDS}')
        sentences[f'{code}_{k:02d}'] = w[['word', 'base', 'sws_file']].to_dict('records')
for k in range(1, N_ITEMS + 1):                          # PLACEHOLDER word list
    sentences[f'WL_{k:02d}'] = sentences[f'{WL_FROM}_{k:02d}']
missing = sorted(set(design['sentence_id']) - set(sentences))
if missing:
    raise ValueError(f'sentences in the design table but not in the lists: {missing}')
print(f'{len(sentences)} sentences')

# %% 2 NOISE POOL (shared by all participants)
with open(os.path.join(NOISE_DIR, 'noise_pool_exact.json')) as fh:
    pool = json.load(fh)
if pool['fs'] != FS:
    raise ValueError(f'noise pool is {pool["fs"]} Hz, expected {FS}')
if pool['onset_thresh_db'] != ONSET_DB or pool['env_win_ms'] != ENV_WIN_MS:
    raise ValueError('noise pool was made with a different onset rule than ONSET_DB / ENV_WIN_MS')
GAIN = pool['shared_gain']
# every token gets pool['pad_samples'] of silence on both sides, like the 10 ms pads of the SWS words, and its
# onset/offset were measured on the padded envelope (see make_noise_pool.py, step 2)
pad = pool['pad_samples']
pool_on, pool_off = pool['onset_sample_padded'], pool['offset_sample_padded']
tokens, token_dev_ms = [], []
for f in pool['token_files']:
    y, fs = load_audio(os.path.join(NOISE_DIR, f))
    if fs != FS or len(y) != pool['n_samples']:
        raise ValueError(f'{f}: {fs} Hz / {len(y)} samples, expected {FS} Hz / {pool["n_samples"]}')
    y = np.r_[np.zeros(pad), y.astype(np.float64), np.zeros(pad)]
    tokens.append(y)
    # where THIS token's onset is when detected on its own (placement uses the pool's expected onset)
    token_dev_ms.append((find_on_offset(y, FS, ONSET_DB, ENV_WIN_MS)[0] - pool_on) / FS * 1000)
noise_sound_ms = (pool_off - pool_on) / FS * 1000
print(f'{len(tokens)} noise tokens, audible {noise_sound_ms:.0f} ms, shared gain {20 * np.log10(GAIN):+.2f} dB; '
      f'own onset vs placed onset {min(token_dev_ms):+.2f} to {max(token_dev_ms):+.2f} ms')

# %% 3 LOAD EVERY WORD ONCE (x shared gain; onset / offset with the trimming rule)
words, changed = {}, []
for path in sorted({w['sws_file'] for s in sentences.values() for w in s}):
    if pool['reference_words'].get(os.path.basename(path)) != md5(path):
        changed.append(os.path.basename(path))
        continue
    y, fs = load_audio(path)
    if fs != FS:
        raise ValueError(f'{path}: {fs} Hz, expected {FS}')
    y = y.astype(np.float64) * GAIN
    on, off = find_on_offset(y, fs, ONSET_DB, ENV_WIN_MS)
    words[path] = dict(y=y, on=on, off=off, sound_ms=(off - on) / fs * 1000,
                       rms=float(np.sqrt(np.mean(y[on:off + 1] ** 2))))
if changed:
    raise ValueError(f'{len(changed)} words are not the ones the noise pool was made from (new or changed '
                     f'since make_noise_pool.py), e.g. {changed[:5]} - rebuild the noise pool first')
dur = np.array([w['sound_ms'] for w in words.values()])
print(f'{len(words)} words, audible {dur.min():.0f}-{dur.max():.0f} ms (median {np.median(dur):.0f})')

lead_n, tail_n = int(round(LEAD_MS / 1000 * FS)), int(round(TAIL_MS / 1000 * FS))
if pool_on >= lead_n - int(round(EDGE_FADE_MS / 1000 * FS)):
    raise ValueError(f'LEAD_MS = {LEAD_MS} ms is shorter than the noise attack before its onset '
                     f'({pool_on / FS * 1000:.1f} ms): the first sound would be cut')
max_early = -min(0, events['jitter_step_ms'].min())
limit = IOI_MS - max_early - MIN_GAP_MS
too_long = [os.path.basename(p) for p, w in words.items() if w['sound_ms'] > limit]
if too_long or noise_sound_ms > limit:
    print(f'WARNING longer than {limit:.0f} ms (may fail the gap check):', ', '.join(too_long or ['noise']))

# %% 4 TOKEN ORDER (per participant, reproducible)
# Each trial draws its 18 noise bursts WITHOUT replacement: no token repeats within a trial.   # CHANGED
TOKEN_SEED = zlib.crc32(f'{pid}_tapping_tokens'.encode())
rng = np.random.default_rng(TOKEN_SEED)
n_noise = int((events.loc[events.trial == design.trial[0], 'kind'] == 'noise').sum())     # 18
if len(tokens) < n_noise:                                                                    # CHANGED
    raise ValueError(f'the noise pool has {len(tokens)} tokens, a trial needs {n_noise} different ones: '
                     f'rebuild it with N_TOKENS >= {n_noise}')
token_order = {t: [int(k) for k in rng.permutation(len(tokens))[:n_noise]]                  # CHANGED
               for t in design['trial']}                                                     # in trial order


# %% 5 BUILD ONE TRIAL
def build_one(ev, sentence_id, order):
    """ev: the 30 rows of this trial in the events table. order: its noise tokens (0-based)."""
    sent = sentences[sentence_id]
    tok = iter(order)
    seq, info = [], []
    for r in ev.itertuples(index=False):
        if r.kind == 'noise':
            k = next(tok)
            seq.append(dict(y=tokens[k], on=pool_on, off=pool_off))
            info.append(dict(word='', file=pool['token_files'][k], token=k + 1,
                             sound_ms=noise_sound_ms, det_ms=token_dev_ms[k]))
        else:
            w = sent[r.pos_in_part - 1]
            wd = words[w['sws_file']]
            seq.append(dict(y=wd['y'], on=wd['on'], off=wd['off']))
            info.append(dict(word=w['word'], file=os.path.basename(w['sws_file']), token=None,
                             sound_ms=wd['sound_ms'], det_ms=0.0))
    seq[-1] = dict(seq[-1], off=len(seq[-1]['y']) - 1)       # keep sound 30's own fade-out (no cut at -40 dB)
    shifts = ev['shift_ms'].to_numpy(float) + LEAD_MS          # cumulative (phase) shifts, unchanged + lead
    y, table = build_sequence(seq, FS, soa_ms=IOI_MS, shifts_ms=shifts, end='offset', edge_fade_ms=EDGE_FADE_MS)
    y = np.r_[y, np.zeros(tail_n, np.float32)]
    onset_ms = np.array([t['onset_ms'] for t in table])
    offset_ms = onset_ms + np.array([i['sound_ms'] for i in info])                       # audible offsets
    return y, onset_ms, offset_ms, info


# %% 6 CHECKS FOR ONE TRIAL (stop at the first failure)
def check_one(trial, ev, order, y, onset_ms, offset_ms, info):
    name = f'trial {trial}'
    tol = 1000 / FS + 1e-6                                     # one sample
    # (a) actual onset = LEAD_MS + intended onset (events table)
    want = ev['intended_onset_ms'].to_numpy(float) + LEAD_MS
    err = onset_ms - want
    assert np.all(np.abs(err) <= tol), f'{name}: onset off by up to {np.abs(err).max():.3f} ms'
    # (b) onset-to-onset intervals: 450 everywhere, 450 + step at a jitter
    assert np.all(np.abs(np.diff(onset_ms) - np.diff(want)) <= tol), f'{name}: intervals differ from the design'
    # (c) silence between sounds
    gaps = onset_ms[1:] - offset_ms[:-1]
    assert gaps.min() >= MIN_GAP_MS, f'{name}: gap of {gaps.min():.1f} ms < {MIN_GAP_MS} ms'
    # (d) onsets re-detected IN THE SIGNAL, sound by sound, with the same rule
    dev = np.zeros(N_EVENTS)
    for k in range(N_EVENTS):
        a_ms = onset_ms[k] - 30 if k == 0 else max(onset_ms[k] - 30, (offset_ms[k - 1] + onset_ms[k]) / 2)
        b_ms = offset_ms[k] + (min(15, gaps[k] / 2) if k < N_EVENTS - 1 else 15)
        a, b = int(a_ms / 1000 * FS), int(b_ms / 1000 * FS)
        on, _ = find_on_offset(y[a:b], FS, ONSET_DB, ENV_WIN_MS)
        dev[k] = (a + on) / FS * 1000 - (onset_ms[k] + info[k]['det_ms'])
    assert np.abs(dev).max() <= REDETECT_TOL_MS, \
        f'{name}: sound {np.abs(dev).argmax() + 1} re-detected {dev[np.abs(dev).argmax()]:+.2f} ms from where it was placed'
    # (e) level
    assert np.abs(y).max() < 1, f'{name}: clips'
    # (f) no noise token twice within the trial                                              # CHANGED
    assert len(set(order)) == len(order), f'{name}: a noise token repeats within the trial'
    return gaps.min(), np.abs(err).max(), np.abs(dev).max()


# %% 7 BUILD EVERY TRIAL (row r = trial r + 1)
ev_by_trial = {t: g for t, g in events.groupby('trial', sort=True)}
built, trial_rows, event_rows = [], [], []
for r, d in enumerate(design.itertuples(index=False)):
    ev, order = ev_by_trial[d.trial], token_order[d.trial]
    y, onset_ms, offset_ms, info = build_one(ev, d.sentence_id, order)
    min_gap, max_err, max_dev = check_one(d.trial, ev, order, y, onset_ms, offset_ms, info)
    built.append(y.astype(np.float32))
    trial_rows.append(dict(trial=d.trial, stim_row=r, n_samples=len(y), dur_ms=round(len(y) / FS * 1000, 2),
                           min_gap_ms=round(min_gap, 2), max_onset_error_ms=round(max_err, 4),
                           max_redetect_ms=round(max_dev, 3),
                           words=' '.join(w['word'] for w in sentences[d.sentence_id]),
                           tokens=' '.join(str(k + 1) for k in order)))
    for k, (e, i) in enumerate(zip(ev.itertuples(index=False), info)):
        event_rows.append(dict(trial=d.trial, event=e.event, word=i['word'], file=i['file'], token=i['token'],
                               intended_onset_in_file_ms=round(e.intended_onset_ms + LEAD_MS, 3),
                               actual_onset_sample=int(round(onset_ms[k] / 1000 * FS)),
                               actual_onset_ms=round(onset_ms[k], 3),
                               actual_offset_ms=round(offset_ms[k], 3),
                               onset_error_ms=round(onset_ms[k] - e.intended_onset_ms - LEAD_MS, 4)))
    if (r + 1) % 25 == 0:
        print(f'{r + 1}/{n_trials}')
stim = pd.DataFrame(trial_rows)
ev_stim = events.merge(pd.DataFrame(event_rows), on=['trial', 'event'], validate='one_to_one')
ev_stim['token'] = ev_stim['token'].astype('Int64')
assert len(ev_stim) == len(events)
# S1 and S2 of a trial are the same words
w = ev_stim[ev_stim.kind == 'word']
assert (w[w.part == 'S1'].file.to_numpy() == w[w.part == 'S2'].file.to_numpy()).all(), 'S1 and S2 differ'

# %% 8 SAVE (to the temporary folder, then rename)
os.makedirs(tmp_dir)
max_len = max(map(len, built))
store = np.lib.format.open_memmap(os.path.join(tmp_dir, 'tapping_stimuli.npy'), mode='w+',
                                  dtype=np.float32, shape=(len(built), max_len))
for r, y in enumerate(built):
    store[r, :len(y)] = y                                   # the rest stays 0
store.flush()

design_stim = design.merge(stim, on='trial', validate='one_to_one')
assert (design_stim['stim_row'].values == np.arange(len(design_stim))).all()
design_stim.to_csv(os.path.join(tmp_dir, 'design_stim.csv'), index=False)
ev_stim.to_csv(os.path.join(tmp_dir, 'events_stim.csv'), index=False)

word_rms_db = 20 * np.log10([w['rms'] for w in words.values()])
noise_rms_db = 20 * np.log10(np.mean([np.sqrt(np.mean(t[pool_on:pool_off + 1] ** 2)) for t in tokens]))
used_files = sorted({w['sws_file'] for s in sentences.values() for w in s})
meta = dict(participant=pid, group=group, built=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            design_table=os.path.abspath(design_path), design_table_md5=md5(design_path),
            events_table=os.path.abspath(events_path), events_table_md5=md5(events_path),
            noise_pool=os.path.abspath(NOISE_DIR), noise_pool_md5=md5(os.path.join(NOISE_DIR, 'noise_pool_exact.json')),
            shared_gain=GAIN, token_seed=TOKEN_SEED, token_onset_vs_placed_ms=[round(v, 3) for v in token_dev_ms],
            fs=FS, ioi_ms=IOI_MS, n_events=N_EVENTS, lead_ms=LEAD_MS, tail_ms=TAIL_MS, edge_fade_ms=EDGE_FADE_MS,
            onset_db=ONSET_DB, env_win_ms=ENV_WIN_MS, min_gap_ms=MIN_GAP_MS, redetect_tol_ms=REDETECT_TOL_MS,
            n_items=N_ITEMS, list_to_type=LIST_TO_TYPE, wl_placeholder_from=WL_FROM,
            n_trials=len(stim), max_samples=int(max_len), dtype='float32',
            levels=dict(word_rms_db_min=round(float(word_rms_db.min()), 2), word_rms_db_max=round(float(word_rms_db.max()), 2),
                        word_rms_db_sd=round(float(word_rms_db.std()), 2), noise_rms_db=round(float(noise_rms_db), 2),
                        noise_minus_mean_word_db=round(float(noise_rms_db - word_rms_db.mean()), 2)),
            words_files={os.path.basename(p): md5(p) for p in used_files})
with open(os.path.join(tmp_dir, 'tapping_stim_meta.json'), 'w') as fh:
    json.dump(meta, fh, indent=2)

if N_EXAMPLE_WAVS:
    ex_dir = os.path.join(tmp_dir, 'examples')
    os.makedirs(ex_dir)
    pick = design_stim.groupby('jitter_pattern', sort=False).head(1).head(N_EXAMPLE_WAVS)   # one per pattern
    for r in pick.itertuples(index=False):
        sf.write(os.path.join(ex_dir, f'trial{r.trial:03d}_{r.sentence_id}_{r.jitter_pattern.replace("+", "")}.wav'),
                 store[r.stim_row, :r.n_samples], FS)

del store
os.rename(tmp_dir, out_dir)
size_mb = os.path.getsize(os.path.join(out_dir, 'tapping_stimuli.npy')) / 1e6
print(f'\n✓ {len(stim)} trials -> {out_dir}/  ({size_mb:.0f} MB)')
print(f'  duration {stim.dur_ms.min():.0f}-{stim.dur_ms.max():.0f} ms; smallest gap {stim.min_gap_ms.min():.1f} ms '
      f'(trial {stim.loc[stim.min_gap_ms.idxmin(), "trial"]})')
print(f'  actual vs intended onsets: max {stim.max_onset_error_ms.max():.4f} ms; '
      f're-detected in the signal: max {stim.max_redetect_ms.max():.3f} ms')
print(f'  levels: words {word_rms_db.min():.1f} to {word_rms_db.max():.1f} dB RMS (SD {word_rms_db.std():.2f}), '
      f'noise {noise_rms_db:.1f} dB ({noise_rms_db - word_rms_db.mean():+.2f} dB vs mean word)')