# %% CHECK THE TAPPING STIMULI BY EYE
"""
Plots the tapping stimuli of ONE participant (stimuli_tapping/<pid>_<group>/) and checks every shift.

The onsets of all 30 sounds are measured AGAIN from the signal itself (NOT copied from events_stim.csv),
without assuming where the sounds should be, with the same rule as everywhere else
(-40 dB re the sound's own peak, 5 ms RMS envelope). Then:

    measured shift of sound k  =  measured onset  -  regular grid position (LEAD + (k-1) * IOI)
                                  -  token offset (noise only, see below)
    expected shift of sound k  =  shift_ms in the events table (cumulative = phase shift)

Token offset: a noise burst is placed by the pool's AVERAGE envelope, so each token's own -40 dB point sits a
fraction of a ms from where it was placed (token_onset_vs_placed_ms in the meta file, measured once per token).
Words have no such offset.

Figure 1 - selected trials, two panels each:
    top    waveform; shaded = the 5 parts (lead-in, S1, A, S2, B); dashed = regular 450 ms grid;
           dots = measured onsets (green = as expected, red = wrong); arrows = jittered sounds; words written above
    bottom shift of every sound: line = expected (steps at the jitters, then stays: phase shift), dots = measured
Figure 2 - ALL trials: expected vs measured shift of every sound (must lie on the diagonal) + error histogram
Figure 3 - where the jitters are: measured shift of every sound of every trial (rows = trials sorted by role
           and pattern), and how many jitters fall on each of the 30 sounds
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import os                                                        # (if not already imported)
PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(PROJECT_DIR)

# %% SETTINGS
PARTICIPANT_ID = 'P01'
GROUP          = 'speech'
STIM_DIR = f'tapping/stimuli_tapping/{PARTICIPANT_ID}_{GROUP}'
PICK       = 'per_pattern'  # 'per_pattern' = one random trial of every jitter pattern (+ N_EXTRA random);
                            # 'random' = N_RANDOM random trials
N_EXTRA    = 1
N_RANDOM   = 6
SEED       = None           # None = new random pick each run; a number = same pick
TRIALS     = None           # or a list of trial numbers, e.g. [21, 51] (overrides PICK)
TOL_MS     = 0.5            # allowed difference between measured and expected shift
ONSET_DB, ENV_WIN_MS = -40, 5
SEG_DB     = -30            # a "sound" = envelope above (overall peak + SEG_DB); then refined with ONSET_DB
MAX_SOUND_SPAN_MS = 600     # pieces of ONE sound span less than this; two neighbouring sounds span more
                            # (sounds ~350 ms, onsets >= IOI - 50 = 400 ms apart -> two sounds >= ~750 ms)

PART_COLOR = {'lead-in': '0.85', 'S1': 'tab:blue', 'A': '0.85', 'S2': 'tab:orange', 'B': '0.85'}

# %% LOAD
meta   = json.load(open(f'{STIM_DIR}/tapping_stim_meta.json'))
FS, IOI, LEAD, NE = meta['fs'], meta['ioi_ms'], meta['lead_ms'], meta['n_events']
TOKEN_DEV = np.array(meta['token_onset_vs_placed_ms'])                 # per token (1-based token k -> [k-1])
stims  = np.load(f'{STIM_DIR}/tapping_stimuli.npy', mmap_mode='r')
design = pd.read_csv(f'{STIM_DIR}/design_stim.csv').set_index('trial', drop=False)   # one row per trial
design.index.name = None                                                              # index = trial number
events = pd.read_csv(f'{STIM_DIR}/events_stim.csv')                                   # one row per sound
ev_by_trial = {t: g.sort_values('event') for t, g in events.groupby('trial')}
grid_ms = LEAD + np.arange(NE) * IOI


def get(trial):
    r = design.loc[trial]
    return np.array(stims[r['stim_row'], :r['n_samples']], dtype=float)


def envelope(y, win_ms=ENV_WIN_MS):
    """Moving-RMS envelope, identical to np.convolve(y**2, ones(n)/n, 'same') but fast (cumulative sum)."""
    n = max(1, int(round(win_ms / 1000 * FS)))
    c = np.r_[0.0, np.cumsum(y ** 2)]
    i = np.arange(len(y))
    lo = np.clip(i - n // 2, 0, len(y))
    hi = np.clip(i + (n - 1) // 2 + 1, 0, len(y))
    return np.sqrt(np.maximum(c[hi] - c[lo], 0) / n)


def measure_onsets(y, trial=''):
    """Onset (ms) of each of the NE sounds, found WITHOUT assuming where the sounds should be.

    1 find the separate sounds: envelope above (overall peak + SEG_DB).
    2 a word with a silent stretch inside (voicing mask) gives 2+ pieces: repeatedly merge the neighbouring pair
      with the SHORTEST combined span until NE sounds remain. A merge spanning more than MAX_SOUND_SPAN_MS would
      join two different sounds -> error.
    3 refine each onset with the trimming rule (peak of THIS sound + ONSET_DB): walk back from the start of the
      sound until the envelope falls below that threshold.
    """
    env = envelope(y)
    above = env > env.max() * 10 ** (SEG_DB / 20)
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
    starts, ends = list(edges[::2]), list(edges[1::2])
    while len(starts) > NE:
        spans = [ends[k + 1] - starts[k] for k in range(len(starts) - 1)]
        k = int(np.argmin(spans))
        if spans[k] / FS * 1000 > MAX_SOUND_SPAN_MS:
            raise ValueError(f'trial {trial}: {len(starts)} sounds and no two of them fit in one sound')
        ends[k] = ends.pop(k + 1); starts.pop(k + 1)
    if len(starts) != NE:
        raise ValueError(f'trial {trial}: found {len(starts)} separate sounds, expected {NE}')
    out = []
    for s, e in zip(starts, ends):
        thr = env[s:e].max() * 10 ** (ONSET_DB / 20)
        i = s
        while i > 0 and env[i - 1] > thr:
            i -= 1
        out.append(i / FS * 1000)
    return np.array(out)


def measured_shift(trial):
    """(measured shift, expected shift, measured onsets) of the NE sounds of a trial, in ms."""
    ev = ev_by_trial[trial]
    on = measure_onsets(get(trial), trial)
    tok = ev['token'].to_numpy(float)
    dev = np.where(ev['kind'].to_numpy() == 'noise', TOKEN_DEV[np.nan_to_num(tok, nan=1).astype(int) - 1], 0.0)
    return on - grid_ms - dev, ev['shift_ms'].to_numpy(float), on


def describe(row):
    """'S2 I3 -15 (sound 22) + A I2 +30 (sound 15)' or 'no jitter'."""
    parts = []
    for p in ('A', 'S2', 'B'):
        if row[f'{p}_jitter']:
            parts.append(f"{p} {row[f'{p}_interval']} {row[f'{p}_shift_ms']:+.0f} (sound {int(row[f'{p}_event'])})")
    return ' + '.join(parts) if parts else 'no jitter'


# %% FIGURE 1 - SELECTED TRIALS
rng = np.random.default_rng(SEED)
if TRIALS is not None:
    pick = list(TRIALS)
elif PICK == 'per_pattern':
    pick = [int(rng.choice(g['trial'].to_numpy())) for _, g in design.groupby('jitter_pattern')]
    rest = design.loc[~design['trial'].isin(pick), 'trial'].to_numpy()
    pick += [int(t) for t in rng.choice(rest, N_EXTRA, replace=False)]
else:
    pick = [int(t) for t in rng.choice(design['trial'].to_numpy(), N_RANDOM, replace=False)]

fig, axes = plt.subplots(2 * len(pick), 1, figsize=(15, 3.6 * len(pick)), squeeze=False,
                         gridspec_kw=dict(height_ratios=[1.4, 1] * len(pick)))
for j, trial in enumerate(pick):
    row, ev = design.loc[trial], ev_by_trial[trial]
    y = get(trial)
    meas, want, on = measured_shift(trial)
    ok = np.abs(meas - want) <= TOL_MS
    t = np.arange(len(y)) / FS * 1000
    jit = ev['jitter_here'].to_numpy()

    # --- top: waveform
    ax = axes[2 * j, 0]
    for part, g in ev.groupby('part', sort=False):
        k0, k1 = g['event'].min() - 1, g['event'].max() - 1
        ax.axvspan(on[k0] - 20, on[k1] + 380, color=PART_COLOR[part], alpha=0.25 if part in ('S1', 'S2') else 0.5,
                   lw=0)
        ax.text((on[k0] + on[k1] + 350) / 2, 1.0, part, ha='center', va='bottom', fontsize=9, fontweight='bold',
                transform=ax.get_xaxis_transform())
    ax.plot(t, y, color='k', lw=0.3)
    peak = np.abs(y).max()
    for k in range(NE):
        ax.axvline(grid_ms[k], color='0.4', ls='--', lw=0.6)
        ax.plot(on[k], peak * 1.05, 'o', ms=3.5, color='tab:green' if ok[k] else 'tab:red', zorder=5)
        if jit[k]:
            ax.annotate(f"{ev['jitter_step_ms'].iloc[k]:+.0f} ms", xy=(on[k], peak * 1.05),
                        xytext=(on[k], peak * 1.6), ha='center', fontsize=8, color='tab:red', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='tab:red', lw=1))
        if ev['kind'].iloc[k] == 'word':
            ax.text(on[k] + 175, -peak * 1.35, ev['word'].iloc[k], ha='center', fontsize=7.5, color='0.25')
    ax.set(xlim=(0, t.max() + 20), ylim=(-peak * 1.6, peak * 1.9), yticks=[])
    status = 'OK' if ok.all() else 'WRONG'
    ax.set_title(f"trial {trial} | block {row['block']} | {row['sentence_id']} ({row['sentence_type']}) | "
                 f"{row['role']} | {row['S2_boundary_status']} | {describe(row)}  ->  {status}",
                 loc='left', fontsize=9, color='tab:green' if ok.all() else 'tab:red', pad=20)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)

    # --- bottom: shift of every sound
    ax = axes[2 * j + 1, 0]
    x = np.arange(1, NE + 1)
    for part, g in ev.groupby('part', sort=False):
        ax.axvspan(g['event'].min() - 0.5, g['event'].max() + 0.5, color=PART_COLOR[part],
                   alpha=0.25 if part in ('S1', 'S2') else 0.5, lw=0)
    ax.step(x, want, where='mid', color='k', lw=1.2, label='expected (events table)')
    ax.scatter(x, meas, s=18, zorder=5, c=np.where(ok, 'tab:green', 'tab:red'), label='measured in the signal')
    for k in np.flatnonzero(jit):
        ax.axvline(k + 1 - 0.5, color='tab:red', lw=0.8, ls=':')
    ax.axhline(0, color='0.6', lw=0.6)
    lim = max(10, np.abs(want).max() + 10)
    ax.set(xlim=(0.5, NE + 0.5), ylim=(-lim, lim), xticks=x, ylabel='shift (ms)')
    ax.tick_params(axis='x', labelsize=7)
    if j == 0:
        ax.legend(fontsize=8, loc='upper left', frameon=False, ncol=2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[-1, 0].set_xlabel('sound (1-30)   [dotted red = jittered sound: it and every later sound move]')
plt.tight_layout()
plt.show()

# %% FIGURE 2 - ALL TRIALS
rows, M = [], np.zeros((len(design), NE))
for i, trial in enumerate(design['trial']):
    meas, want, _ = measured_shift(trial)
    M[i] = meas
    ev = ev_by_trial[trial]
    for k in range(NE):
        rows.append(dict(trial=trial, event=k + 1, part=ev['part'].iloc[k], kind=ev['kind'].iloc[k],
                         expected=want[k], measured=meas[k]))
    if (i + 1) % 50 == 0:
        print(f'measured {i + 1}/{len(design)} trials')
chk = pd.DataFrame(rows)
chk['error'] = chk['measured'] - chk['expected']
bad = chk[chk['error'].abs() > TOL_MS]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
lim = chk['expected'].abs().max() + 10
ax.plot([-lim, lim], [-lim, lim], color='0.7', lw=1, zorder=0)
for kind, c in (('noise', 'tab:gray'), ('word', 'tab:blue')):
    d = chk[chk['kind'] == kind]
    ax.scatter(d['expected'], d['measured'], s=10, alpha=0.4, color=c, label=kind)
if len(bad):
    ax.scatter(bad['expected'], bad['measured'], s=16, color='tab:red', label='wrong')
ax.set(xlabel='expected shift of sound (ms)', ylabel='measured shift (ms)',
       title=f'All {len(design)} trials x {NE} sounds', xlim=(-lim, lim), ylim=(-lim, lim))
ax.legend(fontsize=8, frameon=False)
ax = axes[1]
ax.hist(chk['error'], bins=40, color='tab:blue')
ax.axvspan(-TOL_MS, TOL_MS, color='tab:green', alpha=0.15)
ax.set(xlabel='measured - expected (ms)', ylabel='sounds', title=f'Error (green band = +-{TOL_MS} ms)')
plt.tight_layout()
plt.show()

# %% FIGURE 3 - WHERE THE JITTERS ARE
order = design.sort_values(['role', 'jitter_pattern', 'S2_interval', 'sentence_type', 'trial']).index
rpos = {t: i for i, t in enumerate(design['trial'])}
M_sorted = M[[rpos[t] for t in order]]
fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax = axes[0]
v = max(1.0, np.abs(M_sorted).max())
im = ax.imshow(M_sorted, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(0, -v, v), interpolation='nearest',
               extent=(0.5, NE + 0.5, len(order), 0))
for b in (6.5, 12.5, 18.5, 24.5):
    ax.axvline(b, color='k', lw=0.8)
for k, name in zip((3.5, 9.5, 15.5, 21.5, 27.5), ('lead-in', 'S1', 'A', 'S2', 'B')):
    ax.text(k, -0.5, name, ha='center', va='bottom', fontsize=9, fontweight='bold')
roles = design.loc[order, 'role'].to_numpy()
for i in np.flatnonzero(roles[1:] != roles[:-1]) + 1:
    ax.axhline(i, color='k', lw=0.8)
for r in pd.unique(roles):
    idx = np.flatnonzero(roles == r)
    ax.text(NE + 1, idx.mean(), r, va='center', fontsize=8)
ax.set(xlabel='sound', ylabel='trials (sorted by role, pattern, S2 interval)', xticks=range(1, NE + 1))
ax.tick_params(axis='x', labelsize=7)
plt.colorbar(im, ax=ax, label='measured shift (ms): red = later, blue = earlier', fraction=0.03, pad=0.12)
ax.set_title('Measured shift of every sound (a jitter = colour change along a row)', loc='left', fontsize=10, pad=20)

ax = axes[1]
jp = events[events['jitter_here']]
cnt = jp.groupby(['event', 'jitter_step_ms']).size().unstack(fill_value=0).reindex(range(1, NE + 1), fill_value=0)
bottom = np.zeros(NE)
cmap = plt.get_cmap('RdBu_r')
steps = sorted(cnt.columns)
for s in steps:
    ax.bar(range(1, NE + 1), cnt[s], bottom=bottom, width=0.8, label=f'{s:+.0f} ms',
           color=cmap(0.5 + 0.5 * s / max(abs(x) for x in steps)))
    bottom += cnt[s].to_numpy()
for b in (6.5, 12.5, 18.5, 24.5):
    ax.axvline(b, color='k', lw=0.8)
ax.set(xlabel='sound', ylabel='jitters (all trials)', xticks=range(1, NE + 1), xlim=(0.5, NE + 0.5),
       title='Where the jitters fall (from the table)')
ax.tick_params(axis='x', labelsize=7)
ax.legend(fontsize=7, frameon=False, ncol=2)
plt.tight_layout()
plt.show()

print(f'{len(design)} trials, {len(chk)} sound onsets: max |error| = {chk["error"].abs().max():.3f} ms '
      f'(words {chk.loc[chk.kind == "word", "error"].abs().max():.3f}, noise {chk.loc[chk.kind == "noise", "error"].abs().max():.3f})')
if len(bad):
    print(f'✗ {bad["trial"].nunique()} trials with a wrong shift:')
    print(bad.to_string(index=False))
else:
    print('✓ every sound of every trial is where it should be')