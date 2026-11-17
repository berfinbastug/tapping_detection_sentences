# %% CHECK THE DETECTION STIMULI BY EYE
"""
Plots stimuli of ONE participant (stimuli/<pid>_<group>/) for randomly chosen trials and checks the shifts.

Word onsets are measured AGAIN from the signal itself (not copied from the index), each word in
its own window, with the same rule as the trimming (-40 dB re the word's peak, 5 ms RMS envelope).
Then:  measured shift of word k  =  measured onset  -  regular grid position (LEAD + (k-1) * IOI)

Figure 1 - one panel per trial:
    grey   = the unshifted version of the same sentence, if this participant has it (catch trial)
    colour = the trial's stimulus; shaded words = the words that should move
    dashed = the regular 450 ms grid;  dots = measured onsets
    under each word: measured shift in ms (green = as expected, red = wrong)
Figure 2 - ALL trials of the table: expected vs measured shift of every word (must lie on the diagonal).
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% SETTINGS
PARTICIPANT_ID = 'P02'
GROUP          = 'acoustic'
STIM_DIR   = f'stimuli/{PARTICIPANT_ID}_{GROUP}'
N_RANDOM   = 6            # trials to plot
SEED       = None         # None = new random pick each run; a number = same pick
STIM_IDS   = None         # or a list, e.g. ['S24_01_I2_p50', 'S33_04_I3_m15'] (overrides N_RANDOM)
TOL_MS     = 0.5          # allowed difference between measured and expected shift
ONSET_DB, ENV_WIN_MS = -40, 5

# %% LOAD
meta  = json.load(open(f'{STIM_DIR}/detection_stim_meta.json'))
FS, IOI, LEAD, NW = meta['fs'], meta['ioi_ms'], meta['lead_ms'], meta['n_words']
stims = np.load(f'{STIM_DIR}/detection_stimuli.npy', mmap_mode='r')
design = pd.read_csv(f'{STIM_DIR}/detection_stim_index.csv')      # one row per trial
index = design.drop_duplicates('stim_id').set_index('stim_id')


def get(stim_id):
    r = index.loc[stim_id]
    return np.array(stims[r['stim_row'], :r['n_samples']], dtype=float)


def env_db(y, win_ms=ENV_WIN_MS):
    n = max(1, int(round(win_ms / 1000 * FS)))
    return np.sqrt(np.convolve(y ** 2, np.ones(n) / n, mode='same'))


MAX_WORD_SPAN_MS = 600   # pieces of ONE word span less than this; two neighbouring words span more
                         # (words ~350 ms, onsets >= IOI - 50 = 400 ms apart -> two words >= ~750 ms)


def measure_onsets(y, seg_db=-30, stim=''):
    """Onset (ms) of each word, found WITHOUT assuming where the words should be.

    1 find the separate sounds: envelope above (overall peak + seg_db).
    2 a word with a silent stretch inside (voicing mask) gives 2+ pieces: repeatedly merge the
      neighbouring pair with the SHORTEST combined span until NW sounds remain. A merge spanning
      more than MAX_WORD_SPAN_MS would join two different words -> error.
    3 refine each onset with the trimming rule (peak of THIS word + ONSET_DB): walk back from the
      start of the sound until the envelope falls below that threshold.
    """
    env = env_db(y)
    above = env > env.max() * 10 ** (seg_db / 20)
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
    starts, ends = list(edges[::2]), list(edges[1::2])
    while len(starts) > NW:
        spans = [ends[k + 1] - starts[k] for k in range(len(starts) - 1)]
        k = int(np.argmin(spans))
        if spans[k] / FS * 1000 > MAX_WORD_SPAN_MS:
            raise ValueError(f'{stim}: {len(starts)} sounds and no two of them fit in one word')
        ends[k] = ends.pop(k + 1); starts.pop(k + 1)
    if len(starts) != NW:
        raise ValueError(f'{stim}: found {len(starts)} separate sounds, expected {NW}')
    out = []
    for s, e in zip(starts, ends):
        thr = env[s:e].max() * 10 ** (ONSET_DB / 20)
        i = s
        while i > 0 and env[i - 1] > thr:
            i -= 1
        out.append(i / FS * 1000)
    return np.array(out)


def expected_shifts(interval, shift_ms):
    v = np.zeros(NW)
    if isinstance(interval, str):
        v[int(interval[1:]):] = shift_ms
    return v


grid_ms = LEAD + np.arange(NW) * IOI

# %% FIGURE 1 - SELECTED TRIALS
if STIM_IDS is not None:
    pick = design[design['stim_id'].isin(STIM_IDS)].drop_duplicates('stim_id')
else:
    pick = design.sample(N_RANDOM, random_state=SEED)

fig, axes = plt.subplots(len(pick), 1, figsize=(13, 2.3 * len(pick)), squeeze=False)
for ax, (_, row) in zip(axes[:, 0], pick.iterrows()):
    y   = get(row['stim_id'])
    ref_id = f"{row['sentence_id']}_none"
    ref = get(ref_id) if ref_id in index.index else np.zeros(1)
    t, t_ref = np.arange(len(y)) / FS * 1000, np.arange(len(ref)) / FS * 1000
    on = measure_onsets(y, stim=row['stim_id'])
    meas, want = on - grid_ms, expected_shifts(row['interval'], row['shift_ms'])
    ok = np.abs(meas - want) <= TOL_MS

    ax.plot(t_ref, ref - 1.1, color='0.65', lw=0.4)                     # reference, below
    ax.plot(t, y + 1.1, color='tab:blue', lw=0.4)                        # stimulus, above
    for k in range(NW):
        if want[k] != 0:
            ax.axvspan(on[k], index.loc[row['stim_id'], f'off{k + 1}_ms'], ymin=0.5, ymax=1,
                       color='tab:orange', alpha=0.15, lw=0)
        ax.axvline(grid_ms[k], color='k', ls='--', lw=0.7)
        ax.plot(on[k], 1.1, 'o', ms=4, color='tab:red' if not ok[k] else 'tab:green', zorder=5)
        ax.text(grid_ms[k] + 175, -2.45, f'{meas[k]:+.1f}', ha='center', fontsize=8,
                color='tab:green' if ok[k] else 'tab:red', fontweight='bold')
    words = index.loc[row['stim_id'], 'words'].split()
    for k, w in enumerate(words):
        ax.text(grid_ms[k] + 175, 2.25, w, ha='center', fontsize=8, color='0.3')
    status = 'OK' if ok.all() else 'WRONG'
    ax.set_title(f"trial {row['trial']} | {row['stim_id']} | {row['role']} | expected: "
                 + (f"{row['shift_ms']:+.0f} ms from word {int(row['interval'][1:]) + 1} on"
                    if isinstance(row['interval'], str) else 'no shift')
                 + f"  ->  {status}", loc='left', fontsize=9,
                 color='tab:green' if ok.all() else 'tab:red')
    ax.set(ylim=(-2.7, 2.6), xlim=(0, t.max() + 20), yticks=[-1.1, 1.1])
    ax.set_yticklabels(['unshifted', 'stimulus'], fontsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axes[-1, 0].set_xlabel('Time (ms)   [dashed = regular grid, numbers = measured shift of each word in ms]')
plt.tight_layout()
plt.show()

# %% FIGURE 2 - ALL TRIALS OF THE TABLE
rows = []
for _, row in design.iterrows():
    meas = measure_onsets(get(row['stim_id']), stim=row['stim_id']) - grid_ms
    want = expected_shifts(row['interval'], row['shift_ms'])
    for k in range(NW):
        rows.append(dict(trial=row['trial'], stim_id=row['stim_id'], word=k + 1, expected=want[k], measured=meas[k]))
chk = pd.DataFrame(rows)
chk['error'] = chk['measured'] - chk['expected']
bad = chk[chk['error'].abs() > TOL_MS]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
lim = max(abs(v) for v in meta['shifts_ms']) + 10
ax.plot([-lim, lim], [-lim, lim], color='0.7', lw=1, zorder=0)
ax.scatter(chk['expected'], chk['measured'], s=12, alpha=0.4,
           c=np.where(chk['error'].abs() > TOL_MS, 'tab:red', 'tab:blue'))
ax.set(xlabel='expected shift of word (ms)', ylabel='measured shift (ms)',
       title=f'All {len(design)} trials x {NW} words', xlim=(-lim, lim), ylim=(-lim, lim))
ax = axes[1]
ax.hist(chk['error'], bins=40, color='tab:blue')
ax.axvspan(-TOL_MS, TOL_MS, color='tab:green', alpha=0.15)
ax.set(xlabel='measured - expected (ms)', ylabel='words', title=f'Error (green band = +-{TOL_MS} ms)')
plt.tight_layout()
plt.show()

print(f'{len(design)} trials, {len(chk)} word onsets: max |error| = {chk["error"].abs().max():.3f} ms')
if len(bad):
    print(f'✗ {bad["stim_id"].nunique()} stimuli with a wrong shift:')
    print(bad.to_string(index=False))
else:
    print('✓ every word of every trial is where it should be')