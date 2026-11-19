"""
check_04_rise_by_position.py
Step 4: do onset acoustics differ systematically at phrase boundaries?

Needs checks/03_word_acoustics.csv (run check_03_word_acoustics.py first).

Logic. Words sit on a fixed 450-ms file-onset grid. If word k+1 reaches its perceptual
onset later than word k, the interval between them is effectively LONGER, even though
the physical IOI is identical. A systematically longer effective interval at the
boundary would mimic a boundary pause = the confound.

  eff_shift(I_k) = t50(word k+1) - t50(word k)        [ms; >0 = interval feels longer]
  (also for t10 and t90)

Boundary intervals: 2|4 -> I2 (before word 3), 3|3 -> I3 (before word 4).
Key contrasts at the SAME serial position:
  A. I2 / word 3 : 2|4 (boundary, verb)  vs 3|3 (within, noun)   [vs WL]
  B. I3 / word 4 : 3|3 (boundary, verb)  vs 2|4 (within, Det)    [vs WL]
  C. within each type: boundary interval vs the other within-phrase intervals

Outputs (OUT_DIR): 04_word_level.csv, 04_interval_level.csv, 04_item_flags.csv,
                   04_by_position.png, printed summary.
Run (conda env `psychopy`):  python check_04_rise_by_position.py
"""
import os

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ CONFIG
BASE_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(BASE_DIR)
OUT_DIR   = os.path.join(BASE_DIR, 'checks')
ACOUSTICS = os.path.join(OUT_DIR, '03_word_acoustics.csv')
WORDS_CSV = os.path.join(BASE_DIR, 'sws_words', 'words.csv')
CSV_DIR =os.path.join(BASE_DIR, 'sentence_lists')

LISTS     = {'2|4': 'items_2_4.csv', '3|3': 'items_3_3.csv', 'WL': 'items_wl.csv'}
BOUNDARY  = {'2|4': 2, '3|3': 3}           # interval index of the phrase boundary
MEASURES  = ['t10_ms', 't50_ms', 't90_ms']  # onset landmarks; t50 is the main one
MAIN      = 't50_ms'
FLAG_MS   = 15                              # flag boundary intervals with |eff_shift - type's within mean| > FLAG_MS
# -------------------------------------------------------------------------

try:
    from scipy import stats
except ImportError:
    stats = None

_W2B = {}
if os.path.exists(WORDS_CSV):
    _w = pd.read_csv(WORDS_CSV, dtype=str)
    _W2B = dict(zip(_w['word'].str.strip(), _w['base'].str.strip()))


def base_name(word):
    w = word.strip()
    if w in _W2B:
        return _W2B[w]
    w = w.lower()
    for a, b in (('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')):
        w = w.replace(a, b)
    return w


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(n_a=len(a), n_b=len(b), mean_a=np.nan, mean_b=np.nan, diff=np.nan, d=np.nan, t=np.nan, p=np.nan)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    t, p = (stats.ttest_ind(a, b, equal_var=False) if stats else (np.nan, np.nan))
    return dict(n_a=len(a), n_b=len(b), mean_a=a.mean(), mean_b=b.mean(), diff=a.mean() - b.mean(), d=d, t=t, p=p)


def fmt(r):
    return (f'{r["mean_a"]:7.1f} vs {r["mean_b"]:7.1f}  diff {r["diff"]:+6.1f} ms  '
            f'd = {r["d"]:+.2f}  p = {r["p"]:.3f}  (n {r["n_a"]}/{r["n_b"]})')


# ---------------------------------------------------------------- load
ac = pd.read_csv(ACOUSTICS)
ac = ac[ac.kind == 'word'].set_index('base')

word_rows = []
for t, fn in LISTS.items():
    path = os.path.join(CSV_DIR, fn)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, dtype=str).fillna('')
    slot_cols = [c for c in df.columns if c.strip().lower().startswith('slot')]
    for _, r in df.iterrows():
        for pos, c in enumerate(slot_cols, start=1):
            w = r[c].strip()
            b = base_name(w)
            row = dict(type=t, item=r['item'], pos=pos, slot=c, word_class=c.split('_', 1)[-1], word=w, base=b)
            if b in ac.index:
                for m in MEASURES + ['rise_10_90_ms', 'rms_db', 'onset_class', 'stretch']:
                    if m in ac.columns:
                        row[m] = ac.at[b, m]
            else:
                row['missing_audio'] = True
            word_rows.append(row)
wl = pd.DataFrame(word_rows)
if 'missing_audio' in wl and wl['missing_audio'].fillna(False).any():
    print('[!] no acoustics for:', sorted(wl.loc[wl.missing_audio.fillna(False), 'word'].unique()))
wl.to_csv(os.path.join(OUT_DIR, '04_word_level.csv'), index=False, float_format='%.2f')
types = list(wl.type.unique())

# interval level: I_k between word k and word k+1
iv_rows = []
for (t, item), g in wl.groupby(['type', 'item'], sort=False):
    g = g.sort_values('pos').reset_index(drop=True)
    for k in range(1, len(g)):
        row = dict(type=t, item=item, interval=k, junction=f'{g.word[k-1]} | {g.word[k]}',
                   junction_class=f'{g.word_class[k-1]}|{g.word_class[k]}',
                   boundary=(BOUNDARY.get(t) == k))
        for m in MEASURES:
            row[f'shift_{m}'] = g.loc[k, m] - g.loc[k - 1, m]
        row['next_onset_class'] = g.loc[k, 'onset_class'] if 'onset_class' in g else ''
        iv_rows.append(row)
iv = pd.DataFrame(iv_rows)
iv.to_csv(os.path.join(OUT_DIR, '04_interval_level.csv'), index=False, float_format='%.2f')

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 20)


def hr(s):
    print('\n' + '=' * 78 + f'\n{s}\n' + '=' * 78)


# ---------------------------------------------------------------- 1. word onsets by position
hr(f'1. WORD ONSET LANDMARKS BY TYPE x POSITION (mean ms from file start; SD in brackets)')
for m in MEASURES:
    tab = wl.pivot_table(index='type', columns='pos', values=m, aggfunc='mean').round(1)
    sd = wl.pivot_table(index='type', columns='pos', values=m, aggfunc='std').round(1)
    print(f'\n{m}:')
    print(tab.astype(str) + ' (' + sd.astype(str) + ')')
print('\nOnset class composition per position (counts):')
print(pd.crosstab([wl.type, wl.pos], wl.onset_class).to_string())

# ---------------------------------------------------------------- 2. effective interval shift
hr(f'2. EFFECTIVE INTERVAL SHIFT  t(word k+1) - t(word k)   [>0 = interval feels longer]')
for m in MEASURES:
    col = f'shift_{m}'
    tab = iv.pivot_table(index='type', columns='interval', values=col, aggfunc='mean').round(1)
    sd = iv.pivot_table(index='type', columns='interval', values=col, aggfunc='std').round(1)
    print(f'\n{col}  (boundary: 2|4 = I2, 3|3 = I3)')
    print(tab.astype(str) + ' (' + sd.astype(str) + ')')

# ---------------------------------------------------------------- 3. key contrasts
hr(f'3. KEY CONTRASTS (main measure: {MAIN}; Welch t, Cohen d)')
col = f'shift_{MAIN}'
res = []


def contrast(label, a, b):
    r = welch(a, b)
    r['contrast'] = label
    res.append(r)
    print(f'{label:58s} {fmt(r)}')


print(f'\nEffective interval shift ({col}):')
for k in (2, 3):
    for ta, tb in [('2|4', '3|3'), ('2|4', 'WL'), ('3|3', 'WL')]:
        if ta in types and tb in types:
            sa = iv[(iv.type == ta) & (iv.interval == k)][col]
            sb = iv[(iv.type == tb) & (iv.interval == k)][col]
            tag = lambda t: 'BOUNDARY' if BOUNDARY.get(t) == k else 'within'
            contrast(f'I{k}: {ta} ({tag(ta)}) vs {tb} ({tag(tb)})', sa, sb)
for t in [t for t in types if t in BOUNDARY]:
    sub = iv[iv.type == t]
    contrast(f'{t}: boundary I{BOUNDARY[t]} vs its within-phrase intervals',
             sub[sub.boundary][col], sub[~sub.boundary][col])

print(f'\nOnset of the word itself ({MAIN}):')
for pos in (3, 4):
    for ta, tb in [('2|4', '3|3'), ('2|4', 'WL'), ('3|3', 'WL')]:
        if ta in types and tb in types:
            contrast(f'word {pos}: {ta} vs {tb}',
                     wl[(wl.type == ta) & (wl.pos == pos)][MAIN], wl[(wl.type == tb) & (wl.pos == pos)][MAIN])
for t in [t for t in types if t in BOUNDARY]:
    bpos = BOUNDARY[t] + 1
    sub = wl[(wl.type == t) & (wl.pos > 1)]
    contrast(f'{t}: post-boundary word {bpos} vs other words 2-6',
             sub[sub.pos == bpos][MAIN], sub[sub.pos != bpos][MAIN])
pd.DataFrame(res).to_csv(os.path.join(OUT_DIR, '04_contrasts.csv'), index=False, float_format='%.3f')

# ---------------------------------------------------------------- 4. junction types
hr('4. SHIFT BY JUNCTION CLASS (main measure)')
jt = iv.groupby(['type', 'interval', 'junction_class'])[col].agg(['size', 'mean', 'std']).round(1)
print(jt.to_string())

# ---------------------------------------------------------------- 5. item flags
hr(f'5. ITEMS WHOSE BOUNDARY INTERVAL DEVIATES > {FLAG_MS} ms FROM THE TYPE\'S WITHIN-PHRASE MEAN')
flags = []
for t in [t for t in types if t in BOUNDARY]:
    sub = iv[iv.type == t]
    ref = sub[~sub.boundary][col].mean()
    b = sub[sub.boundary].copy()
    b['dev_from_within_mean'] = b[col] - ref
    f = b[b.dev_from_within_mean.abs() > FLAG_MS]
    print(f'\n{t}: within-phrase mean {ref:+.1f} ms; flagged {len(f)} of {len(b)}')
    if len(f):
        print(f[['item', 'junction', col, 'dev_from_within_mean']].sort_values('dev_from_within_mean')
              .round(1).to_string(index=False))
    flags.append(b)
if flags:
    pd.concat(flags).to_csv(os.path.join(OUT_DIR, '04_item_flags.csv'), index=False, float_format='%.2f')

# ---------------------------------------------------------------- plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for t in types:
        g = wl[wl.type == t].groupby('pos')[MAIN]
        axes[0].errorbar(g.mean().index, g.mean(), yerr=g.sem(), marker='o', capsize=3, label=t)
        g = iv[iv.type == t].groupby('interval')[col]
        axes[1].errorbar(g.mean().index, g.mean(), yerr=g.sem(), marker='o', capsize=3, label=t)
        if t in BOUNDARY:
            k = BOUNDARY[t]
            axes[1].plot(k, g.mean()[k], 's', ms=14, mfc='none', mec='k')
    axes[0].set(xlabel='word position', ylabel=f'{MAIN} (ms from file start)', title='Word onset by position')
    axes[1].axhline(0, color='k', lw=0.6)
    axes[1].set(xlabel='interval', ylabel='effective shift (ms)',
                title='Effective interval shift (square = boundary)')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, '04_by_position.png'), dpi=120)
    print(f'\nPlot: {os.path.join(OUT_DIR, "04_by_position.png")}')
except Exception as e:
    print(f'[plot skipped: {e}]')
print(f'CSVs in: {OUT_DIR}')