# %% TRAINING DESIGN TABLE
"""
One training table for ALL participants (fixed seed -> generate once).

    phase 'demo'     : 2 illustration trials with the demo word list: one regular, one irregular
    phase 'practice' : 6 trials = 3 items (2|4, 3|3, word list) x 2
                       N_REGULAR regular + the rest irregular (half early, half late): late = SHIFT_MS,
                       early = SHIFT_EARLY_MS;
                       random interval per irregular trial (any of I1-I5, all different if possible);
                       an item never gets the same condition twice; never the same item twice in a row

Jitter = phase shift at interval Ik: word k+1 and everything after it move (as in the main experiment).
Output: design_tables/training_design.csv
"""
import os
import numpy as np
import pandas as pd

# %% SETTINGS
TRAINING_CSV   = 'sentence_lists/training_items.csv'
OUT_CSV        = 'design_tables/training_design.csv'
SEED           = 2026

DEMO_ITEM      = 'TR_DEMO_WL'
DEMO_INTERVAL  = 'I3'          # middle of the sequence: easiest to hear in the illustration; None = random
DEMO_SHIFT_MS  = 100            # + = late

PRACTICE_ITEMS = ['TR_S24', 'TR_S33', 'TR_WL']
REPS           = 2             # each practice item twice -> 6 trials
SHIFT_MS       = 90            # late shifts (+)
SHIFT_EARLY_MS = 90            # early shifts (-); with ~350 ms words, more than ~60 leaves < 40 ms between words
N_REGULAR      = 2             # 2 regular + 4 irregular (as in the instructions); 0 = all shifted
INTERVALS      = ['I1', 'I2', 'I3', 'I4', 'I5']

# %% BUILD
rng = np.random.default_rng(SEED)
items = pd.read_csv(TRAINING_CSV).set_index('item_id')


def stim_id(item, interval, shift):
    if interval is None or shift == 0:
        return f'{item}_none'
    return f"{item}_{interval}_{'p' if shift > 0 else 'm'}{abs(int(shift))}"


def trial(phase, item, role, interval, shift):
    jit = interval is not None and shift != 0
    return dict(phase=phase, item_id=item, sentence_type=items.loc[item, 'type'], role=role,
                jittered=jit, interval=interval if jit else None,
                moved_word=int(interval[1:]) + 1 if jit else None,
                size_ms=abs(shift) if jit else None,
                direction=('late' if shift > 0 else 'early') if jit else None,
                shift_ms=shift if jit else 0,
                correct_response='irregular' if jit else 'regular',
                stim_id=stim_id(item, interval, shift))


# demo: regular, then irregular
demo_iv = DEMO_INTERVAL or str(rng.choice(INTERVALS))
demo = [trial('demo', DEMO_ITEM, 'demo_regular', None, 0),
        trial('demo', DEMO_ITEM, 'demo_irregular', demo_iv, DEMO_SHIFT_MS)]

# practice: condition cards
n_trials = len(PRACTICE_ITEMS) * REPS
n_irr = n_trials - N_REGULAR
signs = [-1] * (n_irr // 2) + [1] * (n_irr // 2) + ([int(rng.choice([-1, 1]))] if n_irr % 2 else [])
ivs = list(rng.choice(INTERVALS, size=n_irr, replace=n_irr > len(INTERVALS)))
cards = [('regular', None, 0)] * N_REGULAR + \
        [('irregular', iv, SHIFT_MS if s > 0 else -SHIFT_EARLY_MS) for iv, s in zip(ivs, signs)]

slots = [it for it in PRACTICE_ITEMS for _ in range(REPS)]
for _ in range(10_000):                                    # an item never gets the same condition twice
    deal = [cards[j] for j in rng.permutation(n_trials)]
    conds = {}
    for it, c in zip(slots, deal):
        conds.setdefault(it, []).append(('regular', 0) if c[0] == 'regular' else ('irr', np.sign(c[2])))
    if all(len(set(v)) == len(v) for v in conds.values()):
        break
else:
    raise RuntimeError('could not assign conditions')
practice = [trial('practice', it, 'practice', c[1], c[2]) for it, c in zip(slots, deal)]

for _ in range(10_000):                                    # order: no item twice in a row
    order = [practice[j] for j in rng.permutation(n_trials)]
    if all(a['item_id'] != b['item_id'] for a, b in zip(order, order[1:])):
        break
else:
    raise RuntimeError('could not order the practice trials')

table = pd.DataFrame(demo + order)
table.insert(1, 'trial', table.groupby('phase').cumcount() + 1)
table.insert(3, 'words', [' '.join(items.loc[i, [f'slot{k}' for k in range(1, 7)]]) for i in table['item_id']])

# %% CHECKS
p = table[table.phase == 'practice']
assert len(p) == n_trials and (p.item_id.value_counts() == REPS).all()
assert (p.correct_response == 'regular').sum() == N_REGULAR
assert set(p.size_ms.dropna()) <= {SHIFT_MS, SHIFT_EARLY_MS}
assert abs((p.direction == 'early').sum() - (p.direction == 'late').sum()) <= 1
assert (p.item_id.values[1:] != p.item_id.values[:-1]).all()

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
table.to_csv(OUT_CSV, index=False)
print(f'Saved {OUT_CSV}')
print(table[['phase', 'trial', 'item_id', 'sentence_type', 'role', 'interval', 'shift_ms', 'correct_response', 'stim_id']]
      .to_string(index=False))
# %%
