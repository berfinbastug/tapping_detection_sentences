# %% MAKE THE TAPPING TRAINING DESIGN (ONCE, THE SAME FOR EVERY PARTICIPANT)
"""
Makes the one training table that every participant of the tapping experiment gets (fixed seed: running it again
gives the same table).

Phases
    listen       the example word list (EXAMPLE_ITEM), no shift - participants only listen
    tap_example  the same sound - participants tap; feedback with a figure
    stumble      the same word list with one large shift (STUMBLE_*) - participants tap; written feedback
    test         6 sequences played like main trials: TEST_ITEMS x TEST_REPS; TEST_SHIFTS get one shift each
                 (+ or - TEST_SIZE_MS, random direction), the other trials have no shift

Inputs
    sentence_lists/training_items_tapping.csv   the example word list (TR_TAP_WL)
    sentence_lists/training_items.csv           the detection training items (TR_S24, TR_S33, TR_WL)
Outputs
    design_tables/tapping_training_design.csv   one row per training trial (the examples once each)
    design_tables/tapping_training_events.csv   one row per sound of every distinct stimulus (30 per stim_id):
                                                intended onsets with the same columns as the main event table

Jitter = PHASE SHIFT, as in the main experiment: the shifted sound and every later sound move by the shift.
Interval Ik of a part = the gap before that part's sound k+1. Lead-in 1-6, S1 7-12, A 13-18, S2 19-24, B 25-30.
"""
import os

import numpy as np
import pandas as pd

PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(PROJECT_DIR)


# %% SETTINGS
SEED = 2026

EXAMPLE_ITEM = 'TR_TAP_WL'                 # listen / tap example / stumble example
STUMBLE_PART, STUMBLE_INTERVAL, STUMBLE_SHIFT_MS = 'S2', 'I2', +100      # + = late (as detection's demo)

TEST_ITEMS = ['TR_S24', 'TR_S33', 'TR_WL']  # detection training items (same words as detection's training)
TEST_REPS = 2                               # every item twice -> 6 test trials
TEST_SHIFTS = [('S2', 'I1'), ('S2', 'I2'), ('S2', 'I3'), ('A', 'I2')]   # one shift per trial; the rest unshifted
TEST_SIZE_MS = 90                           # direction drawn at random per shift

ITEM_FILES = ['sentence_lists/training_items_tapping.csv', 'sentence_lists/training_items.csv']
OUTPUT_DIR = 'tapping/design_tables'
IOI_MS, EVENT_MS, N_EVENTS = 450, 350, 30

PARTS = ['lead-in', 'S1', 'A', 'S2', 'B']
PART_KIND = {'lead-in': 'noise', 'S1': 'word', 'A': 'noise', 'S2': 'word', 'B': 'noise'}
PART_OFFSET = {'A': 12, 'S2': 18, 'B': 24}  # 0-based index of each part's first sound


def moved_event(part, interval):
    """1-based number of the first sound that moves for a shift at `interval` of `part`."""
    return PART_OFFSET[part] + int(interval[1:]) + 1


# %% ITEMS
items = pd.concat([pd.read_csv(f, dtype=str) for f in ITEM_FILES], ignore_index=True)
if items['item_id'].duplicated().any():
    raise ValueError(f"item_id used twice in {ITEM_FILES}: {sorted(items.loc[items['item_id'].duplicated(), 'item_id'])}")
items = items.set_index('item_id')
for i in [EXAMPLE_ITEM] + TEST_ITEMS:
    if i not in items.index:
        raise ValueError(f'{i} is not in {ITEM_FILES}')
words_of = lambda i: ' '.join(items.loc[i, [f'slot{k}' for k in range(1, 7)]])


def stim_id(item, part=None, interval=None, shift=0):
    if not shift:
        return f'{item}_none'
    return f"{item}_{part}{interval}_{'p' if shift > 0 else 'm'}{abs(int(shift))}"


def row(phase, trial, item, part=None, interval=None, shift=0):
    return dict(phase=phase, trial=trial, item_id=item, sentence_type=items.loc[item, 'type'], words=words_of(item),
                jittered=bool(shift), jitter_part=part, jitter_interval=interval,
                jitter_event=moved_event(part, interval) if shift else np.nan,
                shift_ms=shift, stim_id=stim_id(item, part, interval, shift))


# %% THE TABLE
rng = np.random.default_rng(SEED)
rows = [row('listen', 1, EXAMPLE_ITEM),
        row('tap_example', 1, EXAMPLE_ITEM),
        row('stumble', 1, EXAMPLE_ITEM, STUMBLE_PART, STUMBLE_INTERVAL, STUMBLE_SHIFT_MS)]

# test run: TEST_ITEMS x TEST_REPS trials; the unshifted trials go to different items; no item twice in a row
n_test = len(TEST_ITEMS) * TEST_REPS
n_clean = n_test - len(TEST_SHIFTS)
for _ in range(10_000):
    trials = [i for i in TEST_ITEMS for _ in range(TEST_REPS)]
    trials = [trials[j] for j in rng.permutation(n_test)]
    if any(a == b for a, b in zip(trials, trials[1:])):
        continue
    conds = [None] * n_clean + [s for s in TEST_SHIFTS]
    conds = [conds[j] for j in rng.permutation(n_test)]
    clean_items = [i for i, c in zip(trials, conds) if c is None]
    if len(set(clean_items)) == len(clean_items):                   # unshifted trials on different items
        break
else:
    raise RuntimeError('could not order the test run')
for k, (item, c) in enumerate(zip(trials, conds), start=1):
    if c is None:
        rows.append(row('test', k, item))
    else:
        rows.append(row('test', k, item, c[0], c[1], int(rng.choice([-1, 1])) * TEST_SIZE_MS))
design = pd.DataFrame(rows)


# %% EVENT TABLE (intended timing of every distinct stimulus)
def events_of(r):
    step = np.zeros(N_EVENTS)
    if r['jittered']:
        step[int(r['jitter_event']) - 1] = r['shift_ms']
    shift = np.cumsum(step)
    onset = np.arange(N_EVENTS) * IOI_MS + shift
    return [dict(stim_id=r['stim_id'], event=e + 1, part=PARTS[e // 6], pos_in_part=e % 6 + 1,
                 kind=PART_KIND[PARTS[e // 6]], grid_ms=e * IOI_MS, jitter_here=bool(step[e] != 0),
                 jitter_step_ms=step[e], shift_ms=shift[e], intended_onset_ms=onset[e],
                 ioi_before_ms=onset[e] - onset[e - 1] if e > 0 else np.nan) for e in range(N_EVENTS)]


stims = design.drop_duplicates('stim_id')
events = pd.DataFrame([e for _, r in stims.iterrows() for e in events_of(r)])


# %% CHECKS
def check(design, events):
    t = design[design.phase == 'test']
    assert len(t) == n_test and (t.item_id.value_counts() == TEST_REPS).all(), 'test items'
    assert (t.item_id.values[1:] != t.item_id.values[:-1]).all(), 'test item twice in a row'
    shifted = t[t.jittered]
    assert sorted(zip(shifted.jitter_part, shifted.jitter_interval)) == sorted(TEST_SHIFTS), 'test shift positions'
    assert (shifted.shift_ms.abs() == TEST_SIZE_MS).all() and (~t.jittered).sum() == n_clean, 'test shift sizes'
    assert t.loc[~t.jittered, 'item_id'].is_unique, 'unshifted test trials on the same item'
    ex = design[design.phase != 'test']
    assert (ex.item_id == EXAMPLE_ITEM).all() and ex.loc[ex.phase == 'stumble', 'shift_ms'].item() == STUMBLE_SHIFT_MS
    assert ex.loc[ex.phase == 'listen', 'stim_id'].item() == ex.loc[ex.phase == 'tap_example', 'stim_id'].item()
    # never shifted: lead-in, S1 and the first sound of every part; one interval changes per shifted stimulus
    assert (events.loc[events.event <= 12, 'shift_ms'] == 0).all()
    assert not events.loc[events.pos_in_part == 1, 'jitter_here'].any()
    for sid, e in events.groupby('stim_id'):
        ioi = np.diff(e.intended_onset_ms.to_numpy())
        assert np.sum(ioi != IOI_MS) == int(design.loc[design.stim_id == sid, 'jittered'].iloc[0]), sid
        assert (ioi - EVENT_MS).min() > 0, f'{sid}: sounds overlap'
    print('✓ All checks passed')


check(design, events)
print(design[['phase', 'trial', 'item_id', 'words', 'jitter_part', 'jitter_interval', 'shift_ms', 'stim_id']]
      .to_string(index=False))

# %% SAVE
os.makedirs(OUTPUT_DIR, exist_ok=True)
design.to_csv(os.path.join(OUTPUT_DIR, 'tapping_training_design.csv'), index=False)
events.to_csv(os.path.join(OUTPUT_DIR, 'tapping_training_events.csv'), index=False)
print('Saved', os.path.join(OUTPUT_DIR, 'tapping_training_design.csv'), 'and',
      os.path.join(OUTPUT_DIR, 'tapping_training_events.csv'))