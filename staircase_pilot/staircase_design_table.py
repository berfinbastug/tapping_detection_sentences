# MODULES
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

import os
PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(PROJECT_DIR)


# %%
# SETTINGS
VALID_GROUPS = ["speech", "acoustic"]

def ask_settings():
    pid = input("Enter participant ID: ").strip().upper()
    while not pid:
        pid = input("Participant ID cannot be empty. Enter participant ID: ").strip().upper()
    group = input(f"Enter experimental group ({' / '.join(VALID_GROUPS)}): ").strip().lower()
    while group not in VALID_GROUPS:
        group = input(f'"{group}" is not valid. Please type {" or ".join(VALID_GROUPS)}: ').strip().lower()
    return pid, group

PARTICIPANT_ID, GROUP = ask_settings()
SEED = None                         # None -> derived from PARTICIPANT_ID (reproducible)

IOI_MS     = 450
EVENT_MS   = 350
ISI_S      = 0.8                    # silence between the two presentations of a trial
RESPONSE_S = 1.5                    # assumed response time (duration estimate only)
ITI_S      = 0.8
OUTPUT_DIR = "staircase_pilot/design_tables_staircase"


# %%
# DESIGN CONSTANTS
N_WORDS    = 6
SOUND_S    = ((N_WORDS - 1) * IOI_MS + EVENT_MS) / 1000          # 2.6 s
TRIAL_S    = 2 * SOUND_S + ISI_S + RESPONSE_S + ITI_S            # ≈ 8.3 s

SENT_TYPES = ["2|4", "3|3", "word list"]
TYPE_CODE  = {"2|4": "S24", "3|3": "S33", "word list": "WL"}
N_ITEMS    = {"2|4": 18, "3|3": 18, "word list": 18}             # items per type: set to the final pilot counts
INTERVALS  = ["I1", "I2", "I3", "I4", "I5"]
DIRECTIONS = ["delay", "advance"]
COPIES     = [1, 2]
STAIRCASES = [f"{d}_{c}" for d in DIRECTIONS for c in COPIES]  # 4 interleaved staircases
N_REPS     = 3                                                   # per type × interval, per staircase
N_PER_SC   = len(SENT_TYPES) * len(INTERVALS) * N_REPS           # 45
N_BLOCKS   = 4
N_TRIALS   = N_PER_SC * len(STAIRCASES)                          # 180

def moved_word(interval):
    # 1-based word that moves for a shift at this interval (phase shift: it and all later words move)
    return int(interval[1:]) + 1



# %%
# DESIGN GENERATOR
def _cells_with_presentation(rng):
    # per direction: every type × interval cell has 6 trials (3 per copy) -> 3 with the shift in the 1st
    # presentation, 3 in the 2nd, split at random across the two copies.
    # Retry until each staircase has 22 or 23 trials with the shift in the 1st presentation.
    cells = [(st, iv) for st in SENT_TYPES for iv in INTERVALS]
    for _ in range(10_000):
        out = {sc: [] for sc in STAIRCASES}
        for d in DIRECTIONS:
            for c in cells:
                pres = rng.permutation([1] * N_REPS + [2] * N_REPS)
                out[f"{d}_1"] += [c + (int(p),) for p in pres[:N_REPS]]
                out[f"{d}_2"] += [c + (int(p),) for p in pres[N_REPS:]]
        if all(abs(sum(x[2] == 1 for x in v) - N_PER_SC / 2) <= 0.5 for v in out.values()):
            return {sc: [v[j] for j in rng.permutation(len(v))] for sc, v in out.items()}   # random order per staircase
    raise RuntimeError("Could not balance shifted presentation")


def _interleave(rng):
    # staircase label per trial: per block 11/11/11/12, each staircase gets its 12 in a different block
    sc_order = [STAIRCASES[j] for j in rng.permutation(len(STAIRCASES))]
    offset = int(rng.integers(N_BLOCKS))
    base = N_PER_SC // N_BLOCKS                                   # 11
    labels = []
    for b in range(N_BLOCKS):
        blk = [sc for i, sc in enumerate(sc_order) for _ in range(base + ((i + offset) % N_BLOCKS == b))]
        labels += [(b + 1, blk[j]) for j in rng.permutation(len(blk))]
    return labels


def make_design(participant_id, group, seed=None):
    if seed is None:
        seed = zlib.crc32(participant_id.encode())
    rng = np.random.default_rng(seed)

    queues = _cells_with_presentation(rng)
    labels = _interleave(rng)
    ids    = {st: [f"{TYPE_CODE[st]}_{i:02d}" for i in range(1, N_ITEMS[st] + 1)] for st in SENT_TYPES}
    decks  = {st: [] for st in SENT_TYPES}
    sc_n, seen, prev, rows = {sc: 0 for sc in STAIRCASES}, {}, None, []

    for trial, (block, sc) in enumerate(labels, start=1):
        st, iv, pres = queues[sc][sc_n[sc]]
        sc_n[sc] += 1
        # item: next from this type's shuffled deck; reshuffle when empty; never the same item twice in a row
        deck = decks[st]
        if not deck or (len(deck) == 1 and deck[0] == prev):
            deck += [ids[st][j] for j in rng.permutation(len(ids[st]))]
        k = next(i for i, x in enumerate(deck) if x != prev)
        item = deck.pop(k)
        prev = item
        seen[item] = seen.get(item, 0) + 1
        d, c = sc.split("_")
        rows.append({"participant": participant_id, "group": group, "seed": seed,
                     "trial": trial, "block": block,
                     "staircase": sc, "direction": d, "copy": int(c), "staircase_trial": sc_n[sc],
                     "sentence_type": st, "sentence_id": item, "exposure": seen[item],
                     "interval": iv, "moved_word": moved_word(iv), "shifted_presentation": pres,
                     # filled in at run time
                     "size_ms": np.nan, "shift_ms": np.nan, "response": np.nan, "correct": np.nan})
    return pd.DataFrame(rows)


# %%
# BUILD A TABLE FOR A PARTICIPANT
df = make_design(PARTICIPANT_ID, GROUP, SEED)
print(f"{len(df)} trials, seed = {df.seed.iloc[0]}")
print(f"trial ≈ {TRIAL_S:.1f} s -> trials ≈ {len(df) * TRIAL_S / 60:.1f} min "
      f"({N_BLOCKS} blocks × {len(df) // N_BLOCKS} trials, {len(df) // N_BLOCKS * TRIAL_S / 60:.1f} min per block)")



# %%
# BALANCE CHECKS
def check_balance(df, verbose=True):
    assert len(df) == N_TRIALS and df.trial.is_unique
    for sc in STAIRCASES:
        d = df[df.staircase == sc]
        assert len(d) == N_PER_SC, sc
        assert (d.staircase_trial.to_numpy() == np.arange(1, N_PER_SC + 1)).all(), sc
        g = d.groupby(["sentence_type", "interval"]).size()
        assert len(g) == 15 and (g == N_REPS).all(), sc                                   # 3 per type × interval
        n1 = (d.shifted_presentation == 1).sum()
        assert n1 in (N_PER_SC // 2, N_PER_SC // 2 + 1), (sc, n1)                       # 22 / 23
    for dr in DIRECTIONS:
        d = df[df.direction == dr]
        p = d.groupby(["sentence_type", "interval", "shifted_presentation"]).size()
        assert len(p) == 30 and (p == N_REPS).all(), dr                                  # 3 / 3 per cell and direction
    # blocks: 45 trials; each staircase 11 or 12 per block
    b = pd.crosstab(df.block, df.staircase)
    assert (b.sum(axis=1) == N_TRIALS // N_BLOCKS).all() and b.isin([11, 12]).all().all()
    # items: each type 60 trials, usage spread evenly
    for st in SENT_TYPES:
        u = df[df.sentence_type == st].sentence_id.value_counts().reindex(
            [f"{TYPE_CODE[st]}_{i:02d}" for i in range(1, N_ITEMS[st] + 1)], fill_value=0)
        assert u.sum() == N_TRIALS // len(SENT_TYPES), st
        assert u.max() - u.min() <= 1, (st, u.min(), u.max())
    assert (df.sentence_id.values[1:] != df.sentence_id.values[:-1]).all(), "item repeated back-to-back"
    if verbose:
        print("✓ All balance checks passed")

check_balance(df)


# SAVE
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
path = Path(OUTPUT_DIR) / f"staircase_{PARTICIPANT_ID}_{GROUP}.csv"
df.to_csv(path, index=False)
print("Saved", path)