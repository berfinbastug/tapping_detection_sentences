# %%
# SWITCH DESIGN TABLE (tapping)
#
# Design table for the switch experiment, for BOTH kinds of participants:
#   mode 'switch'  : blocks 1-3 acoustic (naive), then reveal, blocks 4-9 speech
#   mode 'control' : all 9 blocks acoustic
# The design is the same as in tapping/generate_tapping_design_table.py (copied unchanged below), plus ONE new
# step, balance_phase1(): within each sentence type x role the trial settings are re-dealt among the sentences, so
# that blocks 1-3 contain exactly one third of every condition (each size x direction once per type x critical
# interval; blocks 4-9 twice). Item order, blocks, roles, exposure and the session totals are unchanged
# (checked below; proof and plots in switch_design_balance.ipynb).
#
# Output: tapping/design_tables/tapping_<ID>_acoustic.csv and tapping_events_<ID>_acoustic.csv
#   The group label is ALWAYS 'acoustic' (the sounds are the same for both modes), so build_tapping_stimuli.py
#   runs unchanged with group = acoustic. The mode is stored in the columns mode / phase / phase_condition.
# Run from anywhere: PROJECT_DIR + os.chdir below. Python 3.6 compatible (lab PC).

# %%
# MODULES
import zlib
import numpy as np
import pandas as pd
from pathlib import Path


import os
PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'      # lab: /home/bastugb/Documents/tapping_detection_sentences
os.chdir(PROJECT_DIR)        # relative paths below are inside the project folder



# %%
# SETTINGS
VALID_MODES = ["switch", "control"]

def ask_settings():
    pid = input("Enter participant ID: ").strip().upper()
    while not pid:
        pid = input("Participant ID cannot be empty. Enter participant ID: ").strip().upper()
    mode = input(f"Enter mode ({' / '.join(VALID_MODES)}): ").strip().lower()
    while mode not in VALID_MODES:
        mode = input(f'"{mode}" is not valid. Please type {" or ".join(VALID_MODES)}: ').strip().lower()
    return pid, mode

PARTICIPANT_ID, MODE = ask_settings()
GROUP = "acoustic"                 # do not change: stimuli are built as 'acoustic' for both modes
SEED = None                        # None -> derived from PARTICIPANT_ID (reproducible)


JITTER_SIZES_MS    = [35, 55, 75]  # decided 21 Nov 2026 (staircase_pilot_plan.md)
MIN_EVENTS_BETWEEN = 6             # event-index distance between the 2 jittered events (6 = 5 clean events between)
IOI_MS             = 450           # onset-to-onset
EVENT_MS           = 350           # event duration
ITI_S              = 2.5           # pause between trials (only for the duration estimate)
OUTPUT_DIR         = "tapping/design_tables"

PHASE1_BLOCKS      = [1, 2, 3]     # phase 1 = blocks 1-3 (acoustic for everyone), phase 2 = blocks 4-9
PHASE_CONDITION    = {"switch": {1: "acoustic", 2: "speech"}, "control": {1: "acoustic", 2: "acoustic"}}

# %%
# DESIGN CONSTANTS
N_EVENTS         = 30
TRIAL_DURATION_S = ((N_EVENTS - 1) * IOI_MS + EVENT_MS) / 1000   # 13.4 s

SENT_TYPES  = ["2|4", "3|3", "word list"]
TYPE_CODE   = {"2|4": "S24", "3|3": "S33", "word list": "WL"}
N_SENTENCES = 18                   # per sentence type
N_BLOCKS    = 9                    # 18 sentences / 9 blocks = 2 per role per block
INTERVALS   = ["I1", "I2", "I3", "I4", "I5"]
CRITICAL    = ["I2", "I3"]
S2_FILLER   = ["I1", "I4", "I5"]
DIRECTIONS  = ["early", "late"]
COMBOS      = [(s, d) for s in JITTER_SIZES_MS for d in DIRECTIONS]      # 6 size × direction levels
ROLES       = ["critical_I2", "critical_I3", "filler", "baseline"]

A_INTERVALS = ["I1", "I2"]         # A: I1–I2 (events 14–15) -> at least 3 clean noises before S2   # CHANGED (comment only)
B_INTERVALS = ["I1", "I2"]         # B: I1–I2  (events 26–27) -> at least 3 events after the jitter
BASELINE_PATTERNS = {"none": 15, "A+B": 3}
N_B_NOISE = {"I2": 1, "I1": 1}     # per sentence type: how many S2+noise trials put the noise in B (only I1, I2 allow B)
PATTERN_PARTS = {"none": [], "S2": ["S2"], "A+S2": ["A", "S2"], "S2+B": ["S2", "B"], "A+B": ["A", "B"]}

# 0-based event index of the first event of each part: lead-in 0–5, S1 6–11, A 12–17, S2 18–23, B 24–29.
# Interval Ik of a part = gap before that part's event k+1  ->  moved event index = offset + k.
PART_OFFSET = {"A": 12, "S2": 18, "B": 24}

def gap_index(part, interval):
    # 0-based index of the event that is moved by a jitter at `interval` of `part`
    return PART_OFFSET[part] + int(interval[1:])

def legal_noise(s2_interval):
    # all (part, interval) positions where a noise jitter may go, given the S2 jitter
    s = gap_index("S2", s2_interval)
    opts  = [("A", iv) for iv in A_INTERVALS if s - gap_index("A", iv) >= MIN_EVENTS_BETWEEN]
    opts += [("B", iv) for iv in B_INTERVALS if gap_index("B", iv) - s >= MIN_EVENTS_BETWEEN]
    return opts

for iv in INTERVALS:
    print(f"S2 jitter at {iv} (event {gap_index('S2', iv) + 1}): legal noise positions ->",
          [f"{p} {i} (event {gap_index(p, i) + 1})" for p, i in legal_noise(iv)])
    


# %%
# DESIGN GENERATOR
def _assign_blocks(sids, rng):
    # (sentence, role) -> block; 2 sentences per role per block; 4 different blocks per sentence
    place = {}
    for role in ROLES:
        for j, s in enumerate(rng.permutation(len(sids))):
            place[(sids[s], role)] = j % N_BLOCKS
    clash = lambda s: len(ROLES) - len({place[(s, r)] for r in ROLES})
    total = sum(clash(s) for s in sids)
    for _ in range(200_000):
        if total == 0:
            return place
        bad = [s for s in sids if clash(s)]
        s = bad[rng.integers(len(bad))]
        r = ROLES[rng.integers(len(ROLES))]
        o = sids[rng.integers(len(sids))]
        if o == s or place[(o, r)] == place[(s, r)]:
            continue
        before = clash(s) + clash(o)
        place[(s, r)], place[(o, r)] = place[(o, r)], place[(s, r)]
        after = clash(s) + clash(o)
        if after > before:
            place[(s, r)], place[(o, r)] = place[(o, r)], place[(s, r)]
        else:
            total += after - before
    raise RuntimeError("Block assignment failed")


def _place_noise(ts, n_b, rng):
    # give each trial in ts a noise jitter: n_b of them in B, the rest in A; interval random among legal ones
    for k, j in enumerate(rng.permutation(len(ts))):
        t = ts[j]
        part = "B" if k < n_b else "A"
        opts = [iv for p, iv in legal_noise(t["S2_interval"]) if p == part]
        assert opts, (t["S2_interval"], part)
        t["pattern"] = "A+S2" if part == "A" else "S2+B"
        t[f"{part}_interval"] = str(opts[rng.integers(len(opts))])


def _dist(t):
    parts = PATTERN_PARTS[t["pattern"]]
    if len(parts) < 2:
        return None
    g = sorted(gap_index(p, t[f"{p}_interval"]) for p in parts)
    return g[1] - g[0]


def make_design(participant_id, group, seed=None):
    if seed is None:
        seed = zlib.crc32(participant_id.encode())
    rng = np.random.default_rng(seed)
    all_trials = []

    for st in SENT_TYPES:
        sids = [f"{TYPE_CODE[st]}_{i:02d}" for i in range(1, N_SENTENCES + 1)]
        trials = []
        base = lambda sid, role, pat: {"sentence_type": st, "sentence_id": sid, "role": role, "pattern": pat}

        # --- which size×direction carries an extra noise jitter (6 critical + 6 filler per type)
        crit_noise = {"I2": [], "I3": []}
        for s in JITTER_SIZES_MS:                       # per size: one direction at I2, the other at I3
            d = rng.permutation(DIRECTIONS)
            crit_noise["I2"].append((s, str(d[0])))
            crit_noise["I3"].append((s, str(d[1])))
        early, late = rng.permutation(JITTER_SIZES_MS), rng.permutation(JITTER_SIZES_MS)
        fill_noise = {iv: [(int(early[i]), "early"), (int(late[i]), "late")] for i, iv in enumerate(S2_FILLER)}

        # --- critical: 6 size×direction × 3 reps (one rep of the chosen levels carries noise)
        for role, iv in (("critical_I2", "I2"), ("critical_I3", "I3")):
            cells = [(c, rep == 0 and c in crit_noise[iv]) for c in COMBOS for rep in range(3)]   # 18
            new = []
            for sid, j in zip(sids, rng.permutation(len(cells))):
                combo, noise = cells[j]
                new.append({**base(sid, role, "S2"), "S2_interval": iv, "S2_combo": combo, "noise": noise})   # Python 3.6: no dict |
            _place_noise([t for t in new if t["noise"]], N_B_NOISE.get(iv, 0), rng)
            trials += new

        # --- filler: 3 intervals × 6 size×direction, 1 each (6 of them carry noise)
        fcells = [(iv, c, c in fill_noise[iv]) for iv in S2_FILLER for c in COMBOS]           # 18
        new = []
        for sid, j in zip(sids, rng.permutation(len(fcells))):
            iv, c, noise = fcells[j]
            new.append({**base(sid, "filler", "S2"), "S2_interval": iv, "S2_combo": c, "noise": noise})   # Python 3.6: no dict |
        for iv in S2_FILLER:
            _place_noise([t for t in new if t["noise"] and t["S2_interval"] == iv], N_B_NOISE.get(iv, 0), rng)
        trials += new

        # --- baseline: 15 none + 3 A+B
        bpats = [p for p, n in BASELINE_PATTERNS.items() for _ in range(n)]           # 18
        for sid, j in zip(sids, rng.permutation(len(bpats))):
            t = base(sid, "baseline", bpats[j])
            if bpats[j] == "A+B":
                t["A_interval"] = str(rng.choice(A_INTERVALS))
                t["B_interval"] = str(rng.choice(B_INTERVALS))
            trials.append(t)

        # --- noise sizes: all 18 noise jitters of this type get 6 size×direction × 3
        slots = [(t, p) for t in trials for p in ("A", "B") if p in PATTERN_PARTS[t["pattern"]]]
        assert len(slots) == 18
        for (t, p), j in zip(slots, rng.permutation(len(slots))):
            t[f"{p}_combo"] = COMBOS[j % 6]

        # --- blocks
        place = _assign_blocks(sids, rng)
        for t in trials:
            t["block"] = place[(t["sentence_id"], t["role"])] + 1
        all_trials += trials

    # --- order: shuffle within blocks; no sentence in two consecutive trials
    order = []
    for b in range(1, N_BLOCKS + 1):
        blk = [t for t in all_trials if t["block"] == b]
        for _ in range(10_000):
            blk = [blk[j] for j in rng.permutation(len(blk))]
            seq = ([order[-1]] if order else []) + blk
            if all(x["sentence_id"] != y["sentence_id"] for x, y in zip(seq, seq[1:])):
                break
        else:
            raise RuntimeError("Could not avoid back-to-back repeats")
        order += blk

    out, seen = [], {}
    for i, t in enumerate(order, start=1):
        parts = PATTERN_PARTS[t["pattern"]]
        seen[t["sentence_id"]] = seen.get(t["sentence_id"], 0) + 1
        r = {"participant": participant_id, "group": group, "seed": seed,
             "trial": i, "block": t["block"],
             "sentence_type": t["sentence_type"], "sentence_id": t["sentence_id"],
             "exposure": seen[t["sentence_id"]], "role": t["role"],
             "n_jitters": len(parts), "jitter_pattern": t["pattern"]}
        for part in ("A", "S2", "B"):
            on = part in parts
            size, direction = t[f"{part}_combo"] if on else (np.nan, None)
            r[f"{part}_jitter"] = on
            r[f"{part}_interval"] = t[f"{part}_interval"] if on else None
            r[f"{part}_event"] = gap_index(part, t[f"{part}_interval"]) + 1 if on else np.nan   # 1-based
            r[f"{part}_size_ms"] = size
            r[f"{part}_direction"] = direction
            r[f"{part}_shift_ms"] = (size if direction == "late" else -size) if on else 0
        r["S2_critical"] = t["role"].startswith("critical")
        r["S2_with_noise"] = "S2" in parts and len(parts) == 2
        r["critical_with_noise"] = r["S2_critical"] and r["S2_with_noise"]
        d = _dist(t)
        r["jitter_distance"] = d                                   # event-index distance (>= MIN_EVENTS_BETWEEN)
        r["clean_events_between"] = None if d is None else d - 1  # unjittered events between the two
        out.append(r)
    df = pd.DataFrame(out)
    df["S2_boundary_status"] = np.select(
        [~df.S2_critical, df.sentence_type == "word list",
         ((df.sentence_type == "2|4") & (df.S2_interval == "I2")) | ((df.sentence_type == "3|3") & (df.S2_interval == "I3"))],
        ["(not critical)", "no structure", "boundary"], default="within-phrase")
    return df




# %%
# ======================================================================================== SWITCH ↓
# PHASE BALANCING (the only new step)
# For each sentence type x role there are 18 trials: 6 in blocks 1-3, 12 in blocks 4-9 (fixed by make_design).
# Their 18 SETTINGS (S2 interval / size / direction, extra noise shift and everything derived from them) are
# re-dealt among those 18 sentences:
#   critical I2 / I3 : blocks 1-3 get each size x direction once and 1 of the 3 noise-carrying settings
#   filler           : blocks 1-3 get each size x direction once, each interval twice (1 early + 1 late),
#                      2 of the 6 noise-carrying settings
#   baseline         : blocks 1-3 get 5 unshifted + 1 noise-only (A+B)
# Blocks 4-9 get the rest (= exactly twice each). Which setting goes to which sentence stays random
# (rng seeded from the participant's seed). Never touched: ID_COLS.
ID_COLS = ["participant", "group", "seed", "trial", "block", "sentence_type", "sentence_id", "exposure", "role"]


def _phase1_ok(sub, role):
    if role.startswith("critical"):
        return sub.S2_with_noise.sum() == 1
    if role == "filler":
        return ((sub.S2_interval.value_counts() == 2).all() and sub.S2_interval.nunique() == 3
                and (sub.groupby("S2_interval").S2_direction.nunique() == 2).all()
                and sub.S2_with_noise.sum() == 2)
    return True


def _pick_phase1(settings, role, rng):
    # indices (into settings) of the 6 settings that go to blocks 1-3
    for _ in range(100_000):
        if role == "baseline":
            none = settings.index[settings.jitter_pattern == "none"]
            ab = settings.index[settings.jitter_pattern == "A+B"]
            idx = list(rng.choice(none, 5, replace=False)) + list(rng.choice(ab, 1))
        else:                                                  # one setting per size x direction
            idx = [rng.choice(g.index) for _, g in settings.groupby(["S2_size_ms", "S2_direction"])]
        if _phase1_ok(settings.loc[idx], role):
            return idx
    raise RuntimeError(f"no balanced phase-1 set found for {role}")


def balance_phase1(df, phase1_blocks=PHASE1_BLOCKS):
    rng = np.random.default_rng(int(df.seed.iloc[0]) + 1)
    set_cols = [c for c in df.columns if c not in ID_COLS]
    out = df.copy()
    for (st, role), g in df.groupby(["sentence_type", "role"], sort=True):
        settings = g[set_cols].reset_index(drop=True)
        p1_rows = g.index[g.block.isin(phase1_blocks)]
        p2_rows = g.index[~g.block.isin(phase1_blocks)]
        assert len(p1_rows) == 6 and len(p2_rows) == 12, (st, role)
        p1 = _pick_phase1(settings, role, rng)
        p2 = [i for i in settings.index if i not in p1]
        out.loc[p1_rows, set_cols] = settings.loc[rng.permutation(p1)].to_numpy()
        out.loc[p2_rows, set_cols] = settings.loc[rng.permutation(p2)].to_numpy()
    return out.astype(df.dtypes.to_dict())
# ======================================================================================== SWITCH ↑



# %%
# BUILD A TABLE FOR A GIVEN PARTICIPANT
df_random = make_design(PARTICIPANT_ID, GROUP, SEED)      # = the table generate_tapping_design_table.py makes
df = balance_phase1(df_random)
df["mode"] = MODE
df["phase"] = np.where(df.block.isin(PHASE1_BLOCKS), 1, 2)
df["phase_condition"] = df.phase.map(PHASE_CONDITION[MODE])
print(f"{PARTICIPANT_ID} ({MODE}): {len(df)} trials, {df.sentence_id.nunique()} sentences, seed = {df.seed.iloc[0]}")
print(f"tapping time ≈ {len(df) * (TRIAL_DURATION_S + ITI_S) / 60:.1f} min "
      f"({N_BLOCKS} blocks × {len(df) // N_BLOCKS} trials, {(len(df) // N_BLOCKS) * (TRIAL_DURATION_S + ITI_S) / 60:.1f} min per block)")
print("phase 1 = blocks", PHASE1_BLOCKS, "| condition per phase:", PHASE_CONDITION[MODE])
df.head(12)

# %%
# BALANCE CHECKS
def _noise(d):
    a = d.loc[d.A_jitter, ["A_size_ms", "A_direction", "A_interval"]].set_axis(["size", "dir", "interval"], axis=1).assign(part="A")
    b = d.loc[d.B_jitter, ["B_size_ms", "B_direction", "B_interval"]].set_axis(["size", "dir", "interval"], axis=1).assign(part="B")
    return pd.concat([a, b], ignore_index=True)


def check_balance(df):
    assert len(df) == 216 and df.trial.is_unique
    for st in SENT_TYPES:
        d = df[df.sentence_type == st]
        assert len(d) == 72, st
        assert d.role.value_counts().to_dict() == {r: 18 for r in ROLES}, st
        assert d.n_jitters.value_counts().to_dict() == {0: 15, 1: 42, 2: 15}, st
        s2 = d[d.S2_jitter]
        assert s2.S2_interval.value_counts().to_dict() == {"I2": 18, "I3": 18, "I1": 6, "I4": 6, "I5": 6}, st
        crit = d[d.S2_critical]
        g = crit.groupby(["S2_interval", "S2_size_ms", "S2_direction"])
        assert len(g) == 12 and (g.size() == 3).all(), st                       # 3 reps per cell
        assert crit.S2_with_noise.sum() == 6, st
        assert (crit.groupby(["S2_size_ms", "S2_direction"]).S2_with_noise.sum() == 1).all(), st   # each level once
        assert (crit.groupby(["S2_interval", "S2_size_ms"]).S2_with_noise.sum() == 1).all(), st    # per interval 1 per size
        fl = d[d.role == "filler"]
        f = fl.groupby(["S2_interval", "S2_size_ms", "S2_direction"]).size()
        assert len(f) == 18 and (f == 1).all(), st                              # fillers fully crossed
        assert fl.S2_with_noise.sum() == 6, st
        assert (fl.groupby(["S2_size_ms", "S2_direction"]).S2_with_noise.sum() == 1).all(), st     # each level once
        assert (fl.groupby(["S2_interval", "S2_direction"]).S2_with_noise.sum() == 1).all(), st    # per interval 1 early + 1 late
        assert d.jitter_pattern.value_counts().to_dict() == {"S2": 42, "none": 15, "A+S2": 10, "A+B": 3, "S2+B": 2}, st
        base = d[d.role == "baseline"]
        assert base.jitter_pattern.value_counts().to_dict() == BASELINE_PATTERNS, st
        nz = _noise(d).groupby(["size", "dir"]).size()
        assert len(nz) == 6 and (nz == 3).all(), st                             # noise sizes balanced
    # legal positions
    assert df.loc[df.B_jitter, "B_interval"].isin(B_INTERVALS).all()
    assert df.loc[df.A_jitter, "A_interval"].isin(A_INTERVALS).all()
    assert (df.loc[df.n_jitters == 2, "jitter_distance"] >= MIN_EVENTS_BETWEEN).all(), "jitters too close"
    # sentences: 4 trials, one of each role, 4 different blocks
    g = df.groupby("sentence_id")
    assert (g.size() == 4).all() and (g.role.nunique() == 4).all() and (g.block.nunique() == 4).all()
    assert (df.groupby(["block", "sentence_type"]).size() == 8).all()
    assert (df.groupby(["block", "sentence_type", "role"]).size() == 2).all()
    assert (df.sentence_id.values[1:] != df.sentence_id.values[:-1]).all(), "sentence repeated back-to-back"
    print("✓ All balance checks passed")

check_balance(df)






# %%
# ======================================================================================== SWITCH ↓
# PHASE CHECKS: blocks 1-3 = exactly one third of every condition; nothing else changed
def check_phase_balance(df):
    for st in SENT_TYPES:
        for ph, k in ((1, 1), (2, 2)):                          # phase 1: 1 of each, phase 2: 2 of each
            d = df[(df.sentence_type == st) & (df.phase == ph)]
            crit = d[d.S2_critical]
            g = crit.groupby(["S2_interval", "S2_size_ms", "S2_direction"]).size()
            assert len(g) == 12 and (g == k).all(), (st, ph, "critical size x direction")
            assert (crit.groupby("S2_interval").S2_with_noise.sum() == k).all(), (st, ph, "critical noise")
            fl = d[d.role == "filler"]
            f = fl.groupby(["S2_size_ms", "S2_direction"]).size()
            assert len(f) == 6 and (f == k).all(), (st, ph, "filler size x direction")
            f = fl.groupby(["S2_interval", "S2_direction"]).size()
            assert len(f) == 6 and (f == k).all(), (st, ph, "filler interval x direction")
            assert fl.S2_with_noise.sum() == 2 * k, (st, ph, "filler noise")
            base = d[d.role == "baseline"].jitter_pattern.value_counts().to_dict()
            assert base == {"none": 5 * k, "A+B": k}, (st, ph, "baseline")
    print("✓ Phase balance checks passed (blocks 1-3: one third of every condition, blocks 4-9: two thirds)")


def check_unchanged(df, df_random):
    assert df[ID_COLS].equals(df_random[ID_COLS]), "order / blocks / sentences / roles / exposure changed"
    set_cols = [c for c in df_random.columns if c not in ID_COLS]
    key = lambda d: sorted(map(tuple, d[["sentence_type", "role"] + set_cols].astype(str).to_numpy().tolist()))
    assert key(df) == key(df_random), "the set of trials per sentence type x role changed"
    print("✓ Same order, blocks, sentences, roles and exposure as the unbalanced table; same trials per type x role")

check_phase_balance(df)
check_unchanged(df, df_random)
# ======================================================================================== SWITCH ↑



# %%
# ======================================================================================== NEW ↓
# EVENT TABLE: intended timing of all 30 events of every trial
#
# One row per event (216 trials × 30 events = 6480 rows), saved as tapping_events_<ID>_<GROUP>.csv.
# Jitter = PHASE SHIFT: the jittered event AND every later event move by the shift, so with two
#   jitters the shifts add up. `shift_ms` is therefore cumulative: pass the 30 values of a trial to
#   build_sequence(shifts_ms=...) unchanged.
# These are the INTENDED onsets. The stimulus step adds the ACTUAL onsets (onset_ms returned by
#   build_sequence) and checks actual - intended = 0 (± 1 sample).
#
# Columns: event (1–30), part (lead-in/S1/A/S2/B), pos_in_part (1–6), kind (noise/word),
#   grid_ms (unshifted onset), jitter_here (this event is a jittered one), jitter_step_ms (the shift
#   that starts at this event, 0 otherwise), shift_ms (cumulative shift of this event),
#   intended_onset_ms (= grid_ms + shift_ms; event 1 at 0 ms), ioi_before_ms (onset-to-onset from the
#   previous event).
PARTS     = ["lead-in", "S1", "A", "S2", "B"]          # 6 events each, in this order
PART_KIND = {"lead-in": "noise", "S1": "word", "A": "noise", "S2": "word", "B": "noise"}


def make_event_table(df):
    rows = []
    for t in df.sort_values("trial").itertuples(index=False):
        step = np.zeros(N_EVENTS)                       # shift that STARTS at each event
        for part in ("A", "S2", "B"):
            if getattr(t, f"{part}_jitter"):
                step[int(getattr(t, f"{part}_event")) - 1] += getattr(t, f"{part}_shift_ms")
        shift = np.cumsum(step)                         # phase shift: carried to every later event
        onset = np.arange(N_EVENTS) * IOI_MS + shift
        for e in range(N_EVENTS):
            part = PARTS[e // 6]
            rows.append({"participant": t.participant, "group": t.group, "trial": t.trial, "block": t.block,
                         "sentence_type": t.sentence_type, "sentence_id": t.sentence_id, "role": t.role,
                         "event": e + 1, "part": part, "pos_in_part": e % 6 + 1, "kind": PART_KIND[part],
                         "grid_ms": e * IOI_MS,
                         "jitter_here": bool(step[e] != 0), "jitter_step_ms": step[e],
                         "shift_ms": shift[e], "intended_onset_ms": onset[e],
                         "ioi_before_ms": onset[e] - onset[e - 1] if e > 0 else np.nan})
    return pd.DataFrame(rows)


def check_events(ev, df):
    n = len(df)
    assert len(ev) == n * N_EVENTS
    # every trial has events 1–30 in order (numpy check: works on every pandas version)   # CHANGED (bug fix)
    assert (ev.trial.to_numpy().reshape(n, N_EVENTS) == np.sort(df.trial.to_numpy())[:, None]).all()
    assert (ev.event.to_numpy().reshape(n, N_EVENTS) == np.arange(1, N_EVENTS + 1)).all()
    df = df.sort_values("trial")

    # expected shift steps, rebuilt independently from the design table
    exp = np.zeros((n, N_EVENTS))
    for i, t in enumerate(df.itertuples(index=False)):
        for part in ("A", "S2", "B"):
            if getattr(t, f"{part}_jitter"):
                exp[i, int(getattr(t, f"{part}_event")) - 1] += getattr(t, f"{part}_shift_ms")
    steps  = ev.jitter_step_ms.to_numpy().reshape(n, N_EVENTS)
    shifts = ev.shift_ms.to_numpy().reshape(n, N_EVENTS)
    onsets = ev.intended_onset_ms.to_numpy().reshape(n, N_EVENTS)
    assert np.allclose(steps, exp), "jitter steps differ from the design table"
    assert np.allclose(shifts, np.cumsum(exp, axis=1)), "shifts are not cumulative (phase shift)"
    assert (ev.groupby("trial").jitter_here.sum().to_numpy() == df.n_jitters.to_numpy()).all()

    # never moved: lead-in and S1 (events 1–12); first event of A, S2, B never a jitter point
    assert (ev.loc[ev.event <= 12, "shift_ms"] == 0).all(), "lead-in or S1 moved"
    assert not ev.loc[ev.pos_in_part == 1, "jitter_here"].any(), "first event of a part jittered"

    # phase shift: exactly ONE interval changes per jitter (by the shift), all others stay IOI_MS
    ioi = np.diff(onsets, axis=1)
    assert np.allclose(ioi, IOI_MS + exp[:, 1:]), "an interval other than the jittered one changed"

    # events never overlap
    min_gap = (ioi - EVENT_MS).min()
    assert min_gap > 0, f"events overlap (smallest gap {min_gap} ms)"
    print(f"✓ Event table checks passed ({len(ev)} events; smallest silent gap {min_gap:.0f} ms; "
          f"cumulative shift {shifts.min():+.0f} to {shifts.max():+.0f} ms; "
          f"last onset up to {onsets[:, -1].max():.0f} ms)")


events = make_event_table(df)
check_events(events, df)
# ======================================================================================== NEW ↑






# %%
# SUMMARY: critical trials per phase (rows) x size x direction (columns)
c = df[df.S2_critical]
print(pd.crosstab([c.sentence_type, c.S2_interval, c.phase], [c.S2_size_ms.astype(int), c.S2_direction]))



# %%
# SAVE (never overwrites: delete the old files first if you really want to regenerate an ID)
Path(OUTPUT_DIR).mkdir(exist_ok=True)
path = Path(OUTPUT_DIR) / f"tapping_{PARTICIPANT_ID}_{GROUP}.csv"
ev_path = Path(OUTPUT_DIR) / f"tapping_events_{PARTICIPANT_ID}_{GROUP}.csv"
for p in (path, ev_path):
    if p.exists():
        raise FileExistsError(f"{p} already exists (maybe from the old generator) - not overwriting. "
                              f"Delete it to regenerate {PARTICIPANT_ID}.")
df.to_csv(path, index=False)
print("Saved", path)
events.to_csv(ev_path, index=False)
print("Saved", ev_path)