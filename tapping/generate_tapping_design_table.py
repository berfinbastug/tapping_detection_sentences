# %%
# MODULES
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# after the imports (line 7, "from pathlib import Path"), add:
import os
PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(PROJECT_DIR)        # relative paths below are inside the project folder



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
SEED = None


JITTER_SIZES_MS    = [15, 30, 50]  # PLACEHOLDERS – replace with your real sizes
MIN_EVENTS_BETWEEN = 6             # event-index distance between the 2 jittered events (6 = 5 clean events between)
IOI_MS             = 450           # onset-to-onset
EVENT_MS           = 350           # event duration
ITI_S              = 2.5           # pause between trials (only for the duration estimate)
OUTPUT_DIR         = "tapping/design_tables"

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
                new.append(base(sid, role, "S2") | {"S2_interval": iv, "S2_combo": combo, "noise": noise})
            _place_noise([t for t in new if t["noise"]], N_B_NOISE.get(iv, 0), rng)
            trials += new

        # --- filler: 3 intervals × 6 size×direction, 1 each (6 of them carry noise)
        fcells = [(iv, c, c in fill_noise[iv]) for iv in S2_FILLER for c in COMBOS]           # 18
        new = []
        for sid, j in zip(sids, rng.permutation(len(fcells))):
            iv, c, noise = fcells[j]
            new.append(base(sid, "filler", "S2") | {"S2_interval": iv, "S2_combo": c, "noise": noise})
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
# BUILD A TABLE FOR A GIVEN PARTICIPANT
df = make_design(PARTICIPANT_ID, GROUP, SEED)
print(f"{len(df)} trials, {df.sentence_id.nunique()} sentences, seed = {df.seed.iloc[0]}")
print(f"tapping time ≈ {len(df) * (TRIAL_DURATION_S + ITI_S) / 60:.1f} min "
      f"({N_BLOCKS} blocks × {len(df) // N_BLOCKS} trials, {(len(df) // N_BLOCKS) * (TRIAL_DURATION_S + ITI_S) / 60:.1f} min per block)")
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
# SUMMARY
# print("Trials per sentence type × role")
# display(pd.crosstab(df.sentence_type, df.role, margins=True)[ROLES + ["All"]])
# print("\nJitter pattern × role (all sentence types)")
# display(pd.crosstab(df.jitter_pattern, df.role, margins=True)[ROLES + ["All"]])
# print("\nTrials per sentence type × number of jitters")
# display(pd.crosstab(df.sentence_type, df.n_jitters, margins=True))
# print("\nExample: the 4 trials of the first 3 sentences")
# display(df[df.sentence_id.isin(["S24_01", "S24_02", "S24_03"])]
#         .sort_values(["sentence_id", "trial"])
#         [["sentence_id", "trial", "block", "exposure", "role", "jitter_pattern",
#           "S2_interval", "S2_size_ms", "S2_direction", "A_interval", "B_interval", "clean_events_between"]])



# %%
# FIGURES
# from matplotlib.colors import ListedColormap
# from matplotlib.patches import Rectangle

# plt.rcParams.update({
#     "figure.dpi": 110, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
#     "axes.spines.top": False, "axes.spines.right": False,
#     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
#     "xtick.color": "#52514e", "ytick.color": "#52514e",
#     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
#     "axes.axisbelow": True, "legend.frameon": False,
# })
# ST_COLORS   = {"2|4": "#2a78d6", "3|3": "#eb6834", "word list": "#1baf7a"}
# ROLE_COLORS = {"critical_I2": "#0b0b0b", "critical_I3": "#6250d6", "filler": "#9cc2ef", "baseline": "#c3c2b7"}
# PART_COLORS = {"lead-in": "#d9d8d2", "S1": "#c9c8c0", "A": "#9cc2ef", "S2": "#f2b38f", "B": "#9cc2ef"}
# INK = "#0b0b0b"
# ROLE_SHORT = ["I2", "I3", "filler", "base"]


# def grouped_bars(ax, table, title, xlabel, colors=ST_COLORS, ylabel="Trials"):
#     n = len(table.columns); w = 0.8 / n
#     x = np.arange(len(table.index))
#     for i, c in enumerate(table.columns):
#         bars = ax.bar(x + (i - (n - 1) / 2) * w, table[c], w * 0.92,
#                       color=colors[c], label=c, edgecolor="white", linewidth=1)
#         ax.bar_label(bars, fontsize=8, color="#52514e", padding=1)
#     ax.set_xticks(x, table.index)
#     ax.set_title(title, loc="left"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
#     ax.set_ylim(0, max(table.values.max(), 1) * 1.35)


# def count_heatmap(ax, table, title, vmax=None, fmt=None):
#     vmax = vmax or max(table.values.max(), 1) * 1.3
#     ax.imshow(table.values.astype(float), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
#     for (i, j), v in np.ndenumerate(table.values):
#         ax.text(j, i, fmt(i, j, v) if fmt else int(v), ha="center", va="center", fontsize=9,
#                 color="white" if v > vmax * 0.6 else INK)
#     ax.set_xticks(range(table.shape[1]), table.columns, fontsize=8)
#     ax.set_yticks(range(table.shape[0]), table.index, fontsize=8)
#     ax.grid(False); ax.set_title(title, loc="left", fontsize=10)
#     for s in ax.spines.values(): s.set_visible(False)


# def size_dir_table(d, size_col="S2_size_ms", dir_col="S2_direction"):
#     t = pd.crosstab(d[size_col], d[dir_col]).reindex(index=JITTER_SIZES_MS, columns=DIRECTIONS, fill_value=0)
#     t.index = [f"{s} ms" for s in t.index]
#     return t



# # %%
# # HOW THE FACTORS MAP INTO CRITICAL CONTRASTS
# crit = df[df.S2_critical]
# t = pd.crosstab(crit.S2_interval, crit.sentence_type)[SENT_TYPES]
# status = crit.groupby(["S2_interval", "sentence_type"]).S2_boundary_status.first().unstack()[SENT_TYPES]
# SC = {"boundary": "#eb6834", "within-phrase": "#2a78d6", "no structure": "#1baf7a"}
# fig, ax = plt.subplots(figsize=(7.5, 2.6))
# for i, iv in enumerate(t.index):
#     for j, st in enumerate(SENT_TYPES):
#         s = status.loc[iv, st]
#         ax.add_patch(Rectangle((j, -i - 1), 0.96, 0.92, color=SC[s], alpha=0.85))
#         ax.text(j + 0.48, -i - 0.54, f"{s}\n{t.loc[iv, st]} trials", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
# ax.set_xlim(0, 3); ax.set_ylim(-2.05, 0.05)
# ax.set_xticks(np.arange(3) + 0.48, SENT_TYPES); ax.set_yticks([-0.54, -1.54], [f"interval {iv}" for iv in t.index])
# ax.grid(False); ax.tick_params(length=0)
# for s in ax.spines.values(): s.set_visible(False)
# ax.set_title("Critical S2 jitters: interval × sentence type → boundary status", loc="left")
# plt.tight_layout(); plt.show()
# print(crit.S2_boundary_status.value_counts().to_string())

# # %%
# # JITTER PATTERN WITHIN EACH ROLE
# porder = ["none", "S2", "A+S2", "S2+B", "A+B"]
# fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
# for ax, st in zip(axes, SENT_TYPES):
#     d = df[df.sentence_type == st]
#     t = pd.crosstab(d.jitter_pattern, d.role).reindex(index=porder, columns=ROLES, fill_value=0)
#     t.columns = ROLE_SHORT
#     count_heatmap(ax, t, st, vmax=22)
# fig.suptitle("Jitter pattern × role (trials)", x=0.02, ha="left", fontweight="bold")
# plt.tight_layout(); plt.show()

# # %%
# # JITTER COUNT PER SENTENCE TYPE
# t = pd.crosstab(df.n_jitters, df.sentence_type)[SENT_TYPES]
# t.index = [f"{i} jitter{'s' if i != 1 else ''}" for i in t.index]
# fig, ax = plt.subplots(figsize=(7, 3.2))
# grouped_bars(ax, t, "Trials by number of jitters", "")
# ax.legend(title="Sentence type", loc="upper right")
# plt.tight_layout(); plt.show()



# # %%
# # WHERE S3 JITTER FALLS
# s2 = df[df.S2_jitter]
# t = pd.crosstab(s2.S2_interval, s2.sentence_type).reindex(INTERVALS)[SENT_TYPES]
# fig, ax = plt.subplots(figsize=(7.5, 3.4))
# ax.axvspan(0.5, 2.5, color="#eef4fc", zorder=0)
# grouped_bars(ax, t, "S2 jitters by interval", "Interval within S2")
# ax.text(1.5, t.values.max() * 1.22, "critical", ha="center", color="#2a78d6", fontsize=9)
# ax.legend(title="Sentence type", loc="upper right")
# plt.tight_layout(); plt.show()



# # %%
# # SIZE X DIRECTION IN CRITICAL TRIALS
# fig, axes = plt.subplots(2, 3, figsize=(10, 4.6))
# for c, st in enumerate(SENT_TYPES):
#     d = crit[crit.sentence_type == st]
#     for r, iv in enumerate(CRITICAL):
#         sub = d[d.S2_interval == iv]
#         tot = size_dir_table(sub)
#         wn = size_dir_table(sub[sub.critical_with_noise]).reindex_like(tot).fillna(0)
#         count_heatmap(axes[r, c], tot, f"{st} · {iv}", vmax=5, fmt=lambda i, j, v, wn=wn: f"{int(v)} ({int(wn.values[i, j])})")
# fig.suptitle("Critical S2 jitters: size × direction — trials (with noise)", x=0.02, ha="left", fontweight="bold")
# plt.tight_layout(); plt.show()



# # %%
# # FILLER TRIALS
# # INTERVAL X SIZE X DIRECTION
# fill = df[(df.role == "filler")].copy()
# fill["level"] = fill.S2_size_ms.astype(int).astype(str) + np.where(fill.S2_direction == "late", "↑", "↓")
# levels = [f"{s}{a}" for s in JITTER_SIZES_MS for a in ("↓", "↑")]
# fig, axes = plt.subplots(1, 3, figsize=(11, 2.6))
# for ax, st in zip(axes, SENT_TYPES):
#     d = fill[fill.sentence_type == st]
#     t = pd.crosstab(d.S2_interval, d.level).reindex(index=S2_FILLER, columns=levels, fill_value=0)
#     nz = {(r.S2_interval, r.level): ("+A" if r.jitter_pattern == "A+S2" else "+B") for r in d[d.S2_with_noise].itertuples()}
#     count_heatmap(ax, t, st, vmax=2.5,
#                   fmt=lambda i, j, v, t=t, nz=nz: f"{int(v)} {nz.get((t.index[i], t.columns[j]), '')}".strip())
# fig.suptitle("Filler S2 jitters: interval × size/direction (↓ early, ↑ late)", x=0.02, ha="left", fontweight="bold")
# plt.tight_layout(); plt.show()



# # %%
# # NOISE JITTERS
# fig = plt.figure(figsize=(13, 3.4))
# gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1.6, 1.0])
# for c, st in enumerate(SENT_TYPES):
#     ax = fig.add_subplot(gs[0, c])
#     count_heatmap(ax, size_dir_table(_noise(df[df.sentence_type == st]), "size", "dir"), f"{st} · noise", vmax=5)
# for k, part, ivs in ((3, "A", A_INTERVALS), (4, "B", B_INTERVALS)):
#     ax = fig.add_subplot(gs[0, k])
#     d = df[df[f"{part}_jitter"]]
#     src = np.select([d.role == "baseline", d.role == "filler"], ["A+B (baseline)", "filler + noise"], "critical + noise")
#     t = pd.crosstab(d[f"{part}_interval"], src).reindex(ivs, fill_value=0)
#     SRC_C = {"critical + noise": "#6250d6", "filler + noise": "#9cc2ef", "A+B (baseline)": "#b9b8b2"}
#     t = t[[c for c in SRC_C if c in t]]
#     t.plot.bar(stacked=True, ax=ax, color=[SRC_C[c] for c in t], rot=0, width=0.7)
#     ax.set_title(f"{part} jitters by interval", loc="left"); ax.set_xlabel(f"Interval within {part}"); ax.set_ylabel("Trials")
#     ax.legend(fontsize=7)
# plt.tight_layout(); plt.show()



# # %%
# # SPREAD OVER BLOCKS
# fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
# t = pd.crosstab(df.block, df.sentence_type)[SENT_TYPES]; t.index = [f"B{b}" for b in t.index]
# grouped_bars(axes[0], t, "Trials per block by sentence type", "Block")
# axes[0].set_ylim(0, 12); axes[0].legend(ncols=3, loc="upper center", fontsize=7)
# t = pd.crosstab(df.block, df.role)[ROLES]; t.index = [f"B{b}" for b in t.index]
# grouped_bars(axes[1], t, "Trials per block by role", "Block", colors=ROLE_COLORS)
# axes[1].set_ylim(0, 10); axes[1].legend(ncols=4, loc="upper center", fontsize=7)
# t = pd.crosstab(df.block, df.n_jitters); t.index = [f"B{b}" for b in t.index]; t.columns = [f"{c} jit" for c in t.columns]
# grouped_bars(axes[2], t, "Trials per block by no. of jitters", "Block",
#              colors={"0 jit": "#d9d8d2", "1 jit": "#9cc2ef", "2 jit": "#1f5ea8"})
# axes[2].set_ylim(0, 22); axes[2].legend(ncols=3, loc="upper center", fontsize=7)
# plt.tight_layout(); plt.show()
# print("Jitter count per block is NOT fixed (only roles are) – it varies a little because the noise jitters are spread randomly.")



# %%
Path(OUTPUT_DIR).mkdir(exist_ok=True)
path = Path(OUTPUT_DIR) / f"tapping_{PARTICIPANT_ID}_{GROUP}.csv"
df.to_csv(path, index=False)
print("Saved", path)
# ======================================================================================== NEW ↓
ev_path = Path(OUTPUT_DIR) / f"tapping_events_{PARTICIPANT_ID}_{GROUP}.csv"
events.to_csv(ev_path, index=False)
print("Saved", ev_path)
# ======================================================================================== NEW ↑
# %%