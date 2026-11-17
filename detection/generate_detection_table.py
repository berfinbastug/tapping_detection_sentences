# %%
# REQUIREMENTS
import zlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import os                                                        # (if not already imported)
PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(PROJECT_DIR)
# %%
# SETTINGS
VALID_GROUPS = ["speech", "acoustic"]   # "speech" (informed SWS) or "acoustic" (naive SWS)

def ask_settings():
    # participant ID: must not be empty; upper-cased so "p01" and "P01" give the same seed
    pid = input("Enter participant ID: ").strip().upper()
    while not pid:
        pid = input("Participant ID cannot be empty. Enter participant ID: ").strip().upper()

    # group: keep asking until it is one of the valid groups
    group = input(f"Enter experimental group ({' / '.join(VALID_GROUPS)}): ").strip().lower()
    while group not in VALID_GROUPS:
        group = input(f'"{group}" is not valid. Please type {" or ".join(VALID_GROUPS)}: ').strip().lower()
    return pid, group

PARTICIPANT_ID, GROUP = ask_settings()
SEED           = None              # None -> derived from PARTICIPANT_ID (reproducible)


JITTER_SIZES_MS = [15, 30, 50]     # PLACEHOLDERS – must match the tapping experiment
IOI_MS          = 450
EVENT_MS        = 350
RESPONSE_S      = 1.5              # assumed mean response time (duration estimate only)
ITI_S           = 0.8
OUTPUT_DIR      = "design_tables"

# %%
# further parameters
N_WORDS        = 6 # n words in a sentence
SOUND_S        = ((N_WORDS - 1) * IOI_MS + EVENT_MS) / 1000          # 2.6 s
TRIAL_S        = SOUND_S + RESPONSE_S + ITI_S

SENT_TYPES  = ["2|4", "3|3", "word list"]
TYPE_CODE   = {"2|4": "S24", "3|3": "S33", "word list": "WL"}
N_SENTENCES = 18
N_BLOCKS    = 3
INTERVALS   = ["I1", "I2", "I3", "I4", "I5"]
CRITICAL    = ["I2", "I3"]
S2_FILLER   = ["I1", "I4", "I5"]
DIRECTIONS  = ["early", "late"]
COMBOS      = [(s, d) for s in JITTER_SIZES_MS for d in DIRECTIONS]      # 6 size × direction levels
ROLES       = ["critical_I2", "critical_I3", "filler", "catch"]
N_FILLER    = 6                    # per sentence type (the other 12 third-slot trials are catch)
N_CATCH     = N_SENTENCES - N_FILLER

def moved_word(interval):
    # 1-based word that moves for a jitter at this interval
    return int(interval[1:]) + 1
# %%
# DESIGN GENERATOR
def make_design(participant_id, group, seed=None):
    if seed is None:
        seed = zlib.crc32(participant_id.encode())
    rng = np.random.default_rng(seed)
    trials = []

    for st in SENT_TYPES:
        sids = [f"{TYPE_CODE[st]}_{i:02d}" for i in range(1, N_SENTENCES + 1)]
        perm = rng.permutation(N_SENTENCES)
        groups = [[sids[j] for j in perm[g * 6:(g + 1) * 6]] for g in range(3)]

        # filler cards: each size×direction once, each filler interval 1 early + 1 late
        early, late = rng.permutation(JITTER_SIZES_MS), rng.permutation(JITTER_SIZES_MS)
        fcards = [(iv, (int(early[i]), "early")) for i, iv in enumerate(S2_FILLER)] + \
                 [(iv, (int(late[i]), "late")) for i, iv in enumerate(S2_FILLER)]
        fcards = [fcards[j] for j in rng.permutation(len(fcards))]          # 2 per block after slicing

        for g, grp in enumerate(groups):
            # critical I2 in block g, I3 in block g+1: each block gets every size×direction once
            for role, iv, b in (("critical_I2", "I2", g), ("critical_I3", "I3", (g + 1) % 3)):
                for sid, j in zip(grp, rng.permutation(len(COMBOS))):
                    trials.append(dict(sentence_type=st, sentence_id=sid, role=role, block=b + 1,
                                       interval=iv, combo=COMBOS[j]))
            # third slot in block g+2: 2 filler + 4 catch
            b = (g + 2) % 3
            third = [("filler", fcards[2 * b]), ("filler", fcards[2 * b + 1])] + [("catch", None)] * 4
            for sid, j in zip(grp, rng.permutation(len(third))):
                role, card = third[j]
                t = dict(sentence_type=st, sentence_id=sid, role=role, block=b + 1)
                if card:
                    t |= dict(interval=card[0], combo=card[1])
                trials.append(t)

    # order: shuffle within blocks; no sentence in two consecutive trials
    order = []
    for b in range(1, N_BLOCKS + 1):
        blk = [t for t in trials if t["block"] == b]
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
        seen[t["sentence_id"]] = seen.get(t["sentence_id"], 0) + 1
        on = t["role"] != "catch"
        size, direction = t["combo"] if on else (np.nan, None)
        out.append({"participant": participant_id, "group": group, "seed": seed,
                    "trial": i, "block": t["block"],
                    "sentence_type": t["sentence_type"], "sentence_id": t["sentence_id"],
                    "exposure": seen[t["sentence_id"]], "role": t["role"],
                    "jittered": on, "interval": t.get("interval"),
                    "moved_word": moved_word(t["interval"]) if on else np.nan,
                    "size_ms": size, "direction": direction,
                    "shift_ms": (size if direction == "late" else -size) if on else 0,
                    "correct_response": "irregular" if on else "regular"})
    df = pd.DataFrame(out)
    crit = df.role.str.startswith("critical")
    df["boundary_status"] = np.select(
        [~crit, df.sentence_type == "word list",
         ((df.sentence_type == "2|4") & (df.interval == "I2")) | ((df.sentence_type == "3|3") & (df.interval == "I3"))],
        ["(not critical)", "no structure", "boundary"], default="within-phrase")
    return df
# %%
# BUILD A TABLE FOR A PARTICIPANT
df = make_design(PARTICIPANT_ID, GROUP, SEED)
print(f"{len(df)} trials, {df.sentence_id.nunique()} sentences, seed = {df.seed.iloc[0]}")
print(f"trial ≈ {TRIAL_S:.1f} s -> task ≈ {len(df) * TRIAL_S / 60:.1f} min "
      f"({N_BLOCKS} blocks × {len(df) // N_BLOCKS} trials, {len(df) // N_BLOCKS * TRIAL_S / 60:.1f} min per block)")
df.head(12)
# %%
# BALANCE CHECKS
def check_balance(df):
    assert len(df) == 162 and df.trial.is_unique
    for st in SENT_TYPES:
        d = df[df.sentence_type == st]
        assert len(d) == 54, st
        assert d.role.value_counts().to_dict() == {"critical_I2": 18, "critical_I3": 18, "catch": 12, "filler": 6}, st
        crit = d[d.role.str.startswith("critical")]
        g = crit.groupby(["interval", "size_ms", "direction"]).size()
        assert len(g) == 12 and (g == 3).all(), st                                     # 3 reps per cell
        f = d[d.role == "filler"]
        assert (f.groupby(["size_ms", "direction"]).size() == 1).all() and len(f) == 6, st   # each level once
        assert (f.groupby(["interval", "direction"]).size() == 1).all(), st                 # each interval 1 early + 1 late
        # blocks
        for b in range(1, N_BLOCKS + 1):
            db = d[d.block == b]
            assert db.role.value_counts().to_dict() == {"critical_I2": 6, "critical_I3": 6, "catch": 4, "filler": 2}, (st, b)
            cb = db[db.role.str.startswith("critical")].groupby(["interval", "size_ms", "direction"]).size()
            assert len(cb) == 12 and (cb == 1).all(), (st, b)                          # every level once per interval per block
    # sentences: 3 trials = I2 + I3 + (filler or catch), one per block
    g = df.groupby("sentence_id")
    assert (g.size() == 3).all() and (g.block.nunique() == 3).all()
    assert (g.role.apply(lambda r: set(r) - {"critical_I2", "critical_I3"} <= {"filler", "catch"}
                         and {"critical_I2", "critical_I3"} <= set(r) and len(set(r)) == 3)).all()
    assert (df.sentence_id.values[1:] != df.sentence_id.values[:-1]).all(), "sentence repeated back-to-back"
    print("✓ All balance checks passed")

check_balance(df)

# %%
# SUMMARY TABLE
# print("Trials per sentence type × role")
# display(pd.crosstab(df.sentence_type, df.role, margins=True)[ROLES + ["All"]])
# print("\nCritical trials: boundary status")
# display(df[df.role.str.startswith("critical")].boundary_status.value_counts().to_frame("trials"))
# print("\nExample: the 3 trials of the first 4 sentences")
# display(df[df.sentence_id.isin(["S24_01", "S24_02", "S24_03", "S24_04"])]
#         .sort_values(["sentence_id", "trial"])
#         [["sentence_id", "trial", "block", "exposure", "role", "interval", "moved_word", "size_ms", "direction", "correct_response"]])


# %% FIGURES
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
# ROLE_COLORS = {"critical_I2": "#0b0b0b", "critical_I3": "#6250d6", "filler": "#9cc2ef", "catch": "#c3c2b7"}
# ROLE_SHORT  = {"critical_I2": "I2", "critical_I3": "I3", "filler": "filler", "catch": "catch"}
# INK = "#0b0b0b"
# LEVELS = [f"{s}{a}" for s in JITTER_SIZES_MS for a in ("↓", "↑")]
# df["level"] = np.where(df.jittered, df.size_ms.fillna(0).astype(int).astype(str) + np.where(df.direction == "late", "↑", "↓"), "–")


# def grouped_bars(ax, table, title, xlabel, colors=ST_COLORS, ylabel="Trials"):
#     n = len(table.columns); w = 0.8 / n
#     x = np.arange(len(table.index))
#     for i, c in enumerate(table.columns):
#         bars = ax.bar(x + (i - (n - 1) / 2) * w, table[c], w * 0.92, color=colors[c], label=c, edgecolor="white", linewidth=1)
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


# # %%
# # critical interval x sentence type
# crit = df[df.role.str.startswith("critical")]
# t = pd.crosstab(crit.interval, crit.sentence_type)[SENT_TYPES]
# status = crit.groupby(["interval", "sentence_type"]).boundary_status.first().unstack()[SENT_TYPES]
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
# ax.set_title("Critical shifts: interval × sentence type → boundary status", loc="left")
# plt.tight_layout(); plt.show()


# # %%
# # critical trials size and direction
# fig, axes = plt.subplots(2, 3, figsize=(10, 4.4))
# for c, st in enumerate(SENT_TYPES):
#     for r, iv in enumerate(CRITICAL):
#         sub = crit[(crit.sentence_type == st) & (crit.interval == iv)]
#         t = pd.crosstab(sub.size_ms.astype(int), sub.direction).reindex(index=JITTER_SIZES_MS, columns=DIRECTIONS, fill_value=0)
#         t.index = [f"{s} ms" for s in t.index]
#         count_heatmap(axes[r, c], t, f"{st} · {iv}", vmax=5)
# fig.suptitle("Critical shifts: size × direction (trials)", x=0.02, ha="left", fontweight="bold")
# plt.tight_layout(); plt.show()


# # %%
# # filler trials, interval x size x direction
# fill = df[df.role == "filler"]
# fig, axes = plt.subplots(1, 3, figsize=(11, 2.4))
# for ax, st in zip(axes, SENT_TYPES):
#     d = fill[fill.sentence_type == st]
#     t = pd.crosstab(d.interval, d.level).reindex(index=S2_FILLER, columns=LEVELS, fill_value=0)
#     count_heatmap(ax, t, st, vmax=2)
# fig.suptitle("Filler shifts: interval × size/direction (↓ early, ↑ late)", x=0.02, ha="left", fontweight="bold")
# plt.tight_layout(); plt.show()

# # %%
# # trial classes and shift positions
# fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
# t = pd.crosstab(df.role, df.sentence_type).reindex(ROLES)[SENT_TYPES]; t.index = [ROLE_SHORT[r] for r in t.index]
# grouped_bars(axes[0], t, "Trials per role", "Role"); axes[0].legend(ncols=3, fontsize=8, loc="upper right")
# s = df[df.jittered]
# t = pd.crosstab(s.interval, s.sentence_type).reindex(INTERVALS)[SENT_TYPES]
# axes[1].axvspan(0.5, 2.5, color="#eef4fc", zorder=0)
# grouped_bars(axes[1], t, "Shifts by interval", "Interval"); axes[1].legend(ncols=3, fontsize=8, loc="upper right")
# plt.tight_layout(); plt.show()
# print(f"Shifted trials: {df.jittered.sum()}  (critical {crit.shape[0]} = {crit.shape[0] / df.jittered.sum():.0%}, "
#       f"filler {len(fill)} = {len(fill) / df.jittered.sum():.0%});  catch {(~df.jittered).sum()} = {(~df.jittered).mean():.0%} of all trials")



# # %%
# # blocks
# fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
# t = pd.crosstab(df.block, df.role)[ROLES]; t.index = [f"B{b}" for b in t.index]; t.columns = [ROLE_SHORT[c] for c in t.columns]
# grouped_bars(axes[0], t, "Trials per block by role (all types)", "Block", colors={ROLE_SHORT[k]: v for k, v in ROLE_COLORS.items()})
# axes[0].legend(ncols=4, fontsize=8, loc="upper center")
# cb = crit.assign(cell=crit.interval + " " + crit.level)
# t = pd.crosstab(cb.block, cb.cell).reindex(columns=[f"{iv} {l}" for iv in CRITICAL for l in LEVELS]); t.index = [f"B{b}" for b in t.index]
# count_heatmap(axes[1], t, "Critical trials per block × cell (all types: 3 = once per type)", vmax=5)
# axes[1].tick_params(axis="x", rotation=60)
# plt.tight_layout(); plt.show()


# # %%
# # exposure and sequence
# fig = plt.figure(figsize=(14, 3.2))
# gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2])
# ax = fig.add_subplot(gs[0, 0])
# t = pd.crosstab(df.exposure, df.role)[ROLES]; t.index = [f"{e}." for e in t.index]; t.columns = [ROLE_SHORT[c] for c in t.columns]
# grouped_bars(ax, t, "Role × exposure", "Exposure in detection", colors={ROLE_SHORT[k]: v for k, v in ROLE_COLORS.items()})
# ax.legend(ncols=2, fontsize=7)
# sub = gs[0, 1].subgridspec(3, 1)
# rows = [("Sentence type", df.sentence_type.map({s: i for i, s in enumerate(SENT_TYPES)}), [ST_COLORS[s] for s in SENT_TYPES], SENT_TYPES),
#         ("Role", df.role.map({r: i for i, r in enumerate(ROLES)}), [ROLE_COLORS[r] for r in ROLES], [ROLE_SHORT[r] for r in ROLES]),
#         ("Boundary", df.boundary_status.map({s: i for i, s in enumerate(["(not critical)", "boundary", "within-phrase", "no structure"])}),
#          ["#f1f0ec", "#eb6834", "#2a78d6", "#1baf7a"], ["–", "boundary", "within", "no struct."])]
# bounds = np.flatnonzero(np.diff(df.block.values)) + 1
# for k, (lab, vals, cols, names) in enumerate(rows):
#     a = fig.add_subplot(sub[k])
#     a.imshow(vals.values[None, :], aspect="auto", cmap=ListedColormap(cols), vmin=0, vmax=len(names) - 1, interpolation="nearest")
#     a.set_yticks([0], [lab]); a.set_xticks([]); a.grid(False)
#     for s in a.spines.values(): s.set_visible(False)
#     for b in bounds: a.axvline(b - 0.5, color="#52514e", ls="--", lw=1)
#     a.legend([Rectangle((0, 0), 1, 1, color=c) for c in cols], names, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=7, ncols=len(names))
# plt.tight_layout(); plt.show()


# %%
Path(OUTPUT_DIR).mkdir(exist_ok=True)
path = Path(OUTPUT_DIR) / f"detection_{PARTICIPANT_ID}_{GROUP}.csv"
df.drop(columns="level", errors="ignore").to_csv(path, index=False)
print("Saved", path)
# %%
