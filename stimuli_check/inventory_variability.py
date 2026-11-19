"""
check_01_inventory_variability.py
Step 0 + Step 1: inventory of the item lists against the SWS audio, and a
variability audit per type x slot.

Re-runnable: edit the CSVs (drop / add items) and run again. Nothing is
hard-coded to 29 / 31 items. A word-list CSV is picked up automatically
once it exists.

Run (conda env `psychopy`) from anywhere:
    python check_01_inventory_variability.py
Outputs: printed report + CSVs in OUT_DIR.
"""
import os
import re
import math
import itertools
from collections import Counter

import pandas as pd

# ------------------------------------------------------------------ CONFIG
BASE_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(BASE_DIR)
SWS_DIR  = os.path.join(BASE_DIR, 'sws_words', '3_sws')
OUT_DIR  = os.path.join(BASE_DIR, 'checks')
CSV_DIR =os.path.join(BASE_DIR, 'sentence_lists')

LISTS = {                      # type label -> CSV file (missing files are skipped)
    '2|4': 'items_2_4.csv',
    '3|3': 'items_3_3.csv',
    'WL':  'items_wl.csv',
}
TARGET_N = 35

# Slot classes are read from the column suffix (slot3_V -> 'V').
CLOSED_CLASS = {'P', 'Det', 'Part'}            # small German word classes, cannot be unique
OPEN_CLASS   = {'N', 'V', 'Adj', 'Adv', 'W'}   # 'W' = generic word-list slot

# Flag thresholds
MAX_SHARE_CLOSED   = 0.25   # no closed-class word in > 25 % of items in a slot
MAX_COUNT_OPEN     = 2      # open-class word max 2x in the same slot
MAX_COUNT_OPEN_ALL = 3      # open-class word max 3x anywhere in one type
NEAR_DUP_SHARED    = 2      # items sharing >= this many open-class/particle words in the same slots
# -------------------------------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
VOWEL_NUCLEUS = re.compile(r'[aeiouäöüy]+', re.IGNORECASE)


WORDS_CSV = os.path.join(BASE_DIR, 'sws_words', 'words.csv')   # pipeline's word -> file-base map
_WORD2BASE = {}
if os.path.exists(WORDS_CSV):
    _w = pd.read_csv(WORDS_CSV, dtype=str)
    _WORD2BASE = dict(zip(_w['word'].str.strip(), _w['base'].str.strip()))


def base_name(word):
    """Word -> file base. Uses sws_words/words.csv first (case-sensitive, so the particle
    'weg' -> weg_2 and 'teil' -> teil_2); falls back to Lärm -> laerm, ließ -> liess."""
    w = word.strip()
    if w in _WORD2BASE:
        return _WORD2BASE[w]
    w = w.lower()
    for a, b in (('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')):
        w = w.replace(a, b)
    return w


def slot_class(col):
    return col.split('_', 1)[1] if '_' in col else 'W'


def n_nuclei(word):
    return len(VOWEL_NUCLEUS.findall(word))


def norm_entropy(counts):
    n = sum(counts)
    k = len(counts)
    if k <= 1:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in counts)
    return h / math.log(k)


def hr(title):
    print('\n' + '=' * 78 + f'\n{title}\n' + '=' * 78)


# ---------------------------------------------------------------- load
lists = {}
for t, fn in LISTS.items():
    path = os.path.join(CSV_DIR, fn)
    if not os.path.exists(path):
        print(f'[skip] {t}: {fn} not found')
        continue
    df = pd.read_csv(path, dtype=str).fillna('')
    df.columns = [c.strip() for c in df.columns]
    slot_cols = [c for c in df.columns if c.lower().startswith('slot')]
    for c in slot_cols:
        df[c] = df[c].str.strip()
    lists[t] = (df, slot_cols)

sws_files = {}                       # base -> file name  (e.g. 'laerm' -> 'laerm_sws50ms_dur350ms.wav')
if os.path.isdir(SWS_DIR):
    for f in sorted(os.listdir(SWS_DIR)):
        if f.lower().endswith('.wav') and not f.startswith('.'):
            sws_files[f[:-4].split('_sws')[0].lower()] = f
else:
    print(f'[warn] SWS folder not found: {SWS_DIR}')

# ---------------------------------------------------------------- 0. inventory
hr('0. INVENTORY')
inv_rows = []
for t, (df, slot_cols) in lists.items():
    n = len(df)
    print(f'{t:4s}: {n:2d} items  ->  need {max(0, TARGET_N - n)} more to reach {TARGET_N}')
    if df['item'].duplicated().any():
        print(f'   [!] duplicate item IDs: {df.loc[df["item"].duplicated(), "item"].tolist()}')
    seqs = df[slot_cols].apply(lambda r: ' '.join(r).lower(), axis=1)
    if seqs.duplicated().any():
        print(f'   [!] identical items: {df.loc[seqs.duplicated(keep=False), "item"].tolist()}')
    for _, r in df.iterrows():
        for c in slot_cols:
            w = r[c]
            if w == '':
                print(f'   [!] empty cell: item {r["item"]}, {c}')
                continue
            b = base_name(w)
            inv_rows.append(dict(type=t, item=r['item'], slot=c, slot_class=slot_class(c), word=w,
                                 base=b, audio_found=b in sws_files, audio_file=sws_files.get(b, ''),
                                 n_vowel_nuclei=n_nuclei(w)))
inv = pd.DataFrame(inv_rows)

words = inv.drop_duplicates('word')
missing = words.loc[~words.audio_found, 'word'].tolist()
multi = words.loc[words.n_vowel_nuclei != 1, 'word'].tolist()
print(f'\nUnique word forms: {inv.word.nunique()}  (case-insensitive: {inv.word.str.lower().nunique()})')
print(f'Audio missing in 3_sws/ : {missing if missing else "none"}')
print(f'Not exactly one vowel nucleus (check syllable count by ear): {multi if multi else "none"}')

caps = inv.groupby(inv.word.str.lower())['word'].unique()
caps = caps[caps.apply(len) > 1]
if len(caps):
    print('Same spelling, different capitalisation (typo, or homograph like Weg/weg sharing ONE audio file):')
    for lw, forms in caps.items():
        print(f'   {lw}: {list(forms)}')

# words in the audio folder that no item uses (useful pool for replacements)
used = set(inv.base)
unused = sorted(k for k in sws_files if k not in used)
print(f'\nSWS words not used in any item: {len(unused)} (list saved to 01_unused_sws_words.csv; replacement pool)')
pd.DataFrame({'base': unused, 'file': [sws_files[w] for w in unused]}).to_csv(os.path.join(OUT_DIR, '01_unused_sws_words.csv'), index=False)

inv.to_csv(os.path.join(OUT_DIR, '01_inventory_long.csv'), index=False)

# ---------------------------------------------------------------- 1. variability per slot
hr('1. VARIABILITY PER TYPE x SLOT')
slot_rows, flag_lines = [], []
for t, (df, slot_cols) in lists.items():
    print(f'\n--- {t}  (n = {len(df)}) ---')
    print(f'{"slot":12s} {"class":5s} {"unique":>6s} {"top word (share)":24s} {"evenness":>8s}  flags')
    for c in slot_cols:
        cls = slot_class(c)
        cnt = Counter(df[c].str.lower())
        top, top_n = cnt.most_common(1)[0]
        share = top_n / len(df)
        even = norm_entropy(list(cnt.values()))
        flags = []
        if cls in CLOSED_CLASS and share > MAX_SHARE_CLOSED:
            flags.append(f'top share {share:.0%} > {MAX_SHARE_CLOSED:.0%}')
        if cls not in CLOSED_CLASS:
            over = {w: k for w, k in cnt.items() if k > MAX_COUNT_OPEN}
            if over:
                flags.append(f'>{MAX_COUNT_OPEN}x: {over}')
        print(f'{c:12s} {cls:5s} {len(cnt):6d} {f"{top} ({share:.0%})":24s} {even:8.2f}  {"; ".join(flags)}')
        slot_rows.append(dict(type=t, slot=c, slot_class=cls, n_items=len(df), n_unique=len(cnt),
                              top_word=top, top_share=round(share, 3), evenness=round(even, 3),
                              distribution='; '.join(f'{w}:{k}' for w, k in cnt.most_common()),
                              flags='; '.join(flags)))
    # full distributions for closed-class slots
    for c in slot_cols:
        if slot_class(c) in CLOSED_CLASS:
            cnt = Counter(df[c].str.lower())
            print(f'   {c}: ' + ', '.join(f'{w} {k}' for w, k in cnt.most_common()))
pd.DataFrame(slot_rows).to_csv(os.path.join(OUT_DIR, '01_variability_by_slot.csv'), index=False)

# ---------------------------------------------------------------- 1b. repetition across the whole list
hr('1b. OPEN-CLASS WORDS USED > %d TIMES WITHIN A TYPE (any slot)' % MAX_COUNT_OPEN_ALL)
rep_rows = []
for t, (df, slot_cols) in lists.items():
    sub = inv[(inv.type == t) & (~inv.slot_class.isin(CLOSED_CLASS))]
    cnt = Counter(sub.word.str.lower())
    over = [(w, k) for w, k in cnt.most_common() if k > MAX_COUNT_OPEN_ALL]
    print(f'{t}: ' + (', '.join(f'{w} {k}x' for w, k in over) if over else 'none'))
    rep_rows += [dict(type=t, word=w, count=k) for w, k in cnt.most_common()]
pd.DataFrame(rep_rows).to_csv(os.path.join(OUT_DIR, '01_open_class_counts.csv'), index=False)

hr('1c. WORDS SHARED BETWEEN TYPES (info only; useful for word lists)')
type_words = {t: set(inv[inv.type == t].word.str.lower()) for t in lists}
for a, b in itertools.combinations(type_words, 2):
    shared = sorted(type_words[a] & type_words[b])
    print(f'{a} & {b}: {len(shared)} shared -> {", ".join(shared)}')

# ---------------------------------------------------------------- 1d. repeated word pairs
hr('1d. REPEATED WORD PAIRS (same slots; adjacent pairs, or two open-class/particle words)')
pair_rows = []
for t, (df, slot_cols) in lists.items():
    for ci, cj in itertools.combinations(slot_cols, 2):
        adjacent = slot_cols.index(cj) - slot_cols.index(ci) == 1
        both_content = slot_class(ci) not in {'P', 'Det'} and slot_class(cj) not in {'P', 'Det'}
        if not (adjacent or both_content):
            continue
        if slot_class(ci) in {'P', 'Det'} and slot_class(cj) in {'P', 'Det'}:
            continue
        pairs = df[ci].str.lower() + ' ... ' + df[cj].str.lower()
        for p, k in Counter(pairs).items():
            if k > 1:
                items = df.loc[pairs == p, 'item'].tolist()
                pair_rows.append(dict(type=t, slots=f'{ci}+{cj}', pair=p, count=k, items=items))
pairs_df = pd.DataFrame(pair_rows)
if len(pairs_df):
    pairs_df = pairs_df.sort_values(['type', 'count'], ascending=[True, False])
    for t, g in pairs_df.groupby('type', sort=False):
        print(f'\n--- {t} ---')
        for _, r in g.iterrows():
            print(f'   {r["count"]}x  {r["pair"]:28s} [{r["slots"]}]  items {r["items"]}')
else:
    print('none')
pairs_df.to_csv(os.path.join(OUT_DIR, '01_repeated_pairs.csv'), index=False)

# ---------------------------------------------------------------- 1e. near-duplicate items
hr(f'1e. NEAR-DUPLICATE ITEMS (>= {NEAR_DUP_SHARED} open-class/particle words in the same slots)')
dup_rows = []
for t, (df, slot_cols) in lists.items():
    content_cols = [c for c in slot_cols if slot_class(c) not in {'P', 'Det'}]
    recs = df[['item'] + content_cols].to_dict('records')
    for a, b in itertools.combinations(recs, 2):
        shared = [a[c] for c in content_cols if a[c].lower() == b[c].lower()]
        if len(shared) >= NEAR_DUP_SHARED:
            sa = ' '.join(df.loc[df.item == a['item'], slot_cols].values[0])
            sb = ' '.join(df.loc[df.item == b['item'], slot_cols].values[0])
            dup_rows.append(dict(type=t, item_a=a['item'], item_b=b['item'], shared=shared,
                                 text_a=sa, text_b=sb))
            print(f'{t}: {a["item"]:>3s} "{sa}"  ~  {b["item"]:>3s} "{sb}"   shared {shared}')
if not dup_rows:
    print('none')
pd.DataFrame(dup_rows).to_csv(os.path.join(OUT_DIR, '01_near_duplicates.csv'), index=False)

print(f'\nCSV outputs written to: {OUT_DIR}')