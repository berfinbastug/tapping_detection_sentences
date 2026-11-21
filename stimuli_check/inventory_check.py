"""
check_00_inventory.py
Step 0: inventory of the item lists against the SWS audio.

- item counts and how many are still needed to reach TARGET_N
- duplicate IDs, identical items, empty cells
- every word has an SWS file in 3_sws/ (Lärm -> laerm_..., ließ -> liess_...)
- words that may not be monosyllabic (vowel-nucleus count != 1)
- same spelling with different capitalisation (typo or homograph sharing one file)
- SWS files that no item uses (replacement pool)

Re-runnable after every drop / addition. items_wl.csv is picked up once it exists.
Run (conda env `psychopy`):  python check_00_inventory.py
"""
import os
import re

import pandas as pd


BASE_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(BASE_DIR)


# ------------------------------------------------------------------ CONFIG
CSV_DIR =os.path.join(BASE_DIR, 'sentence_lists')
SWS_DIR  = os.path.join(BASE_DIR, 'sws_words', '3_sws')
OUT_DIR  = os.path.join(BASE_DIR, 'checks')
LISTS = {'2|4': 'items_2_4.csv', '3|3': 'items_3_3.csv', 'WL': 'items_wl.csv'}
TARGET_N = 18
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


# SWS files: base -> file name
sws_files = {}
if os.path.isdir(SWS_DIR):
    for f in sorted(os.listdir(SWS_DIR)):
        if f.lower().endswith('.wav') and not f.startswith('.'):
            sws_files[f[:-4].split('_sws')[0].lower()] = f
else:
    print(f'[warn] SWS folder not found: {SWS_DIR}')
print(f'SWS files found: {len(sws_files)}\n')

rows = []
for t, fn in LISTS.items():
    path = os.path.join(CSV_DIR, fn)
    if not os.path.exists(path):
        print(f'{t:4s}: {fn} not found (skipped)')
        continue
    df = pd.read_csv(path, dtype=str).fillna('')
    df.columns = [c.strip() for c in df.columns]
    slot_cols = [c for c in df.columns if c.lower().startswith('slot')]
    for c in slot_cols:
        df[c] = df[c].str.strip()

    print(f'{t:4s}: {len(df):2d} items, {len(slot_cols)} slots  ->  need {max(0, TARGET_N - len(df))} more for {TARGET_N}')
    if df['item'].duplicated().any():
        print(f'      [!] duplicate item IDs: {df.loc[df["item"].duplicated(), "item"].tolist()}')
    seqs = df[slot_cols].apply(lambda r: ' '.join(r).lower(), axis=1)
    if seqs.duplicated().any():
        print(f'      [!] identical items: {df.loc[seqs.duplicated(keep=False), "item"].tolist()}')

    for _, r in df.iterrows():
        for c in slot_cols:
            w = r[c]
            if w == '':
                print(f'      [!] empty cell: item {r["item"]}, {c}')
                continue
            b = base_name(w)
            rows.append(dict(type=t, item=r['item'], slot=c, word=w, base=b,
                             audio_found=b in sws_files, audio_file=sws_files.get(b, ''),
                             n_vowel_nuclei=len(VOWEL_NUCLEUS.findall(w))))

inv = pd.DataFrame(rows)
words = inv.drop_duplicates('word')

print(f'\nWord tokens: {len(inv)}   unique spellings: {inv.word.nunique()}   unique audio files needed: {inv.base.nunique()}')

missing = words.loc[~words.audio_found, 'word'].tolist()
print(f'\nAudio MISSING in 3_sws/: {missing if missing else "none"}')

multi = words.loc[words.n_vowel_nuclei != 1, 'word'].tolist()
print(f'Not exactly one vowel nucleus (check by ear): {multi if multi else "none"}')

caps = inv.groupby(inv.word.str.lower())['word'].unique()
caps = caps[caps.apply(len) > 1]
print('Same spelling, different capitalisation (check: typo, or homograph -> see file base):')
if len(caps):
    for b, forms in caps.items():
        where = inv[inv.word.str.lower() == b].groupby('word').apply(lambda g: [f'{x.type} it{x.item} {x.slot}' for x in g.itertuples()])
        for form, locs in where.items():
            print(f'   {form:8s} -> {base_name(form):8s} {locs}')
else:
    print('   none')

unused = sorted(b for b in sws_files if b not in set(inv.base))
print(f'\nSWS files not used by any item ({len(unused)}): {", ".join(unused) if unused else "none"}')

inv.to_csv(os.path.join(OUT_DIR, '00_inventory_long.csv'), index=False)
print(f'\nSaved: {os.path.join(OUT_DIR, "00_inventory_long.csv")}')
