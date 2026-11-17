# %%
import os
import re
import sys
import glob
import unicodedata

import pandas as pd
from IPython.display import Audio, display

PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'   # the project folder (repo/, sentence_lists/, sws_words/)
os.chdir(PROJECT_DIR)        # every relative path below ('sws_words/...', 'sentence_lists') is inside the project
                             # folder, wherever this script is started from (it lives in stimuli_generator_codes/)
REPO_DIR = PROJECT_DIR       # folder that CONTAINS the repo package

if REPO_DIR is not None:
    sys.path.insert(0, REPO_DIR)
from repo import synthesize_word



# %%
# SETTINGS
SENTENCE_DIR = 'sentence_lists'      # folder with the item CSV files
OUT_DIR      = 'sws_words/1_tts'     # where the TTS words go
TABLE_DIR    = 'sws_words'           # where words.csv and items.csv go


LOWER_ITEM_INITIAL = True   # 'Nach' (sentence-initial) -> 'nach', so it is the same word as 'nach' elsewhere.
                            # All other words keep their case: noun 'Weg' and adverb 'weg' stay two words.

TTS_LANG    = 'de'
TTS_TLD     = 'de'
TTS_SLOW    = False
TTS_SR      = 44100
TTS_RETRIES = 3
TTS_PAUSE_S = 1.0           # pause between requests (avoids rate limiting)
TTS_TEXT    = {}            # what gTTS is given when it should differ from the spelling, e.g. {'weg': 'wäck'}
OVERWRITE   = False         # True = synthesize again even if the file exists

# %%
# read the sentences
# One row per word in each item. Empty cells and cells with more than one word are reported.
csv_paths = sorted(glob.glob(os.path.join(SENTENCE_DIR, '*.csv')))
if not csv_paths:
    raise FileNotFoundError(f'no CSV files in {os.path.abspath(SENTENCE_DIR)}')

rows = []
for path in csv_paths:
    list_name = os.path.splitext(os.path.basename(path))[0]

    # read the file: UTF-8 first, then the encodings Excel uses; ',' or ';' as separator
    table = None
    for encoding in ('utf-8-sig', 'cp1252', 'mac_roman'):
        try:
            table = pd.read_csv(path, sep=None, engine='python', encoding=encoding, dtype=str)
            break
        except UnicodeDecodeError:
            pass
    if table is None:
        raise ValueError(f'cannot read {path}')
    table.columns = [c.strip() for c in table.columns]

    # the slot columns, in slot order (slot1_..., slot2_..., ...)
    slot_cols = [c for c in table.columns if re.match(r'slot\d+', c, re.I)]
    slot_cols = sorted(slot_cols, key=lambda c: int(re.match(r'slot(\d+)', c, re.I).group(1)))
    if not slot_cols:
        raise ValueError(f'{path}: no slot1_..., slot2_... columns (columns are {list(table.columns)})')

    for row_no, r in table.iterrows():
        cells = [' '.join(str(r[c]).split()) for c in slot_cols]
        if all(w in ('', 'nan') for w in cells):
            continue                                            # empty row
        if 'item' in table.columns and str(r['item']) not in ('', 'nan'):
            item = int(float(r['item']))
        else:
            item = row_no + 1
        for pos, word in enumerate(cells, start=1):
            if pos == 1 and LOWER_ITEM_INITIAL:
                word = word.lower()
            rows.append(dict(list=list_name, item=item, pos=pos, slot=slot_cols[pos - 1], word=word))

occ = pd.DataFrame(rows)

# checks
empty = occ[occ['word'].isin(['', 'nan'])]
multi = occ[occ['word'].str.contains(' ')]
for list_name, n in occ.groupby('list')['item'].nunique().items():
    print(f'{list_name}: {n} items')
if len(empty):
    print('\nEMPTY cells:')
    print(empty[['list', 'item', 'slot']].to_string(index=False))
if len(multi):
    print('\nCells with more than one word:')
    print(multi[['list', 'item', 'slot', 'word']].to_string(index=False))
occ = occ[~occ['word'].isin(['', 'nan'])].reset_index(drop=True)


# %%
# distinct words and filenames
words = occ.groupby('word', sort=False).size().reset_index(name='n_occ')

UMLAUTS = str.maketrans({'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue', 'ß': 'ss'})
bases, seen = [], {}
for word in words['word']:
    base = unicodedata.normalize('NFKD', word.translate(UMLAUTS)).encode('ascii', 'ignore').decode('ascii')
    base = re.sub(r'[^A-Za-z0-9_-]+', '_', base).strip('_').lower() or 'word'
    seen[base] = seen.get(base, 0) + 1
    bases.append(base if seen[base] == 1 else f'{base}_{seen[base]}')

words['base'] = bases
words['tts_text'] = [TTS_TEXT.get(w, w) for w in words['word']]
words['tts_file'] = [os.path.join(OUT_DIR, b + '.wav') for b in words['base']]

print(f'{len(words)} distinct words')
same_spelling = words[words['base'].str.replace(r'_\d+$', '', regex=True).duplicated(keep=False)]
if len(same_spelling):
    print('Same spelling, different case (listen whether gTTS says each the way you mean it):',
          ', '.join(f"{r.word} -> {os.path.basename(r.tts_file)}" for r in same_spelling.itertuples()))
words

# %%
# Synthesize (gTTS)
os.makedirs(OUT_DIR, exist_ok=True)

failed = []
n_new = 0
for i, row in words.iterrows():
    if os.path.exists(row['tts_file']) and not OVERWRITE:
        continue
    try:
        synthesize_word(row['tts_text'], row['tts_file'],
                        lang=TTS_LANG, tld=TTS_TLD, slow=TTS_SLOW, sr=TTS_SR,
                        retries=TTS_RETRIES, pause_s=TTS_PAUSE_S)
        n_new += 1
    except Exception as e:
        failed.append(row['word'])
        print(f"FAILED: {row['word']} ({type(e).__name__}: {e})")
    if (i + 1) % 10 == 0:
        print(f'{i + 1}/{len(words)}')

words['tts_ok'] = [os.path.exists(p) for p in words['tts_file']]
print(f"\n{n_new} new files, {words['tts_ok'].sum()}/{len(words)} words now in {OUT_DIR}/")
if failed:
    print('Failed (run this cell again):', ', '.join(failed))
# %%
# Save the tables

# * `words.csv`: one row per distinct word, with its TTS file
# * `items.csv`: one row per word in each item (list, item, position), with the file of that word

words.to_csv(os.path.join(TABLE_DIR, 'words.csv'), index=False)
items_out = occ.merge(words[['word', 'base', 'tts_file']], on='word', how='left')
items_out.to_csv(os.path.join(TABLE_DIR, 'items.csv'), index=False)
print('Saved', os.path.join(TABLE_DIR, 'words.csv'), 'and', os.path.join(TABLE_DIR, 'items.csv'))


# %%
# optionally listen
# LISTEN = list(words['word'][:8])     # or a list of words, e.g. ['Weg', 'weg']

# for word in LISTEN:
#     path = words.loc[words['word'] == word, 'tts_file'].iloc[0]
#     if os.path.exists(path):
#         print(word)
#         display(Audio(path))