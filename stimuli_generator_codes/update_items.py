"""
update_items_csv.py
Rewrite sws_words/items.csv so it holds ONLY the final material:
    items_2_4  (18 selected sentences, sentence_lists/items_2_4.csv)
    items_3_3  (18 selected sentences, sentence_lists/items_3_3.csv)
    items_wl   (18 word lists,         sentence_lists/items_wl.csv)

No TTS, no new audio: every word is looked up in the existing sws_words/words.csv
(case-sensitive, so 'Teil' and 'teil' keep their own files) and checked against sws_info.csv.
words.csv and sws_info.csv are NOT changed (training stimuli also use them).

The old items.csv is kept as sws_words/items_before_update.csv.

Run (conda env `psychopy`):  python update_items_csv.py
"""
import os
import shutil

import pandas as pd

# ------------------------------------------------------------------ CONFIG
PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
LISTS       = ['items_2_4', 'items_3_3', 'items_wl']    # files in sentence_lists/ (without .csv)
N_ITEMS     = 18
N_WORDS     = 6

os.chdir(PROJECT_DIR)
ITEMS_CSV    = 'sws_words/items.csv'
BACKUP_CSV   = 'sws_words/items_before_update.csv'
WORDS_CSV    = 'sws_words/words.csv'
SWS_INFO_CSV = 'sws_words/sws_info.csv'

# ------------------------------------------------------------------ BUILD
words = pd.read_csv(WORDS_CSV)
if words['word'].duplicated().any():
    raise ValueError(f'{WORDS_CSV}: duplicated words {sorted(words.loc[words["word"].duplicated(), "word"])}')
word_map = words.set_index('word')[['base', 'tts_file']]
sws_bases = set(pd.read_csv(SWS_INFO_CSV)['base'])

rows = []
for lst in LISTS:
    d = pd.read_csv(os.path.join('sentence_lists', f'{lst}.csv')).sort_values('item')
    slots = [c for c in d.columns if c.startswith('slot')]
    if len(d) != N_ITEMS or len(slots) != N_WORDS or d['item'].duplicated().any():
        raise ValueError(f'{lst}: {len(d)} items / {len(slots)} slots, expected {N_ITEMS} / {N_WORDS}')
    for r in d.itertuples(index=False):
        for pos, slot in enumerate(slots, start=1):
            rows.append(dict(list=lst, item=r.item, pos=pos, slot=slot, word=getattr(r, slot)))

new = pd.DataFrame(rows)
missing = sorted(set(new['word']) - set(word_map.index))
if missing:
    raise ValueError(f'words with no entry in {WORDS_CSV} (would need new TTS): {missing}')
new = new.join(word_map, on='word')
no_sws = sorted(set(new['base']) - sws_bases)
if no_sws:
    raise ValueError(f'bases with no SWS file in {SWS_INFO_CSV}: {no_sws}')

# ------------------------------------------------------------------ SAVE
if not os.path.exists(BACKUP_CSV):
    shutil.copy(ITEMS_CSV, BACKUP_CSV)
new.to_csv(ITEMS_CSV, index=False)

print(f'Saved {ITEMS_CSV}: {len(new)} rows (old version: {BACKUP_CSV})')
print(new.groupby('list')['item'].nunique().to_string())