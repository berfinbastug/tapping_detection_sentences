"""
check_05_boundary_match.py
Quick check after generating new words: onset class (F / M / S) of every word in the
v2 lists, and whether the boundary pair matches (2|4: slot2|slot3, 3|3: slot3|slot4).

Run AFTER check_03_word_acoustics.py (needs checks/03_word_acoustics.csv).
Run (conda env `psychopy`):  python check_05_boundary_match.py
"""
import os
import pandas as pd

BASE_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
os.chdir(BASE_DIR)
CSV_DIR =os.path.join(BASE_DIR, 'sentence_lists')
ACOUSTICS = os.path.join(BASE_DIR, 'checks', '03_word_acoustics.csv')
WORDS_CSV = os.path.join(BASE_DIR, 'sws_words', 'words.csv')
LISTS     = {'2|4': ('items_2_4.csv', 2), '3|3': ('items_3_3.csv', 3)}   # boundary after slot k
F_MAX, M_MAX = 35, 80          # t50 class limits in ms (F <= 35 < M < 80 <= S)
WATCH = ['fiel', 'brennt', 'stand', 'kam', 'zog', 'trank', 'floh', 'stieg', 'fuhr', 'brach',
         'Tür', 'Haus', 'Wald', 'Zug', 'Strom', 'Gold', 'Freund', 'Wind', 'Schein', 'still', 'ab',
         'Frust', 'rutscht', 'taut']

w2b = {}
if os.path.exists(WORDS_CSV):
    w = pd.read_csv(WORDS_CSV, dtype=str)
    w2b = dict(zip(w['word'].str.strip(), w['base'].str.strip()))


def base_name(word):
    if word in w2b:
        return w2b[word]
    x = word.strip().lower()
    for a, b in (('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')):
        x = x.replace(a, b)
    return x


ac = pd.read_csv(ACOUSTICS)
ac = ac[ac.kind == 'word']
t50 = dict(zip(ac.base, ac.t50_ms))
t10 = dict(zip(ac.base, ac.t10_ms))
snd = dict(zip(ac.base, ac.sound_ms))


def cls(word):
    b = base_name(word)
    if b not in t50:
        return '?'
    v = t50[b]
    return 'F' if v <= F_MAX else ('M' if v < M_MAX else 'S')


print('=== New / watched words ===')
print(f'{"word":9s} {"class":5s} {"t10":>6s} {"t50":>6s} {"sound_ms":>8s}  notes')
for wd in WATCH:
    b = base_name(wd)
    if b in t50:
        note = []
        if t10[b] > 15:
            note.append('weak first frame (t10 > 15)')
        if snd[b] < 330:
            note.append('SHORT sound (capped?)')
        print(f'{wd:9s} {cls(wd):5s} {t10[b]:6.1f} {t50[b]:6.1f} {snd[b]:8.1f}  {"; ".join(note)}')
    else:
        print(f'{wd:9s} {"?":5s}   (no audio yet)')

for t, (fn, k) in LISTS.items():
    path = os.path.join(CSV_DIR, fn)
    if not os.path.exists(path):
        print(f'\n[skip] {fn} not found')
        continue
    d = pd.read_csv(path, dtype=str).fillna('')
    sc = [c for c in d.columns if c.startswith('slot')]
    print(f'\n=== {t}: boundary pairs (slot{k} | slot{k+1}) ===')
    n_ok = n_bad = n_open = 0
    for _, r in d.iterrows():
        a, b = r[sc[k - 1]], r[sc[k]]
        ca, cb = cls(a), cls(b)
        s_words = [x for x in (r[c] for c in sc) if cls(x) == 'S']
        if '?' in (ca, cb):
            tag, n_open = 'OPEN', n_open + 1
        elif ca == cb:
            tag, n_ok = 'ok', n_ok + 1
        else:
            tag, n_bad = 'MISMATCH', n_bad + 1
        if tag != 'ok' or s_words:
            extra = f'  S: {s_words}' if s_words else ''
            print(f'{r["item"]:>3s} {" ".join(r[c] for c in sc):36s} {a}({ca}) | {b}({cb})  {tag}{extra}')
            if tag == 'MISMATCH':
                print(f'      status/backup: {r.get("status", "")}')
    print(f'--> ok {n_ok}, mismatch {n_bad}, open (no audio) {n_open}')