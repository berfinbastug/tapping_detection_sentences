"""
make_word_lists.py
Word lists by constrained randomization.

The word tokens of the 36 selected sentences (18 x 2|4 + 18 x 3|3 = 216 tokens) are pooled by
word class (P, Det, N, V, Part, from the slot names). Each list has the word classes of one
sentence type: 9 lists like 2|4 (P N V Det N Part), 9 like 3|3 (P Det N V Det N). Its words are
drawn at random from the class pools (each token used at most once) and put in random order.
So the word lists contain the same kinds of words as the sentences, only scrambled.
A draw is rejected and redrawn if
  - the same word occurs twice in the list, or
  - two NEIGHBOURING words form a meaningful unit:
      phrase         preposition + article/noun   (auf dem, im Raum)
                     article + noun               (der Schnee)
      clause         noun + verb, verb + noun     (Hund lief, lag Schnee)
                     article + verb               (der kam = 'he came')
      compound       noun + noun                  (Bier Bauch)
      particle verb  particle/preposition + verb, verb + particle/preposition (weg lief, lief weg)
      fusion         preposition/particle + preposition/particle (zu vor = zuvor, durch aus)
    Exceptions (cannot form a unit):
      - article + noun, and im/am/beim/vom/zum + noun, are allowed when they do NOT agree in
        gender (die Mann, ein Nacht, im Bank): ungrammatical, so not a phrase (GENDER below);
      - the contracted prepositions im, am, beim, vom, zum already contain an article, so they
        may precede an article or a verb;
      - 'in' and 'für' are never verb particles, so they may precede a verb.
  - any pair in BLOCK_PAIRS (add pairs by hand if a list still sounds like a unit).
Word classes come from the slot names of the sentence files (slot1_P, slot2_Det, ...).

Output: word_lists/items_wl.csv  (item, slot1_W ... slot6_W)
Optional check: t50 shift per interval (checks/03_word_acoustics.csv + sws_words/words.csv).

Do NOT put items_wl.csv into sentence_lists/ and run sentence_lists_to_tts.py:
it lowercases the first word of each row (Tor -> tor) and would download NEW audio.

Run (conda env `psychopy`):  python make_word_lists.py
"""
import os
import random

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ CONFIG
PROJECT_DIR = '/Users/bastugb/Desktop/tapping_detection_sentences'
SENT_DIR    = os.path.join(PROJECT_DIR, 'sentence_lists')
SENT_FILES  = ['items_2_4.csv', 'items_3_3.csv']   # order matters for reproducibility
OUT_DIR     = os.path.join(PROJECT_DIR, 'word_lists')
OUT_FILE    = 'items_wl.csv'
N_LISTS     = 18
LIST_LEN    = 6
SEED        = 1
OVERWRITE   = True         # True = replace an existing items_wl.csv
MAX_DRAWS   = 20000        # random draws per list before restarting the set
MAX_SETS    = 200          # restarts of the whole set

CONTRACTED  = {'im', 'am', 'beim', 'vom', 'zum'}   # preposition + article in one word
NOT_PARTICLE = {'in', 'für'}                       # prepositions that are never verb particles
# noun gender (m, f, n); 'num' = numeral (always blocked after an article)
GENDER = {
    'm': 'Raum Rhein Blitz Wein Mut Baum Turm Schrei Graf Teil Druck Ort Markt Feind Mai Lärm Mann '
         'Brief Schein Sohn Wald Brand Chor Strom Zug Schnee Bach Laut Berg Hund Arzt Rauch',
    'f': 'Wut Bank Nacht Maus Tür',
    'n': 'Reh Fest Teil Tor Mahl Bier Gleis Dach Licht Kind Boot Gras Loch Haus',
    'num': 'neun zwei',
}
# genders an article (or contracted preposition) agrees with, for a bare singular noun
AGREES = {'der': 'mf', 'die': 'f', 'dem': 'mn', 'den': 'm',
          'ein': 'mn', 'kein': 'mn', 'mein': 'mn', 'dein': 'mn', 'sein': 'mn', 'ihr': 'mn',
          'im': 'mn', 'am': 'mn', 'beim': 'mn', 'vom': 'mn', 'zum': 'mn'}
BLOCK_PAIRS = set()        # extra pairs to forbid, lowercase, e.g. {('mein', 'heim')}

ACOUSTICS   = os.path.join(PROJECT_DIR, 'checks', '03_word_acoustics.csv')   # optional check
WORDS_CSV   = os.path.join(PROJECT_DIR, 'sws_words', 'words.csv')
# -------------------------------------------------------------------------

# 1 word tokens and word classes from the sentence files
tokens, classes, templates = [], {}, []
for fn in SENT_FILES:
    df = pd.read_csv(os.path.join(SENT_DIR, fn), dtype=str).fillna('')
    df.columns = [c.strip() for c in df.columns]
    slot_cols = [c for c in df.columns if c.lower().startswith('slot')]
    templates += [[c.split('_', 1)[1] for c in slot_cols]] * (N_LISTS // len(SENT_FILES))
    for _, r in df.iterrows():
        for c in slot_cols:
            w = r[c].strip()
            tokens.append((w, c.split('_', 1)[1]))
            classes.setdefault(w.lower(), set()).add(c.split('_', 1)[1])   # slot3_V -> 'V'
print(f'{len(tokens)} word tokens from {", ".join(SENT_FILES)}')
gender = {}
for g, ws in GENDER.items():
    for w in ws.split():
        gender.setdefault(w.lower(), set()).add(g)
no_gender = sorted({w for w in classes if 'N' in classes[w] and w not in gender})
if no_gender:
    raise ValueError(f'add these nouns to GENDER: {no_gender}')


def agrees(art, noun):
    g = gender.get(noun.lower(), set())
    return 'num' in g or bool(set(AGREES.get(art.lower(), 'mfn')) & g)


def roles(w):
    l, c = w.lower(), classes[w.lower()]
    r = set()
    if c & {'P', 'Part'}:
        r.add('contracted' if l in CONTRACTED else 'prep_only' if l in NOT_PARTICLE else 'particle')
    if 'Det' in c:
        r.add('det')
    if 'N' in c:
        r.add('noun')
    if 'V' in c:
        r.add('verb')
    return r


def unit(a, b):
    """Reason why the neighbours a, b form a meaningful unit, or None."""
    ra, rb = roles(a), roles(b)
    if (a.lower(), b.lower()) in BLOCK_PAIRS:
        return 'blocked by hand'
    if 'contracted' in ra and 'noun' in rb and agrees(a, b):
        return 'phrase (P + N)'
    if ra & {'prep_only', 'particle'} and 'noun' in rb:
        return 'phrase (P + N)'
    if ra & {'prep_only', 'particle'} and 'det' in rb:
        return 'phrase (P + Det)'
    if 'det' in ra and 'noun' in rb and agrees(a, b):
        return 'phrase (Det + N)'
    if 'det' in ra and 'verb' in rb:
        return 'clause (Det + V)'
    if 'noun' in ra and 'verb' in rb:
        return 'clause (N + V)'
    if 'verb' in ra and 'noun' in rb:
        return 'clause (V + N)'
    if 'noun' in ra and 'noun' in rb:
        return 'compound (N + N)'
    if 'particle' in ra and 'verb' in rb:
        return 'particle verb (Part + V)'
    if 'verb' in ra and rb & {'particle', 'prep_only'}:
        return 'particle verb (V + Part)'
    if 'particle' in ra and rb & {'particle', 'prep_only'}:
        return 'fusion (P + P)'
    if ra & {'contracted', 'prep_only'} and 'particle' in rb:
        return 'fusion (P + P)'
    return None


def ok(l):
    return (len({w.lower() for w in l}) == len(l)
            and all(unit(a, b) is None for a, b in zip(l, l[1:])))


# 2 draw the lists (if a class pool runs out of valid draws, start the whole set again)
rng = random.Random(SEED)
for attempt in range(1, MAX_SETS + 1):
    pools = {}
    for w, c in tokens:
        pools.setdefault(c, []).append(w)
    lists = []
    for k, templ in enumerate(templates):
        for n in range(MAX_DRAWS):
            pick = {}
            for c in sorted(set(templ)):
                pick[c] = rng.sample(range(len(pools[c])), templ.count(c))
            cand = [pools[c][i] for c in sorted(pick) for i in pick[c]]
            rng.shuffle(cand)
            if ok(cand):
                break
        else:
            break                                   # stuck: restart the set
        lists.append(cand)
        for c, idx in pick.items():
            for i in sorted(idx, reverse=True):
                pools[c].pop(i)
    if len(lists) == N_LISTS:
        break
else:
    raise RuntimeError(f'no valid set of {N_LISTS} lists in {MAX_SETS} attempts')
print(f'seed {SEED}: valid set found on attempt {attempt}')
print('list types:', ' '.join('2|4' if 'Part' in t else '3|3' for t in templates))

# 3 save
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, OUT_FILE)
if os.path.exists(out_path) and not OVERWRITE:
    raise FileExistsError(f'{out_path} exists - set OVERWRITE = True to replace it')
out = pd.DataFrame([[k + 1] + l for k, l in enumerate(lists)],
                   columns=['item'] + [f'slot{p}_W' for p in range(1, LIST_LEN + 1)])
out.to_csv(out_path, index=False)
print(out.to_string(index=False))
print(f'\nSaved: {out_path}')

# 4 optional acoustic check: t50 shift per interval (t50 of word k+1 minus word k)
if os.path.exists(ACOUSTICS) and os.path.exists(WORDS_CSV):
    w2b = dict(pd.read_csv(WORDS_CSV, dtype=str)[['word', 'base']].values)
    ac = pd.read_csv(ACOUSTICS)
    t50 = dict(ac.loc[ac.kind == 'word', ['base', 't50_ms']].values)
    missing = sorted({w for l in lists for w in l if w2b.get(w) not in t50})
    if missing:
        print('[check skipped] no acoustics for:', missing)
    else:
        sh = np.array([[t50[w2b[b]] - t50[w2b[a]] for a, b in zip(l, l[1:])] for l in lists])
        print('t50 shift per interval (ms):', ' '.join(f'I{k + 1} {m:+.1f}' for k, m in enumerate(sh.mean(0))))
else:
    print('[check skipped] 03_word_acoustics.csv or words.csv not found')