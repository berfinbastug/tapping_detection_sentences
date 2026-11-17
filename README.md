# tapping_detection_sentences

Experiment 3 of the thesis: **is timing perception different at syntactic phrase boundaries than within phrases?**

German six-word sequences are turned into **discrete sine-wave speech (SWS)**: every word is equally long and
sits on an isochronous 450 ms grid, so a phrase boundary exists only in the syntax. Two between-subject groups
hear the same sounds:

- **speech (informed):** told that the sounds are computer-generated speech; sees the words on screen.
- **acoustic (naive):** told only that they hear sounds; nothing on screen suggests speech.

Three sequence types: **2|4** sentences (`im Moor | glomm ein Licht auf`), **3|3** sentences
(`auf dem Dach | lag der Schnee`) and **word lists** (no structure). Timing shifts (phase shifts) at interval 2
or 3 fall on a boundary in one sentence type and inside a phrase in the other.

One session = **spontaneous tapping → tapping training → tapping experiment → detection experiment**.

| Part | Task | Length |
|---|---|---|
| Spontaneous tapping | tap at your own pace, no sound | 1 min |
| Tapping training | examples with feedback + 6-sequence test run | ~5 min |
| Tapping experiment | tap along with 30-sound sequences (6 noises, 6 words, 6 noises, the same 6 words, 6 noises); 216 sequences in 9 blocks | ~70 min with breaks |
| Detection experiment | regular or irregular? (6 words); 162 sequences in 3 blocks, with its own training | ~20 min + breaks |

Detailed documentation of every file, setting and output column: the Notion page **📌 15.11 (directory)**.
Where the notes and the code disagree, the code is right.

---

## Folder structure

```
tapping_detection_sentences/
├── repo/                        library: word -> TTS -> trim -> SWS, word-shaped noise, sequences on a grid
├── sentence_lists/              the word material (one CSV per list)
│   ├── items_2_4.csv            2|4 sentences (first 18 items used)
│   ├── items_3_3.csv            3|3 sentences (first 18 items used)
│   ├── training_items.csv       detection training items (TR_DEMO_WL, TR_S24, TR_S33, TR_WL)
│   └── training_items_tapping.csv   tapping training example word list (TR_TAP_WL)
├── sws_words/                   every word at every stage (shared by both experiments)
│   ├── words.csv  items.csv  trim_info.csv  sws_info.csv
│   ├── 1_tts/  2_trimmed/  3_sws/            (audio, not on GitHub)
│   └── noise_pool/                           (tapping noise tokens, not on GitHub)
├── stimuli_generator_codes/     the word pipeline + noise pool + stim_helpers.py
├── detection/                   the detection experiment
│   ├── design_tables/           per-participant tables + training_design.csv
│   ├── design_tables_stim/      training table with stim_row / n_samples (read by the experiment)
│   ├── stimuli/<ID>_<group>/    per-participant sounds                (not on GitHub)
│   ├── training_stimuli/        training sounds                       (not on GitHub)
│   └── data/   data_test/       session data (lab / laptop tests)     (not on GitHub)
├── tapping/                     the tapping experiment (+ training, spontaneous tapping, hardware test)
│   ├── design_tables/           per-participant tables + event tables + training tables
│   ├── stimuli_tapping/<ID>_<group>/   per-participant sounds          (not on GitHub)
│   ├── training_stimuli_tapping/       training sounds                (not on GitHub)
│   └── data/   data_test/   tapping_test/   recordings and data       (not on GitHub)
└── archive/                     old versions, not used
```

Everything that is generated (audio, stimuli) or personal (participant data) is excluded in `.gitignore`.
It can all be rebuilt from the code, the sentence lists and the design tables, but **back up the data
separately**: it exists only on the laptop / the lab server.

---

## Setup

**Python environment:** use the conda environment **`psychopy`** for every script in this project
(the `base` environment has a broken pandas/numpy combination).

```bash
conda activate psychopy
cd ~/Desktop/tapping_detection_sentences
```

Packages used: `psychopy`, `psychtoolbox`, `numpy`, `scipy`, `pandas`, `soundfile`, `matplotlib`,
`praat-parselmouth`, `gTTS` + `pydub` (only for making new words; needs internet).

**Paths:** every script knows the project folder through one setting at its top
(`PROJECT_DIR` in the stimulus scripts, `MAIN_DIR` in the params files), so scripts can be started from any
folder. In the lab, change these to the server path (see *Lab checklist* below).

**Participant IDs:** type them in upper case (`P01`), and use every ID only once across both groups
(the random seed comes from the ID only).

---

## Running order

### 1. Words (shared; once, and again whenever the sentence lists change)

| Step | Script (`stimuli_generator_codes/`) | Output |
|---|---|---|
| 1 | `sentence_lists_to_tts.py` — every distinct word once with gTTS (needs internet) | `sws_words/1_tts/`, `words.csv`, `items.csv` |
| 2 | `trim_tts_words.py` — cut silence around each word | `sws_words/2_trimmed/`, `trim_info.csv` |
| 3 | `make_sws_words_trimmed.py` — discrete SWS, every word 350 ms | `sws_words/3_sws/`, `sws_info.csv` |
| 4 | `make_noise_pool.py` — 20 noise tokens matched to the words (tapping only) | `sws_words/noise_pool/` |

If the words change, rebuild the noise pool and then **all** tapping stimuli (main and training). The tapping
builders stop with a clear message if a word does not match the pool.

### 2. Detection

| When | Script (`detection/`) | Output |
|---|---|---|
| once | `make_training_design.py` | `design_tables/training_design.csv` |
| once | `build_training_stimuli.py` | `training_stimuli/`, `design_tables_stim/training_design.csv` |
| per participant | `generate_detection_table.py` (asks ID + group) | `design_tables/detection_<ID>_<group>.csv` |
| per participant | `build_detection_stimuli.py` (asks ID + group) | `stimuli/<ID>_<group>/` |
| per participant | `plot_detection_stimuli.py` (set ID + group in the file) — checks every word is where it should be | figures only |
| session | `run_detection_speech.py` / `run_detection_acoustic.py` | `data/` (lab) or `data_test/` (laptop) |

### 3. Tapping

| When | Script (`tapping/`) | Output |
|---|---|---|
| once | `make_training_design_tapping.py` | `design_tables/tapping_training_design.csv` + `_events.csv` |
| once | `build_tapping_training_stimuli.py` | `training_stimuli_tapping/` |
| per participant | `generate_tapping_design_table.py` (asks ID + group) | `design_tables/tapping_<ID>_<group>.csv` + `tapping_events_<ID>_<group>.csv` |
| per participant | `build_tapping_stimuli.py` (asks ID + group) | `stimuli_tapping/<ID>_<group>/` (~520 MB) |
| per participant | `plot_tapping_stimuli.py` (set ID + group in the file) — checks every sound | figures only |
| session | `baseline_tap.py` — spontaneous tapping, 1 min | `data/` or `data_test/`: `spontaneous_<ID>_<time>.wav/.png/.json` |
| session | `run_tapping_training_speech.py` / `run_tapping_training_acoustic.py` | `tapping_training_<ID>_<group>_<time>.tsv` + raw wavs |
| session | `run_tapping_speech.py` / `run_tapping_acoustic.py` | `tapping_<ID>_<group>_<time>.tsv` + one raw wav per trial |
| any time | `tapping_hardware_test.py` — checks that the microphone records the taps | `tapping_test/` |

### A session (one participant)

```bash
conda activate psychopy
cd ~/Desktop/tapping_detection_sentences
python tapping/baseline_tap.py                     # spontaneous tapping
python tapping/run_tapping_training_speech.py      # or ..._acoustic.py
python tapping/run_tapping_speech.py               # or ..._acoustic.py
python detection/run_detection_speech.py           # or ..._acoustic.py
```

Before the session: the participant's stimuli must be built (detection and tapping), and the training
stimuli of both experiments must exist.

---

## How the experiments save data

- **Every trial is saved right away** (one row appended to a `.tsv`, plus a raw `.wav` per tapping trial).
  A crash or the escape key loses at most the trial that was running.
- **Nothing is ever overwritten:** every session writes new files with its start time in the name.
  Resuming after a crash (start block > 1 in the dialog) writes its own new files.
- **Tapping recordings are raw:** exactly the samples the audio interface delivered (32-bit float wav, no
  filtering). The data file logs where the sound starts in each recording (`sound_start_in_rec_s`), and the
  session folder gets a copy of the onset table, so every tap can be put on the stimulus time line later.
- **Online tap check:** after each tapping trial a quick check runs on a *copy* of the recording (taps per sound,
  double taps, inter-tap intervals). It drives the console output, the break-screen feedback and the training
  pass rule; the values are logged in the `online_*` columns. The saved wav is never changed.
- **Snapshots:** each session also saves which stimulus files and settings were used (`*_stiminfo_*`,
  `*_stimmeta_*`).

---

## Laptop vs lab

Every experiment has two params files: `*_params.py` (lab) and `*_params_laptop.py` (imports the lab file and
changes only paths, audio devices, window size and test-run length). The run scripts choose with
`SETUP = 'laptop'` or `'lab'` at the top. `baseline_tap.py` has its own `SETUP` line.

| | Laptop (test) | Lab (booth) |
|---|---|---|
| Sound | EarPods | the audio interface (full duplex) |
| Recording | TASCAM US-2x2HR, desk microphone on input 2 | interface input 2 = microphone, input 1 = loopback cable |
| Saved channels | microphone only | loopback (sound as played) + microphone |
| Data folder | `data_test/` | `data/` |
| Blocks | shortened (`MAX_TRIALS_PER_BLOCK`) | full |

### Lab checklist

- [ ] `SETUP = 'lab'` in all run scripts and in `baseline_tap.py`.
- [ ] Server path: `MAIN_DIR` in `detection/detection_params.py`, `tapping/tapping_params.py`,
      `tapping/tapping_training_params.py`, `baseline_tap.py`; `PROJECT_DIR` in the stimulus scripts if they
      run on the lab machine. Copy the built stimuli (`detection/stimuli/`, `detection/training_stimuli/`,
      `detection/design_tables_stim/`, `tapping/stimuli_tapping/`, `tapping/training_stimuli_tapping/`) there.
- [ ] Audio device names (`DEVICE_NAME`, `INPUT_DEVICE_NAME`) and `LATENCY_CLASS` match the booth; the tapping
      training has its own copy of these settings (`tapping_training_params.py`).
- [ ] Loopback cable from an output into input 1; microphone in input 2.
- [ ] `STIM_GAIN` (level) set; sound in both ears; texts, ■ and ● fit and display on the booth screen.
- [ ] One full test session; open the output files and one recording.

---

## The `repo/` package

Functions only: every setting is passed in by the calling script. Scripts import it with `from repo import …`
(the project folder is put on the path). The SWS signal processing itself comes from
`stimuli_generator_codes/stim_helpers.py`, which `repo/sws.py` calls unchanged.

| Module | What it does |
|---|---|
| `tts.py` | gTTS (or edge-tts) → mono WAV |
| `trim.py` | trim silence around TTS words; trim SWS to its audible part |
| `equalize.py` | equal word durations by stretching the formant/voicing tracks (50 ms steps stay 50 ms) |
| `sws.py` | wrappers around `stim_helpers.py`: tracks, discrete SWS, fade/normalize |
| `audio.py` | load/save/resample; the −40 dB onset rule used everywhere |
| `sequence.py` | put sounds on the onset grid (phase shifts); noise token order |
| `noise.py` | word-shaped noise pool (spectrum, envelope, rise time and level matched to the words) |
| `pipeline.py` | all word steps in one call (not used by the current scripts) |

**Onset rule of the whole project:** a sound starts where its 5 ms RMS envelope rises above its peak − 40 dB.
The same rule trims the words, places them on the grid, and checks the built stimuli.