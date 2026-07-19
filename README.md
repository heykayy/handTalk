# HandTalk-INDIA : ISL Real-Time Translator

A real-time **HandTalk-INDIA : Indian Sign Language (ISL) recognition system** built with TensorFlow, MediaPipe, and OpenCV.  
It runs entirely on CPU and supports two detection modes — letter-by-letter spelling and full-sentence signing.

> **Platform support: Windows only, for now.** Both the source code (via `pyttsx3`'s SAPI5 backend, `pywin32`, `comtypes`) and the packaged desktop app (built via GitHub Actions on `windows-latest`) currently target Windows exclusively. macOS/Linux support isn't available yet.

---

## Features

- **Word Mode** — detects individual ISL hand signs (digits 1–9, letters A–Z) and assembles them into words and sentences
- **Sentence Mode** — records a sequence of body/hand landmarks and classifies complete signed phrases
- **Text-to-Speech** — speaks recognised words and matched phrases aloud via `pyttsx3`
- **ISL Phrase Dictionary** — 50+ common ISL phrases auto-matched as you type
- **Live hold-progress ring** — visual feedback shows when a letter is about to commit
- **Screenshot saving** — press `S` at any time to save the current frame

---

## Pretrained Models

The trained `.keras` model files are **not committed to this repository** because of their size. Download them from Hugging Face and place each one in the folder shown below before running `predict.py`:

| Model | Link | Save to |
|---|---|---|
| Letter (word) model | [shutupmalfoy/isl_letter](https://huggingface.co/shutupmalfoy/isl_letter) | `word/isl_final_model.keras` |
| Sentence model | [shutupmalfoy/isl_sentence](https://huggingface.co/shutupmalfoy/isl_sentence) | `sentence/isl_sentence_model.keras` |

The label maps (`word/label_map.json` and `sentence/sentence_label_map.json`) are small enough to live directly in this GitHub repo, so no separate download is needed for those — only the two `.keras` model files above have to be pulled from Hugging Face.

Paths to both models are resolved automatically relative to the app's own location (see `app_paths.py`), so this works the same way whether you're running from source or from the packaged desktop app — no manual path editing needed.

---

## Project Structure

```
ISL/
│
├── word/                          # Word-mode artefacts
│   ├── isl_final_model.keras      # Final trained letter model  [download from HF]
│   ├── isl_best_model.keras       # Phase 1 checkpoint          [gitignored]
│   ├── label_map.json             # Class index → letter map    [in repo]
│   ├── sentence_builder.py        # Letter → word → sentence engine + phrase dict
│   ├── train.ipynb                # ← Train the letter (word) model
│   ├── word_model.py              # MobileNetV2 architecture + fine-tune helpers
│   └── plots/                     # Training curve images       [gitignored]
│
├── sentence/                      # Sentence-mode artefacts
│   ├── isl_sentence_model.keras   # Trained sentence model      [download from HF]
│   ├── sentence_label_map.json    # Class index → phrase map    [in repo]
│   ├── sentence_model.py          # Model architecture + constants + saved-model paths
│   └── train_sentence.ipynb       # ← Train the sentence (GRU) model
│
├── saved_predictions/             # Screenshots saved at runtime [gitignored]
│
├── utils.py                       # Data generators, smoothers, drawing utils
├── predict.py                     # Real-time webcam inference (both modes)
│
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.9 – 3.11
- Webcam
- A working TTS backend for `pyttsx3` (Windows: SAPI5, ships built in. macOS: NSSpeechSynthesizer, ships built in. Linux: install `espeak` — `sudo apt install espeak`)

Install dependencies:

```bash
pip install tensorflow==2.19.1 mediapipe opencv-python pyttsx3 scikit-learn matplotlib numpy
```

> **GPU users:** replace `tensorflow` with `tensorflow-gpu` for faster training.  
> Mixed-precision (`float16`) is automatically beneficial only on GPU — the training script leaves it off for CPU.

---

## Quick Start

### 1 — Get the models

Either download the pretrained models from Hugging Face (see [Pretrained Models](#pretrained-models) above), or train your own with the steps below.

### 2 — Prepare your dataset (only if training from scratch)

Organise images into one sub-folder per letter/digit:

```
isl_word/
├── 1/   (1200 images)
...
├── A/   (1200 images)
├── B/   (1200 images)
...
└── Z/   (1200 images)
```

### 3 — Train the letter model (only if training from scratch)

Open `word/train.ipynb` and update the dataset path at the top:

```python
DATASET_PATH = r"C:\path\to\isl_word"
```

Run all cells. Training runs in two phases:

| Phase | What happens | Typical duration |
|---|---|---|
| Phase 1 | Classification head trained, MobileNetV2 frozen | ~5–10 min |
| Phase 2 | Top 45 MobileNetV2 layers fine-tuned | ~5–8 min |

Output files saved automatically to `word/`.

Sentence Mode has its own trainer, `sentence/train_sentence.ipynb`, which extracts MediaPipe Holistic landmarks from short video clips (one folder per phrase) and trains a Conv1D + GRU sequence model. It's a separate pipeline from the letter model and only needs to be run if you're retraining the sentence model yourself.

### 4 — Run live detection

```bash
python predict.py
```

---

## Controls

### Both modes
| Key | Action |
|---|---|
| `M` | Toggle Word ↔ Sentence mode |
| `S` | Save screenshot |
| `Q` | Quit |

### Word Mode
| Key | Action |
|---|---|
| `SPACE` | Confirm current word, start next word |
| `BACKSPACE` | Delete last letter (or restore last word) |
| `ENTER` | Speak full sentence / show matched phrase |
| `C` | Clear everything |
| `H` | Toggle ISL phrase cheat-sheet |

### Sentence Mode
| Key | Action |
|---|---|
| `R` | Start / cancel a signing recording |

---

## Training Configuration

Key hyper-parameters in `word/train.ipynb` (tuned for 1200 images/class):

| Parameter | Value | Notes |
|---|---|---|
| `BATCH_SIZE` | 64 | Increase to 128 if ≥8 GB RAM free |
| `PHASE1_EPOCHS` | 20 | Early stopping kicks in earlier with more data |
| `PHASE1_LR` | 1e-3 | Safe with batch size 64 |
| `FINE_TUNE_LR` | 5e-5 | Conservative enough to avoid catastrophic forgetting |
| `UNFREEZE_N` | 45 | Top 45 MobileNetV2 layers unfrozen in Phase 2 |

---

## Model Architecture

### Word (letter) model

```
Input (224×224×3)
    └── MobileNetV2 (ImageNet pretrained, frozen in Phase 1)
            └── GlobalAveragePooling2D
                └── BatchNormalization
                    └── Dense(384, relu) → Dropout(0.4)
                        └── Dense(128, relu) → Dropout(0.3)
                            └── Dense(35, softmax)   # digits 1-9 + A-Z
```

### Sentence model

```
Input (45 frames × 258 landmark features)
    └── Conv1D(64) → BatchNorm → Conv1D(64) → BatchNorm → MaxPool → Dropout
        └── GRU(128, return_sequences=True) → GRU(64)
            └── BatchNormalization → Dense(64, relu) → Dropout(0.4)
                └── Dense(num_phrases, softmax)
```

Total params: ~85 K

---

## Inference Pipeline (Word Mode)

1. MediaPipe Hands detects the **primary hand** (closest to frame centre)
2. Hand ROI is cropped with 30% padding
3. Skin segmentation replaces the background with the training-set green
4. ROI resized to 224×224, normalised to `[0, 1]`
5. Letter model predicts class; majority-vote smoother (3 frames) removes flicker
6. `SentenceBuilder` commits a letter only after it is held stably for **10 frames** (~0.33 s at 30 fps)

## Inference Pipeline (Sentence Mode)

1. MediaPipe Holistic tracks both hands + upper-body pose while `R` recording is active
2. Each frame's landmarks are flattened into a 258-value vector (left hand 63 + right hand 63 + pose 132)
3. Once 45 frames are collected, the sequence is normalised (per-frame, zero-mean/unit-std) to match training
4. The GRU model classifies the whole sequence as one of the known phrases and speaks the result

---

## Text-to-Speech

Text-to-speech is handled by `pyttsx3`, run on a background worker thread so it never blocks the video loop. A dedicated `pyttsx3` engine is created fresh for every phrase spoken. This matters: reusing a single long-lived engine instance across many `say()`/`runAndWait()` calls is a known pyttsx3 failure mode (most noticeable on Windows/SAPI5) where the first utterance plays fine and everything after it goes silent — recreating the engine per utterance avoids that.

If you still get no audio at all:
- Confirm `pyttsx3` installed cleanly (`pip show pyttsx3`) and that a system TTS voice is available (see the Requirements section above for per-OS backends).
- On Linux, make sure `espeak` (or `espeak-ng`) is installed — pyttsx3 has no built-in fallback without it.
- Check the console for `[TTS] Speech failed: ...` log lines, which indicate the underlying engine call raised an error.

---

## Desktop App (GitHub Releases)

**Download (Windows only):** grab the latest `HandTalk-INDIA-Windows.zip` from this repo's [Releases](../../releases) page, unzip it, and run `HandTalk-INDIA.exe` — no Python install required. There is currently no macOS or Linux build.

Besides the source code in this repo, HandTalk-INDIA can be packaged into a standalone desktop app and published as a downloadable file on the repo's **Releases** page. See:

- `predict.spec` — PyInstaller build spec (bundles the models, label maps, and MediaPipe's internal data files)
- `.github/workflows/build-release.yml` — builds the Windows executable automatically and attaches it to a Release whenever you push a version tag
- `app_paths.py` — resolves file paths correctly whether the app is running from source or from the packaged executable

Full step-by-step instructions for building and publishing a release are covered separately; see the project's release process notes.

---

## Known Limitations

- Word Mode is designed for **static ISL hand signs** (digits 1–9, letters A–Z); dynamic/motion signs are handled by Sentence Mode only
- Background segmentation uses HSV + YCrCb skin detection — performance may vary under poor or inconsistent lighting
- Sentence Mode's phrase vocabulary is limited to whatever classes it was trained on (see `sentence/sentence_label_map.json`) — it does not generalise to arbitrary sentences

---

## Dataset Reference

Letter vocabulary sourced from:
https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl

Phrase vocabulary sourced from:  
https://www.kaggle.com/datasets/biswajit002/isl-video-sentences-dataset-for-recognition

---

## License

MIT— free to use, modify, and distribute.

---

## Author

Creator of **HandTalk-INDIA** : **heykayy** & **komal05-web**
