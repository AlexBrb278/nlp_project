# Noisy Intent Detection for Chatbots
**University Politehnica of Bucharest — NLP Project 2026**

**Team:** Alexandru-Mihai Barbu · Ioan Vlad Istrate · Robert-George Zamfir

---

## Project Overview

This project investigates the robustness of intent classification models under noisy input conditions. We evaluate how different noise types affect BERT-based classifiers and propose multiple techniques to improve robustness: contrastive learning, character-level/token-free encoders, and prototypical networks.

**Dataset:** CLINC150 — 150 intents, 10 domains, 23,700 utterances + 1,000 out-of-scope queries

---

## Repository Structure

```
├── whisper_noise/
│   ├── generate_audio.py          # Text → MP3 using edge-tts (en-AU-WilliamNeural)
│   └── transcribe_whisper.py      # MP3 → noisy text using OpenAI Whisper
│
├── prototypical_networks/
├── proiectnlp.ipynb               # BERT baseline fine-tuning + noise evaluation
├── contrastive_learning.ipynb     # BERT + NT-Xent contrastive fine-tuning (150 classes)
├── bert_contrastive_proto.ipynb   # BERT + contrastive + prototypical (151 classes + OOS)
├── canine_contrastive_prototype.ipynb # CANINE + contrastive + prototypical (150 classes)
├── canine_contrastive_proto_oos.ipynb # CANINE + contrastive + prototypical (151 classes + OOS)
├── charLevelTokenizers.ipynb      # CharBiLSTM + CANINE-s baseline experiments (Vlad)
├── add_prototypical_networks.py   # Prototypical network implementation (Robert)
├── noisy_test_set2.json           # Pre-generated noisy test sets (all 7 noise types)
└── README.md
```

---

## Noise Types

Seven noise types are applied to the test set. Training data stays clean throughout.

| Noise Type | Method | Description |
|---|---|---|
| Keyboard | nlpaug KeyboardAug (15%) | QWERTY proximity character substitution |
| Spelling | nlpaug SpellingAug (20%) | Probabilistic misspelling dictionary |
| Synonyms | nlpaug SynonymAug (20%) | WordNet semantic substitution |
| Casing | str.upper() | All-caps transformation |
| Abbreviations | Custom 50-word map | SMS-style substitutions (pwd, u, tmrw...) |
| Whisper ASR | edge-tts → Whisper base | Neural TTS → real ASR transcription errors |

---

## How to Run

### Prerequisites

```bash
pip install transformers torch nlpaug openai-whisper edge-tts
```

For Whisper audio generation, also install ffmpeg:
- Windows: download from https://ffmpeg.org and add to PATH
- Linux/Colab: `apt-get install -y ffmpeg`

All notebooks are designed to run on **Google Colab with a T4 or A100 GPU**.

Mount your Google Drive and update the `CLINC_PATH` and `SAVE_DIR` paths at the top of each notebook to match your Drive structure.

---

### Step 1 — Get the Dataset

Download CLINC150 from the official repository:
```
https://github.com/clinc/oos-eval
```
Place `data_full.json` at:
```
/content/drive/MyDrive/NLP/oos-eval/data/data_full.json
```

---

### Step 2 — Generate Whisper Noise (optional)

The pre-generated noisy test sets are already in `noisy_test_set2.json`. If you want to regenerate the Whisper noise from scratch:

```bash
# Step 1 — generate audio files (~4500 MP3s)
python whisper_noise/simulare.py

# Step 2 — transcribe with Whisper
python whisper_noise/whisper_noise.py
```
The output is merged into `noisy_test_set2.json` under the `"whisper"` key.
The Whisper noise pipeline produces the most realistic noise in the project. It converts each test utterance to speech using Microsoft Edge's neural TTS engine (`en-AU-WilliamNeural`) and transcribes it back with OpenAI Whisper, producing genuine ASR errors — homophones, missing words, punctuation artifacts — that no rule-based system generates. Of 4,500 test utterances, **1,345 received meaningful substitutions**. The remaining utterances were transcribed correctly, which itself is a finding: short, clear chatbot queries are relatively resilient to ASR errors.


---

### Step 3 — Run Experiments

Run notebooks in this order:

**1. BERT Baseline + Noise Pipeline**
```
proiectnlp.ipynb
```
Fine-tunes bert-base-uncased on CLINC150 and evaluates on all noise types.
Expected clean accuracy: **96.22%**

**2. BERT + Contrastive Learning (150 classes)**
```
contrastive_learning.ipynb
```
Applies NT-Xent contrastive fine-tuning on top of the trained BERT.
Expected keyboard accuracy: **74.33%** (+31 points over baseline)

**3. BERT + Contrastive + Prototypical (151 classes with OOS)**
```
bert_contrastive_proto.ipynb 
```
Adds OOS as class 151, trains with contrastive loss, then builds prototypes.
Expected OOS recall: **31.50%** | Keyboard: **74.42%**

**4. CANINE + Contrastive + Prototypical (150 classes)**
```
canine_contrastive_prototype.ipynb
```
Replaces BERT encoder with CANINE-s (token-free, character-level).
Expected keyboard accuracy: **83.82%** (best keyboard result in the project)

**5. CANINE + Contrastive + Prototypical (151 classes with OOS)**
```
canine_contrastive_proto_oos.ipynb
```
Full combined model with OOS detection.
Expected keyboard: **81.69%** | OOS recall: **39.90%**

**6. Character-Level Encoders (Vlad)**
```
charLevelTokenizers.ipynb
```
CharBiLSTM and CANINE-s baseline experiments with OOS recall evaluation.

**7. Prototypical Networks**
```
add_prototypical_networks.py
```
Prototype-based classification using BERT embeddings.

---

## Key Results

| Model | Clean Acc | Keyboard | OOS Recall |
|---|---|---|---|
| BERT baseline | 96.22% | 43.31% | — |
| BERT + Contrastive | 96.60% | 74.33% | 31.50% |
| CANINE clean | 91.60% | 69.76% | — |
| CANINE + Contrastive + Proto (150) | 92.02% | **83.82%** | — |
| CANINE + Contrastive + Proto (151) | 90.71% | 81.69% | **39.90%** |

**Main finding:** Combining CANINE's token-free encoding with contrastive fine-tuning achieves the highest keyboard noise robustness (83.82%) — 40 points above the BERT baseline and 10 points above BERT + Contrastive — by eliminating the WordPiece tokenization bottleneck and aligning embeddings of clean and noisy utterance pairs simultaneously.
