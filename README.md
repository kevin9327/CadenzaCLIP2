# ICASSP 2027 Grand Challenge: 2nd Cadenza Lyrics Intelligibility Prediction Challenge (CLIP2)

**Team ID:** `T001` (`Kevin So AI Acoustic Lab`)  
**System Classification:** Multimodal Hybrid (`Audio1` + `Audio2` + `Severity` + `Lyrics`)

---

## Overview

This repository contains our full training, feature extraction, cross-validation, and submission pipeline for the **ICASSP 2027 2nd Cadenza Lyrics Intelligibility Prediction Challenge (CLIP2)**:

1. **Binaural Acoustic + Lexical Solver ([`clip2_binaural_stoi_lexical_solver.py`](./clip2_binaural_stoi_lexical_solver.py))**:
   - Extracts 14 binaural psychoacoustic and lexical features across `Audio1` (hearing-loss-simulated stereo audio), `Audio2` (unprocessed reference stereo audio), `Severity` (`No Loss`, `Mild`, `Moderate`), and ground-truth `Lyrics` prompts:
     - **Vocal-Band Dynamic SNR Crest (`450–3,600 Hz`)**: 90th-to-20th percentile short-time energy range (`r = +0.493` with human intelligibility).
     - **Better-Ear Subband Short-Time Envelope Correlation (`Audio1` vs. `Audio2`)**: 5-band Butterworth decomposition (`[200, 450, 900, 1800, 3600, 6800] Hz`) after polyphase resampling and FFT cross-correlation alignment.
     - **Lexical Predictability**: `char_per_word`, `function_word_ratio`, and `log(1 + n_words)`.
2. **Full-Corpus (`N = 20,297`) Prompt-Grouped 5-Fold Cross-Validation & Submission Generator ([`build_real_submissions.py`](./build_real_submissions.py))**:
   - Trained across all `20,297` labeled training scenes from the official CLIP2 v1.0 dataset (`Zenodo Record 23189482`).
   - Empirically calibrates hearing-loss severity priors (`No Loss: 64.33%`, `Mild: 38.45%`, `Moderate: 30.41%`) combined with Severity × Lexical/Phonetic interaction features (`vowel_ratio`, `plosive_ratio`, `fricative_ratio`, `nasal_ratio`, `stop_ratio`, `uniq_ratio`, `syllables_per_word`) and 3,500-dim Word + Character N-gram TF-IDF Ridge Regression (`alpha = 25.0`, prompt-grouped 5-fold CV so zero prompts leak across folds).

---

## Benchmark Results

| System Variant | Modalities | Evaluation Split | RMSE (%) | PCC / Gain |
| :--- | :--- | :---: | :---: | :---: |
| Constant Mean Baseline | None | Stereo Audio Benchmark | `41.25%` | — |
| Binaural Acoustic Only | `Audio1`, `Audio2`, `Severity` | Stereo Audio Benchmark | `31.84%` | `-22.8%` |
| **Binaural STOI-Vocal + Lexical Solver** | **`Audio1`, `Audio2`, `Severity`, `Lyrics`** | **Stereo Audio Benchmark** | **`24.26%`** | **`-41.2%`** |
| `T001b` Severity-Only Ablation | `Severity` | Full `N = 20,297` 5-Fold Prompt-Grouped CV | `35.11%` | `PCC = 0.3596` |
| **`T001` Full Severity × Lexical-Phonetic + TF-IDF Ridge** | **`Severity`, `Lyrics`, `Prompt Stats`** | **Full `N = 20,297` 5-Fold Prompt-Grouped CV** | **`34.01%`** | **`PCC = 0.4242`** |

---

## Official Submission Files

- [`ICASSP2027_valid_T001.csv`](./ICASSP2027_valid_T001.csv) — Official Validation Set (`2,117` scenes) predictions for primary system `T001`.
- [`ICASSP2027_eval_T001.csv`](./ICASSP2027_eval_T001.csv) — Official Evaluation Set (`2,189` scenes) predictions for primary system `T001`.
- [`ICASSP2027_valid_T001b.csv`](./ICASSP2027_valid_T001b.csv) — Official Validation Set (`2,117` scenes) predictions for ablation system `T001b`.
- [`ICASSP2027_eval_T001b.csv`](./ICASSP2027_eval_T001b.csv) — Official Evaluation Set (`2,189` scenes) predictions for ablation system `T001b`.
- [`CLIP2_ICASSP2027_Report.pdf`](./CLIP2_ICASSP2027_Report.pdf) — 2-Page ICASSP 2027 Technical Report (`CLIP2_ICASSP2027_REPORT.md`).
