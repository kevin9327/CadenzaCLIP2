# EvalAI & ICASSP 2027 Grand Challenge — CLIP2: The 2nd Cadenza Lyrics Intelligibility Prediction Challenge
**Official EvalAI Portal:** [`https://eval.ai/web/challenges/challenge-page/2703/overview`](https://eval.ai/web/challenges/challenge-page/2703/overview)
**Official Documentation:** [`https://cadenzachallenge.org/docs/clip2/intro`](https://cadenzachallenge.org/docs/clip2/intro)
**Registration Form (for TEAM_ID):** [`https://cadenzachallenge.org/docs/clip2/take_part/registration`](https://cadenzachallenge.org/docs/clip2/take_part/registration)
**Final Package Upload Form:** [`https://forms.gle/NPAF6w5jcfghBGVR7`](https://forms.gle/NPAF6w5jcfghBGVR7)
**Submission Deadline:** October 30 / November 1, 2026 (Top 5 teams invited to submit 2-page ICASSP 2027 Grand Challenge paper in Toronto, Canada)

---

## 1. Critical Reverse-Engineered Rules & Dual CSV Format Contract

Per `cadenzachallenge.org/docs/clip2/take_part/rules`, `leaderboard`, and `submission`:
1. **Inputs Explicitly Allowed at Validation & Evaluation Time:**
   - `Audio1` (`<SPLIT>/signals/<signal_id>.flac`): Stereo 44.1 kHz audio with HOA+HRTF reverberation, background noise, and simulated hearing loss (`No Loss`, `Mild`, or `Moderate`).
   - `Audio2` (`<SPLIT>/unprocessed/<signal_id>_unproc.flac`): Stereo 44.1 kHz audio at the ear-canal entrance **without** the hearing loss simulation!
   - `Severity` (`hearing_loss` field in `valid_metadata.json` / `eval_metadata.json`): `"No Loss"`, `"Mild"`, or `"Moderate"`.
   - `Lyrics` (`prompt` and `n_words` fields in `valid_metadata.json` / `eval_metadata.json`): Ground-truth normalized English lyric text!
2. **Scale Trap (`0 to 100` vs `0 to 1`):**
   - Although `train_metadata.json` stores `correctness` in `[0.0, 1.0]`, **both the EvalAI Validation Leaderboard and the ICASSP 2027 Final CSV require `intelligibility_score` in percentage points `[0, 100]`**! Submitting `[0, 1]` on EvalAI results in an catastrophic RMSE of `~55.0`.
3. **Header Trap (EvalAI vs Final Submission):**
   - **EvalAI Validation Leaderboard (`evalai_valid_submission.csv`):** Must **NOT** include CSV headers (`<signal_id>,<intelligibility_score>` starting on line 1, no spaces after comma).
   - **ICASSP 2027 Final Submission (`ICASSP2027_valid_<TEAM_ID>.csv` & `ICASSP2027_eval_<TEAM_ID>.csv`):** **MUST** include the 2-column header `signal_id,intelligibility_score`.

---

## 2. Architecture of [`cadenza_clip2_solver.py`](file:///c:/Users/swsz9/Downloads/WindowexeHideWindow_2026-03-21/Distributables/CadenzaCLIP2/cadenza_clip2_solver.py)

Unlike the official `Baseline-Mono` (which downmixes stereo HRTF audio to mono and ignores `Audio2`, `Severity`, and `Lyrics`), our solver exploits all four allowed evaluation modalities:
1. **FFT Cross-Correlation Delay Alignment + Better-Ear Subband Envelope Correlation (`better_ear_stoi_corr`):**
   Compensates for the hearing-loss simulator group delay between `Audio1` and `Audio2`, then computes short-time (25 ms window, 10 ms hop) log-envelope correlations across 5 vocal/formant/consonant bands (`200–6800 Hz`) on Left and Right channels separately, taking the **Better-Ear maximum** to preserve HRTF head-shadow advantage.
2. **High-Frequency Consonantal Retention (`hf_consonant_retention_db`, `1.8–6.8 kHz`):**
   Measures the exact dB preservation of fricatives and stops (`/s/, /f/, /t/, /k/, /p/`) relative to low-frequency vowel energy between `Audio1` and `Audio2`.
3. **Vocal Mid-to-Side Ratio & 2–8 Hz Syllabic Modulation Depth:**
   Separates center-panned lead vocals ($M = (L+R)/2$) from wide stereo accompaniment ($S = (L-R)/2$) and computes the $2\text{--}8\text{ Hz}$ modulation transfer ratio.
4. **Lexical & Phonetic Prompt Priors:**
   Extracts `log_n_words`, `char_per_word`, confusable stop/fricative density, and common function-word familiarity from `prompt`.
