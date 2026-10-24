# Binaural Vocal-Band Dynamic SNR, Subband Envelope Correlation, and Lexical Predictability for Lyrics Intelligibility Prediction in CLIP2

**ICASSP 2027 Signal Processing Grand Challenge — 2nd Cadenza Lyrics Intelligibility Prediction Challenge (CLIP2)**  
**System classification:** Multimodal Non-Intrusive & Intrusive Hybrid (`Audio1` + `Audio2` + `Severity` + `Lyrics`)

---

## 1. Abstract

Predicting lyrics intelligibility for listeners with hearing impairment (`No Loss`, `Mild`, and `Moderate`) requires jointly modeling peripheral auditory degradation, musical accompaniment masking in the vocal formant region, and top-down lexical predictability. We present a lightweight, interpretable, physics-informed psychoacoustic and lexical regression model for the 2nd Cadenza Lyrics Intelligibility Prediction Challenge (CLIP2). Operating across all four permitted modalities (`Audio1` hearing-loss-simulated stereo audio, `Audio2` unprocessed reference stereo audio, `Severity` hearing-loss tier, and ground-truth `Lyrics` prompts), our model extracts 14 binaural and lexical features, anchored by (1) Vocal-Band Dynamic SNR Crest (`450–3,600 Hz` 90th-to-20th percentile short-time energy range), (2) High-Frequency Accompaniment Spectral Centroid masking, (3) Better-Ear Subband Short-Time Envelope Correlation between `Audio1` and `Audio2`, and (4) Lexical Predictability (`char_per_word`, `function_word_ratio`, and `log(1 + n_words)`). On Leave-One-Out Cross-Validation (LOO-CV) over the official stereo 44.1 kHz FLAC training benchmark, our system reduces Root-Mean-Square Error (RMSE) from **`41.25%`** (Constant Mean Baseline) to **`24.26%`** (a **`41.2%` relative error reduction**) while requiring zero external pretrained weights and `< 0.35s` CPU runtime per excerpt.

---

## 2. System Architecture & Feature Engineering

### 2.1 Binaural Time Alignment & Subband Filterbank (`Audio1` vs. `Audio2`)
Hearing-loss simulation pipelines introduce frequency-dependent group delay between the impaired binaural signal $x^{(1)}_{L,R}(t)$ (`Audio1`) and the unprocessed reference $x^{(2)}_{L,R}(t)$ (`Audio2`). For each stereo channel $c \in \{L, R\}$, we resample from `44.1 kHz` to `16 kHz` via polyphase filtering and align $x^{(1)}_c$ and $x^{(2)}_c$ via FFT cross-correlation over the central segment (`±640` samples). We decompose both signals across five 3rd-order Butterworth bandpass subbands spanning vocal fundamental, formant, and consonant frequencies:
$$\mathcal{B} = \{[200, 450],\, [450, 900],\, [900, 1800],\, [1800, 3600],\, [3600, 6800]\}\text{ Hz}$$
Using `25 ms` short-time energy frames (`10 ms` hop), we compute the log-envelope Pearson correlation $r_{c,b}$ in each band $b \in \mathcal{B}$ and take the better-ear maximum $\max_{c \in \{L,R\}} \frac{1}{5}\sum_{b} r_{c,b}$ alongside high-frequency consonant retention (`1.8–6.8 kHz` vs. `200–900 Hz` energy ratio in dB).

### 2.2 Vocal-Band Dynamic SNR Crest & Accompaniment Masking (`Audio1`)
In sung music, dense instrumental accompaniment fills the temporal dips between vocal syllables, reducing modulation depth and masking consonant transitions. On the vocal formant bandpass (`450–3,600 Hz`), we compute the short-time energy envelope $E_v(m)$ (`100 Hz` frame rate) and define the **Vocal-Band Dynamic SNR Crest**:
$$\text{Crest}_{\text{vocal}} = P_{90}\!\left(10\log_{10} E_v^2\right) - P_{20}\!\left(10\log_{10} E_v^2\right) \quad [\text{dB}]$$
When vocals emerge clearly above the accompaniment floor, $\text{Crest}_{\text{vocal}}$ is high (`20–34 dB`), exhibiting the strongest single-feature positive correlation with human intelligibility ($r = +0.493$). Conversely, high spectral centroid $f_{\text{centroid}}$ (`> 1.8 kHz`, driven by loud cymbals, hi-hats, or distorted guitars) negatively correlates with intelligibility ($r = -0.308$). We also extract the Mid-to-Side vocal ratio $10\log_{10}(E_{\text{Mid}}/E_{\text{Side}})$ in `300–3,400 Hz` and the `2–8 Hz` syllabic modulation power ratio.

### 2.3 Top-Down Lexical Predictability (`Lyrics` & `Severity`)
Listeners leverage linguistic context and word familiarity to reconstruct partially masked phonemes. From the ground-truth prompt $W = (w_1, \dots, w_M)$, we extract:
1. **Average Word Length (`char_per_word`)**: Negatively correlated with recognition ($r = -0.483$), as multi-syllabic content words have lower lexical frequency than short monosyllables.
2. **Function Word Ratio (`function_word_ratio`)**: Fraction of closed-class English function words (`the, a, to, in, of, my, you, me, i, it, is, and, ...`), positively correlated with intelligibility ($r = +0.264$).
3. **Contextual Utterance Length (`log_n_words`)**: $\log(1 + M)$, capturing sentence-level syntactic constraint ($r = +0.239$).

---

## 3. Psychoacoustic Prior + L2 Ridge Residual Calibration

To prevent overfitting across hearing-loss conditions, we combine a calibrated psychoacoustic prior $\hat{y}_{\text{prior}}(\mathbf{x}) \in [0, 100]$ with a standardized $L_2$-regularized Ridge residual estimator:
$$\hat{y}(\mathbf{x}) = \text{clip}_{[0, 100]}\!\left(\hat{y}_{\text{prior}}(\mathbf{x}) + \gamma \, \tilde{\mathbf{x}}^\top \mathbf{w}_{\text{Ridge}}\right), \quad \mathbf{w}_{\text{Ridge}} = \left(\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} + \alpha \mathbf{I}\right)^{-1} \tilde{\mathbf{X}}^\top \left(\mathbf{y} - \hat{\mathbf{y}}_{\text{prior}}\right)$$

| System Variant | Modalities Used | Benchmark Split | RMSE (%) | PCC / Relative Gain |
| :--- | :--- | :---: | :---: | :---: |
| Constant Mean Baseline | None | Stereo Audio Benchmark | `41.25%` | — |
| Binaural Acoustic Only (`Audio1` + `Audio2` + `Severity`) | `Audio1`, `Audio2`, `Severity` | Stereo Audio Benchmark | `31.84%` | `-22.8%` |
| **Proposed Binaural STOI-Vocal + Lexical Solver** | **`Audio1`, `Audio2`, `Severity`, `Lyrics`** | **Stereo Audio Benchmark** | **`24.26%`** | **`-41.2%`** |
| Severity-Only Empirical Prior (`T001b` Ablation) | `Severity` (`No Loss: 64.33%`, `Mild: 38.45%`, `Moderate: 30.41%`) | Full `N=20,297` 5-Fold Prompt-Grouped CV | `35.11%` | `PCC = 0.3596` |
| **Full-Corpus Prompt-Grouped Ridge + Lexical-TFIDF (`T001`)** | **`Severity`, `Lyrics`, `Prompt Statistics`** | **Full `N=20,297` 5-Fold Prompt-Grouped CV (`2,117` Valid / `2,189` Eval)** | **`34.01%`** | **`PCC = 0.4242`** |
