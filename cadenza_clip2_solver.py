#!/usr/bin/env python3
"""
EvalAI & ICASSP 2027 Grand Challenge: CLIP2 — The 2nd Cadenza Lyrics Intelligibility Prediction Challenge
Binaural Better-Ear Subband Envelope-Correlation (STOI-Vocal) + Syllabic Modulation + Lexical Solver.

Supports:
1. Direct execution on unpacked `cadenza_data/clip2/` (`train`, `valid`, `eval` splits)
2. Automatic extraction & verification on `data/cadenza_clip2_data.demo.v1.0.zip`
3. Dual-format CSV generation:
   - `evalai_valid_submission.csv`: headerless `<signal_id>,<intelligibility_score>` in [0, 100] for EvalAI (#2703)
   - `ICASSP2027_valid_T001.csv` / `ICASSP2027_eval_T001.csv`: headered `signal_id,intelligibility_score` in [0, 100]
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, resample_poly


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 12.0) -> np.ndarray:
    """Closed-form L2 Ridge regression weights w = (X^T X + alpha I)^(-1) X^T y."""
    d = X.shape[1]
    A = X.T @ X + alpha * np.eye(d, dtype=np.float64)
    b = X.T @ y
    return np.linalg.solve(A, b)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)) ** 2)))


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _read_flac_or_wav(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Read stereo 44.1 kHz FLAC/WAV file into float32 array of shape (N, 2) resampled to `target_sr`.
    Uses soundfile/scipy or wave/pyarrow/ffmpeg fallback.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        # Fallback: decode via scipy or ffmpeg if soundfile is unavailable
        import subprocess
        cmd = [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(target_sr), "-"
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0 and len(res.stdout) > 0:
            audio = np.frombuffer(res.stdout, dtype=np.float32).reshape(-1, 2)
            return audio, target_sr
        raise RuntimeError(f"Could not decode audio file: {path}")

    if sr != target_sr and len(audio) > 0:
        from math import gcd
        g = gcd(int(sr), int(target_sr))
        up = int(target_sr) // g
        down = int(sr) // g
        left = resample_poly(audio[:, 0], up, down)
        right = resample_poly(audio[:, 1], up, down) if audio.shape[1] > 1 else left
        audio = np.column_stack([left, right]).astype(np.float32)
    return audio, target_sr


def _align_signals(sig_a: np.ndarray, sig_b: np.ndarray, max_shift: int = 640) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align Audio1 (hearing-loss simulated) and Audio2 (unprocessed reference) to compensate
    for hearing-loss simulator group delay, returning equal-length 1D arrays.
    """
    n = min(len(sig_a), len(sig_b))
    if n < 2048:
        return sig_a[:n], sig_b[:n]
    a = sig_a[:n]
    b = sig_b[:n]
    # Estimate lag on central segment via FFT cross-correlation
    mid_start = n // 4
    mid_end = min(mid_start + 32768, n)
    seg_a = a[mid_start:mid_end] - np.mean(a[mid_start:mid_end])
    seg_b = b[mid_start:mid_end] - np.mean(b[mid_start:mid_end])
    m = len(seg_a)
    fft_len = 1 << (2 * m - 1).bit_length()
    xcorr = np.fft.irfft(np.fft.rfft(seg_a, fft_len) * np.conj(np.fft.rfft(seg_b, fft_len)), fft_len)
    # Check lags in [-max_shift, max_shift]
    pos_lags = xcorr[:max_shift + 1]
    neg_lags = xcorr[-max_shift:]
    combined = np.concatenate([neg_lags, pos_lags])
    best_idx = int(np.argmax(combined))
    lag = best_idx - max_shift
    if lag > 0:
        return a[lag:n], b[: n - lag]
    elif lag < 0:
        lag_abs = -lag
        return a[: n - lag_abs], b[lag_abs:n]
    return a, b


def extract_binaural_and_lexical_features(
    audio1_path: Path,
    audio2_path: Path,
    meta: Dict,
    sr: int = 16000,
) -> np.ndarray:
    """
    Extract 14 physically grounded psychoacoustic + binaural STOI + lexical features:
    0: hl_no_loss (binary indicator)
    1: hl_mild (binary indicator)
    2: hl_moderate (binary indicator)
    3: better_ear_stoi_corr (max across L/R of subband temporal envelope correlation Audio1 vs Audio2)
    4: binaural_stoi_mean (mean across L/R of subband temporal envelope correlation)
    5: hf_consonant_retention_db (1.5-6 kHz energy retention in dB between Audio1 and Audio2, better ear)
    6: vocal_mid_to_side_db (center-panned vocal Mid vs Side accompaniment ratio in 300-3400 Hz)
    7: syllabic_4hz_modulation_depth (2-8 Hz envelope modulation index in vocal band)
    8: dynamic_snr_crest_db (90th vs 20th percentile short-time energy range in vocal band)
    9: spectral_centroid_khz (better-ear spectral centroid in kHz)
    10: log_n_words (log(1 + n_words))
    11: char_per_word (average word length in normalized prompt)
    12: stop_fricative_ratio (fraction of high-frequency confusable consonants s,f,t,k,p,th,sh in prompt)
    13: function_word_ratio (fraction of short common function words in prompt)
    """
    hl_str = str(meta.get("hearing_loss", "Moderate")).strip().lower()
    hl_no = 1.0 if "no" in hl_str else 0.0
    hl_mild = 1.0 if "mild" in hl_str else 0.0
    hl_mod = 1.0 if "mod" in hl_str else 0.0

    aud1, _ = _read_flac_or_wav(audio1_path, target_sr=sr)
    aud2, _ = _read_flac_or_wav(audio2_path, target_sr=sr)

    # Subband filterbank edges (Hz) covering vocal fundamentals + formants + consonants
    bands = [(200, 450), (450, 900), (900, 1800), (1800, 3600), (3600, 6800)]
    win_samples = int(0.025 * sr)  # 25 ms short-time energy frame
    hop_samples = int(0.010 * sr)  # 10 ms hop

    ear_corrs = []
    ear_hf_retentions = []
    ear_mod_depths = []
    ear_crests = []
    ear_centroids = []

    for ch in range(2):
        s1_raw = aud1[:, min(ch, aud1.shape[1] - 1)]
        s2_raw = aud2[:, min(ch, aud2.shape[1] - 1)]
        s1, s2 = _align_signals(s1_raw, s2_raw)

        # Compute subband envelope correlations (STOI-like short-time envelope correlation)
        band_corrs = []
        band_e1 = []
        band_e2 = []
        for f_lo, f_hi in bands:
            sos = butter(3, [f_lo / (0.5 * sr), min(0.98, f_hi / (0.5 * sr))], btype="bandpass", output="sos")
            b1 = sosfilt(sos, s1)
            b2 = sosfilt(sos, s2)
            # Frame RMS envelopes
            n_frames = max(1, (len(b1) - win_samples) // hop_samples)
            idx = np.arange(n_frames)[:, None] * hop_samples + np.arange(win_samples)[None, :]
            env1 = np.sqrt(np.mean(b1[idx] ** 2, axis=1) + 1e-12)
            env2 = np.sqrt(np.mean(b2[idx] ** 2, axis=1) + 1e-12)
            # Log-envelope correlation
            le1 = np.log10(env1)
            le2 = np.log10(env2)
            if np.std(le1) > 1e-6 and np.std(le2) > 1e-6:
                c = float(np.corrcoef(le1, le2)[0, 1])
            else:
                c = 1.0 if hl_no > 0.5 else 0.5
            band_corrs.append(np.clip(c, 0.0, 1.0))
            band_e1.append(float(np.mean(env1 ** 2)))
            band_e2.append(float(np.mean(env2 ** 2)))

        ear_corrs.append(float(np.mean(band_corrs)))

        # High-frequency consonant retention (bands 3 & 4: 1800-6800 Hz vs low band 0 & 1)
        hf_ratio_1 = (band_e1[3] + band_e1[4] + 1e-12) / (band_e1[0] + band_e1[1] + 1e-9)
        hf_ratio_2 = (band_e2[3] + band_e2[4] + 1e-12) / (band_e2[0] + band_e2[1] + 1e-9)
        hf_ret_db = 10.0 * np.log10(max(hf_ratio_1 / max(hf_ratio_2, 1e-12), 1e-6))
        ear_hf_retentions.append(float(np.clip(hf_ret_db, -40.0, 10.0)))

        # Vocal formant band (450 - 3600 Hz) syllabic modulation depth (2 - 8 Hz)
        sos_voc = butter(3, [450.0 / (0.5 * sr), 3600.0 / (0.5 * sr)], btype="bandpass", output="sos")
        v1 = sosfilt(sos_voc, s1)
        n_frames = max(4, (len(v1) - win_samples) // hop_samples)
        idx = np.arange(n_frames)[:, None] * hop_samples + np.arange(win_samples)[None, :]
        env_v = np.sqrt(np.mean(v1[idx] ** 2, axis=1) + 1e-12)
        # FFT of 100 Hz frame envelope to get 2-8 Hz syllabic modulation power
        env_zero = env_v - np.mean(env_v)
        spec_env = np.abs(np.fft.rfft(env_zero)) ** 2
        freqs_env = np.fft.rfftfreq(len(env_zero), d=0.010)
        syl_mask = (freqs_env >= 2.0) & (freqs_env <= 8.0)
        mod_depth = float(np.sum(spec_env[syl_mask]) / (np.sum(spec_env) + 1e-9)) if np.any(syl_mask) else 0.3
        ear_mod_depths.append(mod_depth)

        # Dynamic crest (90th vs 20th percentile in dB)
        p90, p20 = np.percentile(10.0 * np.log10(env_v ** 2 + 1e-12), [90, 20])
        ear_crests.append(float(p90 - p20))

        # Spectral centroid in kHz
        mag = np.abs(np.fft.rfft(s1))
        f_axis = np.fft.rfftfreq(len(s1), d=1.0 / sr) / 1000.0
        centroid = float(np.sum(f_axis * mag) / (np.sum(mag) + 1e-9))
        ear_centroids.append(centroid)

    # Mid-to-Side vocal ratio on Audio1
    mid_sig = 0.5 * (aud1[:, 0] + aud1[:, 1])
    side_sig = 0.5 * (aud1[:, 0] - aud1[:, 1])
    sos_ms = butter(3, [300.0 / (0.5 * sr), 3400.0 / (0.5 * sr)], btype="bandpass", output="sos")
    mid_e = float(np.mean(sosfilt(sos_ms, mid_sig) ** 2) + 1e-12)
    side_e = float(np.mean(sosfilt(sos_ms, side_sig) ** 2) + 1e-12)
    vocal_ms_db = float(np.clip(10.0 * np.log10(mid_e / side_e), -15.0, 25.0))

    # Lexical features from ground-truth lyrics prompt
    prompt = str(meta.get("prompt", meta.get("original_prompt", ""))).lower().strip()
    words = re.findall(r"[a-z']+", prompt)
    n_words = float(meta.get("n_words", max(len(words), 1)))
    log_n_words = float(np.log1p(n_words))
    chars = "".join(words)
    char_per_word = float(len(chars) / max(len(words), 1))
    confusable_cnt = sum(1 for c in chars if c in "sftkpvz")
    stop_fric_ratio = float(confusable_cnt / max(len(chars), 1))
    func_words = {"the", "a", "an", "to", "in", "on", "of", "for", "with", "my", "you", "me", "i", "it", "is", "are", "and", "that"}
    func_ratio = float(sum(1 for w in words if w in func_words) / max(len(words), 1))

    return np.array([
        hl_no,
        hl_mild,
        hl_mod,
        max(ear_corrs),
        float(np.mean(ear_corrs)),
        max(ear_hf_retentions),
        vocal_ms_db,
        max(ear_mod_depths),
        max(ear_crests),
        max(ear_centroids),
        log_n_words,
        char_per_word,
        stop_fric_ratio,
        func_ratio,
    ], dtype=np.float64)


def physics_prior_score_0_to_1(feat: np.ndarray) -> float:
    """
    Calibrated psychoacoustic prior mapping extracted binaural & lexical features
    to word recognition rate in [0, 1].
    """
    hl_no, hl_mild, hl_mod = feat[0], feat[1], feat[2]
    better_stoi = feat[3]
    hf_ret_db = feat[5]
    crest_db = feat[8]
    centroid_khz = feat[9]
    log_n_words = feat[10]
    char_per_word = feat[11]
    func_ratio = feat[13]

    # Base intelligibility anchor
    base = 0.45 + 0.06 * hl_mild + 0.02 * hl_no - 0.02 * hl_mod
    # Vocal dynamic SNR crest (90th vs 20th percentile in 450-3600 Hz vocal band) & accompaniment masking
    vocal_snr_term = 0.018 * np.clip(crest_db - 14.0, -10.0, 20.0) - 0.14 * np.clip(centroid_khz - 1.30, -0.7, 1.5)
    # Binaural STOI + HF retention
    binaural_term = 0.10 * (better_stoi - 0.95) + 0.004 * np.clip(hf_ret_db + 7.0, -10.0, 10.0)
    # Lexical predictability: shorter words, higher function-word familiarity, richer sentence context
    lex_term = (
        -0.22 * (char_per_word - 4.0)
        + 0.32 * (func_ratio - 0.29)
        + 0.35 * (log_n_words - 2.07)
    )

    return float(np.clip(base + vocal_snr_term + binaural_term + lex_term, 0.02, 0.98))


def ensure_data_unpacked(root_dir: Path) -> Path:
    """Ensure demo or full dataset is unpacked and return the `clip2` directory path."""
    for candidate in [root_dir / "cadenza_data" / "clip2", root_dir / "demo_data" / "clip2", DATA_DIR / "demo_data" / "clip2"]:
        if (candidate / "metadata" / "train_metadata.json").exists():
            return candidate

    demo_zip = DATA_DIR / "cadenza_clip2_data.demo.v1.0.zip"
    if demo_zip.exists():
        with zipfile.ZipFile(demo_zip, "r") as zf:
            zf.extractall(DATA_DIR)
        return DATA_DIR / "demo_data" / "clip2"

    raise FileNotFoundError(f"Could not find clip2 data directory or {demo_zip}")


def run_pipeline(data_root: Path, team_id: str = "T001") -> None:
    clip2_dir = ensure_data_unpacked(data_root)
    meta_dir = clip2_dir / "metadata"

    train_meta = json.loads((meta_dir / "train_metadata.json").read_text(encoding="utf-8"))
    valid_meta = json.loads((meta_dir / "valid_metadata.json").read_text(encoding="utf-8"))

    print(f"[CLIP2] Loaded train_metadata ({len(train_meta)} samples) and valid_metadata ({len(valid_meta)} samples) from {clip2_dir}")

    X_train = []
    y_train = []
    for item in train_meta:
        sig = item["signal"]
        p1 = clip2_dir / "train" / "signals" / f"{sig}.flac"
        p2 = clip2_dir / "train" / "unprocessed" / f"{sig}_unproc.flac"
        feat = extract_binaural_and_lexical_features(p1, p2, item)
        X_train.append(feat)
        y_train.append(float(item["correctness"]) * 100.0)

    X_train_arr = np.vstack(X_train)
    y_train_arr = np.array(y_train, dtype=np.float64)

    # Leave-One-Out CV on training set comparing Constant Mean Baseline vs Binaural Psychoacoustic Solver
    loo_const_preds = []
    loo_our_preds = []
    alpha_reg = 45.0 if len(y_train_arr) < 50 else 15.0
    ridge_weight = 0.25 if len(y_train_arr) < 50 else 0.60
    for i in range(len(y_train_arr)):
        mask = np.ones(len(y_train_arr), dtype=bool)
        mask[i] = False
        loo_const_preds.append(float(np.mean(y_train_arr[mask])))

        # Fit Ridge residual model on standardized key features
        mu = np.mean(X_train_arr[mask], axis=0)
        sd = np.std(X_train_arr[mask], axis=0) + 1e-6
        X_tr_norm = (X_train_arr[mask] - mu) / sd
        X_te_norm = (X_train_arr[i : i + 1] - mu) / sd

        prior_tr = np.array([physics_prior_score_0_to_1(f) * 100.0 for f in X_train_arr[mask]])
        prior_te = physics_prior_score_0_to_1(X_train_arr[i]) * 100.0

        w_ridge = _fit_ridge(X_tr_norm, y_train_arr[mask] - prior_tr, alpha=alpha_reg)
        pred_i = float(np.clip(prior_te + ridge_weight * float((X_te_norm @ w_ridge)[0]), 0.0, 100.0))
        loo_our_preds.append(pred_i)

    rmse_const = _rmse(y_train_arr, loo_const_preds)
    rmse_ours = _rmse(y_train_arr, loo_our_preds)
    print(f"[Validation LOO-CV] Mean Baseline RMSE: {rmse_const:.2f}% | Binaural STOI-Lexical Solver RMSE: {rmse_ours:.2f}%")

    # Fit full model on all training samples
    mu_all = np.mean(X_train_arr, axis=0)
    sd_all = np.std(X_train_arr, axis=0) + 1e-6
    X_all_norm = (X_train_arr - mu_all) / sd_all
    prior_all = np.array([physics_prior_score_0_to_1(f) * 100.0 for f in X_train_arr])
    w_full = _fit_ridge(X_all_norm, y_train_arr - prior_all, alpha=alpha_reg)

    # Predict on validation set
    valid_rows_no_header = []
    valid_rows_with_header = ["signal_id,intelligibility_score"]

    for item in valid_meta:
        sig = item["signal"]
        p1 = clip2_dir / "valid" / "signals" / f"{sig}.flac"
        p2 = clip2_dir / "valid" / "unprocessed" / f"{sig}_unproc.flac"
        feat = extract_binaural_and_lexical_features(p1, p2, item)
        feat_norm = (feat - mu_all) / sd_all
        prior_val = physics_prior_score_0_to_1(feat) * 100.0
        pred_val = float(np.clip(prior_val + ridge_weight * float(feat_norm @ w_full), 0.0, 100.0))
        valid_rows_no_header.append(f"{sig},{pred_val:.4f}")
        valid_rows_with_header.append(f"{sig},{pred_val:.4f}")
        print(f"  [Valid] signal={sig} | HL={item.get('hearing_loss'):8s} | n_words={item.get('n_words')} -> pred={pred_val:.2f}%")

    evalai_csv = BASE_DIR / "evalai_valid_submission.csv"
    icassp_valid_csv = BASE_DIR / f"ICASSP2027_valid_{team_id}.csv"
    evalai_csv.write_text("\n".join(valid_rows_no_header) + "\n", encoding="utf-8")
    icassp_valid_csv.write_text("\n".join(valid_rows_with_header) + "\n", encoding="utf-8")
    print(f"[Saved] {evalai_csv} (EvalAI headerless format, {evalai_csv.stat().st_size} bytes)")
    print(f"[Saved] {icassp_valid_csv} (ICASSP 2027 headered format, {icassp_valid_csv.stat().st_size} bytes)")

    # If eval_metadata.json is present, also generate ICASSP2027_eval_<team_id>.csv
    eval_meta_path = meta_dir / "eval_metadata.json"
    if eval_meta_path.exists():
        eval_meta = json.loads(eval_meta_path.read_text(encoding="utf-8"))
        eval_rows = ["signal_id,intelligibility_score"]
        for item in eval_meta:
            sig = item["signal"]
            p1 = clip2_dir / "eval" / "signals" / f"{sig}.flac"
            p2 = clip2_dir / "eval" / "unprocessed" / f"{sig}_unproc.flac"
            feat = extract_binaural_and_lexical_features(p1, p2, item)
            feat_norm = (feat - mu_all) / sd_all
            prior_val = physics_prior_score_0_to_1(feat) * 100.0
            pred_val = float(np.clip(prior_val + ridge_weight * float(feat_norm @ w_full), 0.0, 100.0))
            eval_rows.append(f"{sig},{pred_val:.4f}")
        icassp_eval_csv = BASE_DIR / f"ICASSP2027_eval_{team_id}.csv"
        icassp_eval_csv.write_text("\n".join(eval_rows) + "\n", encoding="utf-8")
        print(f"[Saved] {icassp_eval_csv} ({len(eval_meta)} eval predictions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cadenza CLIP2 Binaural Intelligibility Solver")
    parser.add_argument("--data-root", type=Path, default=BASE_DIR, help="Path to unpacked cadenza_data or demo_data")
    parser.add_argument("--team-id", type=str, default="T001", help="Team ID for ICASSP2027 CSV filenames")
    args = parser.parse_args()
    run_pipeline(args.data_root, args.team_id)
