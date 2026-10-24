import json
import re
from collections import Counter
import numpy as np
import pandas as pd

tr = pd.DataFrame(json.load(open("CadenzaCLIP2/train_metadata.json", encoding="utf-8")))
va = pd.DataFrame(json.load(open("CadenzaCLIP2/valid_metadata.json", encoding="utf-8")))
ev = pd.DataFrame(json.load(open("CadenzaCLIP2/eval_metadata.json", encoding="utf-8")))

y = (tr["correctness"] * 100.0).to_numpy()
print(f"Train samples: {len(tr)}, Valid samples: {len(va)}, Eval samples: {len(ev)}")
print("Global mean baseline RMSE:", round(float(np.sqrt(np.mean((y - y.mean()) ** 2))), 4))

sev_means = tr.groupby("hearing_loss")["correctness"].transform("mean").to_numpy() * 100.0
print("Severity-only baseline RMSE:", round(float(np.sqrt(np.mean((y - sev_means) ** 2))), 4))
print("Severity-only PCC:", round(float(np.corrcoef(y, sev_means)[0, 1]), 4))

# Build corpus word and char-trigram frequencies
all_prompts = pd.concat([tr["prompt"], va["prompt"], ev["prompt"]])
word_counts = Counter()
trigram_counts = Counter()
for p in all_prompts:
    s = str(p).lower().strip()
    for w in re.findall(r"[a-z']+", s):
        word_counts[w] += 1
    padded = f" {s} "
    for i in range(len(padded) - 2):
        trigram_counts[padded[i : i + 3]] += 1

top_words = [w for w, _ in word_counts.most_common(250)]
top_trigrams = [tg for tg, _ in trigram_counts.most_common(200)]


def featurize(df: pd.DataFrame, include_ngrams: bool = True) -> np.ndarray:
    hl = df["hearing_loss"].astype(str)
    hl_no = (hl == "No Loss").astype(float).to_numpy()
    hl_mi = (hl == "Mild").astype(float).to_numpy()
    hl_mo = (hl == "Moderate").astype(float).to_numpy()
    nw = df["n_words"].astype(float).to_numpy()
    log_nw = np.log1p(nw)
    p_cnt = df.groupby("prompt")["signal"].transform("count").astype(float).to_numpy()

    feats = []
    ngram_rows = []
    for orig, prm, n_w in zip(df["original_prompt"], df["prompt"], nw):
        orig_s = str(orig)
        prm_s = str(prm).lower().strip()
        words = re.findall(r"[a-z']+", prm_s)
        w_set = set(words)
        chars = "".join(words)
        n_ch = max(len(chars), 1)
        n_wd = max(len(words), 1)
        cpw = n_ch / n_wd
        has_contraction = float(len(orig_s.split()) != len(prm_s.split()) or "'" in orig_s)
        has_upper_mid = float(any(w[0].isupper() for w in orig_s.split()[1:]) if len(orig_s.split()) > 1 else 0.0)
        vowels = sum(1 for c in chars if c in "aeiouy") / n_ch
        stops = sum(1 for c in chars if c in "ptkbdg") / n_ch
        frics = sum(1 for c in chars if c in "sfvszh") / n_ch
        nasals = sum(1 for c in chars if c in "mn") / n_ch
        liquids = sum(1 for c in chars if c in "lrw") / n_ch
        wf = [np.log1p(word_counts[w]) for w in words]
        mean_wf = float(np.mean(wf)) if wf else 0.0
        min_wf = float(np.min(wf)) if wf else 0.0
        max_wf = float(np.max(wf)) if wf else 0.0
        rare_w = sum(1 for w in words if word_counts[w] <= 5) / n_wd
        short_w = sum(1 for w in words if len(w) <= 3) / n_wd
        long_w = sum(1 for w in words if len(w) >= 7) / n_wd
        uniq_ratio = len(w_set) / n_wd
        feats.append([
            cpw, np.log1p(n_ch), has_contraction, has_upper_mid,
            vowels, stops, frics, nasals, liquids,
            mean_wf, min_wf, max_wf, rare_w, short_w, long_w, uniq_ratio
        ])
        if include_ngrams:
            padded = f" {prm_s} "
            tgs = Counter(padded[i : i + 3] for i in range(len(padded) - 2))
            row_ng = [1.0 if w in w_set else 0.0 for w in top_words] + [float(tgs[tg]) / n_ch for tg in top_trigrams]
            ngram_rows.append(row_ng)

    F = np.array(feats, dtype=float)
    X_base = np.column_stack([
        hl_no, hl_mi, hl_mo, nw, log_nw, p_cnt,
        hl_no * log_nw, hl_mi * log_nw, hl_mo * log_nw,
        F,
        hl_no[:, None] * F,
        hl_mi[:, None] * F,
    ])
    if include_ngrams:
        X_ng = np.array(ngram_rows, dtype=float)
        return np.column_stack([X_base, X_ng])
    return X_base


# 5-fold Prompt-Grouped Cross-Validation
prompts = list(tr["prompt"].unique())
np.random.seed(42)
np.random.shuffle(prompts)
p_to_fold = {p: i % 5 for i, p in enumerate(prompts)}
folds = tr["prompt"].map(p_to_fold).to_numpy()

for mode_name, inc_ng, alpha in [("Ablation (T001b: Severity+Phonetic)", False, 500.0), ("Main Full (T001 / T001a: +Corpus N-Grams)", True, 2500.0)]:
    X_tr = featurize(tr, include_ngrams=inc_ng)
    X_va = featurize(va, include_ngrams=inc_ng)
    X_ev = featurize(ev, include_ngrams=inc_ng)

    mu = X_tr.mean(0)
    sd = X_tr.std(0) + 1e-9
    Xs_tr = (X_tr - mu) / sd
    Xs_va = (X_va - mu) / sd
    Xs_ev = (X_ev - mu) / sd

    oof = np.zeros_like(y)
    pred_va = np.zeros(len(va))
    pred_ev = np.zeros(len(ev))

    for f in range(5):
        va_m = folds == f
        X_t = np.column_stack([np.ones((~va_m).sum()), Xs_tr[~va_m]])
        X_v = np.column_stack([np.ones(va_m.sum()), Xs_tr[va_m]])
        I = np.eye(X_t.shape[1])
        I[0, 0] = 0.0
        w = np.linalg.solve(X_t.T @ X_t + alpha * I, X_t.T @ y[~va_m])
        oof[va_m] = np.clip(X_v @ w, 0.0, 100.0)
        pred_va += np.clip(np.column_stack([np.ones(len(Xs_va)), Xs_va]) @ w, 0.0, 100.0) / 5.0
        pred_ev += np.clip(np.column_stack([np.ones(len(Xs_ev)), Xs_ev]) @ w, 0.0, 100.0) / 5.0

    rmse = float(np.sqrt(np.mean((y - oof) ** 2)))
    pcc = float(np.corrcoef(y, oof)[0, 1])
    print(f"[{mode_name}] 5-Fold Prompt-Grouped CV -> RMSE: {rmse:.4f}, PCC: {pcc:.4f}")
    print(f"  Valid pred summary: mean={pred_va.mean():.2f}, std={pred_va.std():.2f}, min={pred_va.min():.2f}, max={pred_va.max():.2f}")
    print(f"  Eval  pred summary: mean={pred_ev.mean():.2f}, std={pred_ev.std():.2f}, min={pred_ev.min():.2f}, max={pred_ev.max():.2f}")

    df_va_out = pd.DataFrame({"signal_id": va["signal"], "intelligibility_score": np.round(pred_va, 4)})
    df_ev_out = pd.DataFrame({"signal_id": ev["signal"], "intelligibility_score": np.round(pred_ev, 4)})

    if inc_ng:
        df_va_out.to_csv("CadenzaCLIP2/ICASSP2027_valid_T001.csv", index=False)
        df_ev_out.to_csv("CadenzaCLIP2/ICASSP2027_eval_T001.csv", index=False)
        df_va_out.to_csv("CadenzaCLIP2/ICASSP2027_valid_T001a.csv", index=False)
        df_ev_out.to_csv("CadenzaCLIP2/ICASSP2027_eval_T001a.csv", index=False)
        df_va_out.to_csv("CadenzaCLIP2/evalai_valid_submission.csv", index=False, header=False)
    else:
        df_va_out.to_csv("CadenzaCLIP2/ICASSP2027_valid_T001b.csv", index=False)
        df_ev_out.to_csv("CadenzaCLIP2/ICASSP2027_eval_T001b.csv", index=False)

print("Successfully generated all ICASSP2027_valid/eval CSVs!")
