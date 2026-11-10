"""
Ödev 6 — Kendi açıklamalarınızı değerlendirin
---------------------------------------------
Model: Ödev 2'deki LightGBM (log-odds). Yöntemler: SHAP-müd, SHAP-yol, LIME (kendi), Örtme.
Kontroller: Rastgele açıklama, Ort-uzaklık (modelden bağımsız: ortalamadan z-uzaklığı).
Aynı arka plan B (dev'den 200 satır) SHAP-müd, LIME ve Örtme'de ortaktır.

  1) Akıl sağlığı : kademeli yaprak rastgeleleştirme (%25/50/75/100, 3 tohum) ve karıştırılmış
                    etiketlerle yeniden eğitilmiş model (3 tohum); kontrol: Rastgele, Ort-uzaklık
  2) Sadakat      : en önemli k özniteliği dolguyla değiştirme eğrileri; 3 dolgu (ortalama,
                    rastgele arka plan satırı, yakın komşu); 2 ölçüt (MoRF alanı, PGI@3);
                    yöntem sıralamalarının dolgular arası uyumu Kendall τ (+ önyükleme)
  3) Kararlılık   : kararı değiştirmeyen küçük gürültü (ε = 0.01, 0.05 sd); açıklama değişimi,
                    tahmin değişimi ve model duyarlılığı (geçilen bölünme eşiği) birlikte
  4) Uyuşma       : ilk-1, ilk-3, sıra korelasyonu, işaret uyumu; en çok uyuşmayan çift için
                    deneyler: E1 toplamsal (etkileşimsiz) model, E2 LIME çekirdeği, E3 LIME tohum gürültüsü
  5) Özet tablo

Çalıştırma:
  python odev6_evaluate.py           # önce odev2_rashomon.py çalışmış olmalı
  python odev6_evaluate.py --fast    # duman testi
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from src.data import FEATURES, PAY_COLS, ROOT, SEED, get_dev_holdout
from src.explainers import (CONTROLS, METHODS, LeafRandomizer, agreement, explain_all, lime, occlusion,
                            random_attr, raw_predict, rel_l2, shap_interventional, split_thresholds,
                            threshold_crossings, zscore)
from src.models import build_lgbm

warnings.filterwarnings("ignore")

COL = {"SHAP-müd": "#2a78d6", "SHAP-yol": "#eb6834", "LIME": "#1baf7a", "Örtme": "#eda100",
       "Rastgele": "#8a8984", "Ort-uzaklık": "#52514e"}
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": FAINT, "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})
CONT = ["LIMIT_BAL", "AGE"] + [f"BILL_AMT{k}" for k in range(1, 7)] + [f"PAY_AMT{k}" for k in range(1, 7)]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--data", default=None)
    return ap.parse_args()


def summarize(values: dict, extra: dict) -> list[dict]:
    """{ölçüt: dizi} → ortalama, sd, n satırları."""
    rows = []
    for k, v in values.items():
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        rows.append({**extra, "ölçüt": k, "ort": float(v.mean()) if len(v) else np.nan,
                     "sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan, "n": int(len(v))})
    return rows


def all_phi(booster, X, B, seeds, lime_samples, mu, sd, rnd_seed):
    phi, bases = explain_all(booster, X, B, seeds, lime_samples)
    phi["Rastgele"] = random_attr(len(X), X.shape[1], rnd_seed)
    phi["Ort-uzaklık"] = zscore(X, mu, sd)
    return phi, bases


# ------------------------------------------------------------------ 4) uyuşma
def step_agreement(phi, out, fig):
    names = METHODS + CONTROLS
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows += summarize(agreement(phi[a], phi[b]), {"A": a, "B": b})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "agreement.csv", index=False)
    piv = {m: tab[tab["ölçüt"] == m].pivot(index="A", columns="B", values="ort") for m in ("spearman_abs", "ilk3", "ilk1")}
    # en çok uyuşmayan YÖNTEM çifti (kontroller hariç), |φ| sıra korelasyonuna göre
    meth = tab[tab["A"].isin(METHODS) & tab["B"].isin(METHODS) & (tab["ölçüt"] == "spearman_abs")]
    worst = meth.sort_values("ort").iloc[0]
    # ısı haritaları
    f, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for ax, (m, title) in zip(axes, [("spearman_abs", "|φ| sıra korelasyonu (Spearman)"),
                                     ("ilk3", "ilk-3 örtüşmesi"), ("ilk1", "ilk-1 uyumu")]):
        M = np.full((len(names), len(names)), np.nan)
        for _, r in tab[tab["ölçüt"] == m].iterrows():
            i, j = names.index(r["A"]), names.index(r["B"])
            M[i, j] = M[j, i] = r["ort"]
        np.fill_diagonal(M, 1.0)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(names)), names if ax is axes[0] else [""] * len(names), fontsize=6.5)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.grid(False)
    f.tight_layout()
    f.savefig(fig / "agreement.png")
    plt.close(f)
    return tab, (worst["A"], worst["B"], float(worst["ort"]))


# ------------------------------------------------------------------ 2) sadakat
def fill_rows_for(fill, X, Xdev, R, rng, nn=None, mu=None):
    n, d = X.shape
    if fill == "ortalama":
        return np.broadcast_to(mu, (n, 1, d)).copy()
    if fill == "rastgele":
        return Xdev[rng.integers(0, len(Xdev), (n, R))]
    if fill == "yakın-komşu":
        return Xdev[nn]                                      # (n, R, d)
    raise ValueError(fill)


def replace_curve(predict, x, order, fills):
    """F[k] = ort_r f(x, ilk k öznitelik ← fills[r]),  k = 0..d."""
    d = len(x)
    pos = np.empty(d, int)
    pos[order] = np.arange(d)
    mask = pos[None, :] < np.arange(d + 1)[:, None]          # (d+1, d)
    rows = np.where(mask[:, None, :], fills[None, :, :], x[None, None, :])
    return predict(rows.reshape(-1, d)).reshape(d + 1, len(fills)).mean(1)


def step_fidelity(predict, phi, X, Xdev, mu, sdv, R, n_rand, out, fig, rng):
    n, d = X.shape
    Z = zscore(Xdev, mu, sdv)
    nnm = NearestNeighbors(n_neighbors=R).fit(Z)
    nn = nnm.kneighbors(zscore(X, mu, sdv), return_distance=False)
    fills = {f: fill_rows_for(f, X, Xdev, R, rng, nn, mu) for f in ("ortalama", "rastgele", "yakın-komşu")}
    names = METHODS + CONTROLS
    curves, per = {}, []
    for fname, FR in fills.items():
        base = np.array([predict(FR[i]).mean() for i in range(n)])  # tüm öznitelikler değişince
        fx = predict(X)
        s = np.sign(fx - base)
        for m in names:
            H = np.empty((n, d + 1))
            pgi = np.empty(n)
            for i in range(n):
                if m == "Rastgele":
                    # rastgele sıra kontrolü: n_rand rastgele sıranın ortalaması
                    Fs = [replace_curve(predict, X[i], rng.permutation(d), FR[i]) for _ in range(n_rand)]
                    F = np.mean(Fs, 0)
                    Fm = F
                elif m == "Ort-uzaklık":
                    F = Fm = replace_curve(predict, X[i], np.argsort(-np.abs(phi[m][i])), FR[i])
                else:
                    # MoRF: kestirimi tabandan uzaklaştıran (destekleyen) öznitelikler önce
                    F = replace_curve(predict, X[i], np.argsort(-(s[i] * phi[m][i])), FR[i])
                    Fm = replace_curve(predict, X[i], np.argsort(-np.abs(phi[m][i])), FR[i])
                H[i] = s[i] * (F - F[-1])
                pgi[i] = abs(Fm[0] - Fm[3])
            curves[(fname, m)] = H
            area = H.mean(1)                                   # k = 0..d ortalaması (düşük = sadık)
            per.append(pd.DataFrame({"dolgu": fname, "yöntem": m, "örnek": np.arange(n),
                                     "MoRF_alanı": area, "PGI@3": pgi, "başlangıç_açığı": H[:, 0]}))
    per = pd.concat(per, ignore_index=True)
    per.to_csv(out / "fidelity_per_instance.csv", index=False)
    summ = per.groupby(["dolgu", "yöntem"])[["MoRF_alanı", "PGI@3"]].agg(["mean", "std", "count"])
    summ.to_csv(out / "fidelity_summary.csv")
    # yöntem sıralamaları ve Kendall τ (dolgular arası, ölçütler arası) + örnek önyüklemesi
    def ranking(df, metric):
        mm = df.groupby(["dolgu", "yöntem"])[metric].mean().unstack(0)
        return mm if metric == "PGI@3" else -mm             # yüksek = iyi olacak biçimde
    taus = []
    fnames = list(fills)
    # eşli farklar (aynı örnek): yöntem − SHAP-müd; ort, sd, %95 GA ve "daha iyi" oranı
    paired = []
    for fname in fnames:
        for metric, low_better in (("MoRF_alanı", True), ("PGI@3", False)):
            ref = per[(per["dolgu"] == fname) & (per["yöntem"] == "SHAP-müd")].set_index("örnek")[metric]
            for m in names:
                if m == "SHAP-müd":
                    continue
                v = per[(per["dolgu"] == fname) & (per["yöntem"] == m)].set_index("örnek")[metric]
                dl = (v - ref).to_numpy()
                se = dl.std(ddof=1) / np.sqrt(len(dl))
                paired.append({"dolgu": fname, "ölçüt": metric, "yöntem": m, "fark_ort": dl.mean(),
                               "fark_sd": dl.std(ddof=1), "GA95_alt": dl.mean() - 1.96 * se,
                               "GA95_üst": dl.mean() + 1.96 * se,
                               "SHAP_müd'den_iyi_oranı": float(np.mean(dl < 0 if low_better else dl > 0))})
    pd.DataFrame(paired).to_csv(out / "fidelity_paired.csv", index=False)
    for metric in ("MoRF_alanı", "PGI@3"):
        full = ranking(per, metric)
        boots = {(a, b): [] for i, a in enumerate(fnames) for b in fnames[i + 1:]}
        for _ in range(200):
            idx = rng.integers(0, n, n)
            sub = per[per["örnek"].isin(idx)]
            sub = sub.merge(pd.DataFrame({"örnek": idx}), on="örnek")   # tekrarları koru
            rk = ranking(sub, metric)
            for (a, b) in boots:
                boots[(a, b)].append(kendalltau(rk[a], rk[b]).statistic)
        for (a, b), v in boots.items():
            taus.append({"ölçüt": metric, "dolgu_A": a, "dolgu_B": b,
                         "tau": kendalltau(full[a], full[b]).statistic,
                         "tau_yalnız_yöntemler": kendalltau(full.loc[METHODS, a], full.loc[METHODS, b]).statistic,
                         "tau_önyükleme_ort": float(np.mean(v)), "tau_önyükleme_sd": float(np.std(v, ddof=1)),
                         "sıra_A": " > ".join(full[a].sort_values(ascending=False).index),
                         "sıra_B": " > ".join(full[b].sort_values(ascending=False).index)})
    for fname in fnames:                                       # aynı dolguda iki ölçüt arası
        a, b = ranking(per, "MoRF_alanı")[fname], ranking(per, "PGI@3")[fname]
        taus.append({"ölçüt": "MoRF↔PGI@3", "dolgu_A": fname, "dolgu_B": fname,
                     "tau": kendalltau(a, b).statistic,
                     "tau_yalnız_yöntemler": kendalltau(a.loc[METHODS], b.loc[METHODS]).statistic,
                     "tau_önyükleme_ort": np.nan, "tau_önyükleme_sd": np.nan,
                     "sıra_A": " > ".join(a.sort_values(ascending=False).index),
                     "sıra_B": " > ".join(b.sort_values(ascending=False).index)})
    taus = pd.DataFrame(taus)
    taus.to_csv(out / "fidelity_kendall.csv", index=False)
    # şekil: dolgu başına MoRF eğrileri
    f, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=False)
    for ax, fname in zip(axes, fnames):
        for m in names:
            H = curves[(fname, m)]
            ls = "--" if m == "Rastgele" else (":" if m == "Ort-uzaklık" else "-")
            ax.plot(range(d + 1), H.mean(0), ls, color=COL[m], lw=1.5 if m in METHODS else 1.2, label=m)
        ax.set_title(f"dolgu: {fname}", fontsize=8, color=INK)
        ax.set_xlabel("değiştirilen öznitelik sayısı k")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("kalan kanıt (log-odds)")
    h, l = axes[0].get_legend_handles_labels()
    f.tight_layout(rect=(0, 0.08, 1, 1))
    f.legend(h, l, loc="lower center", ncol=len(names), fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    f.savefig(fig / "fidelity_curves.png")
    plt.close(f)
    return summ, taus


# ------------------------------------------------------------------ 1) akıl sağlığı
def step_sanity(booster, phi0, X, B, seeds, lime_samples, mu, sdv, Xdev, ydev, Xh, yh, params,
                n_seeds, out, fig):
    rows, aucs = [], []
    z = np.abs(zscore(X, mu, sdv))

    def compare(phi_r, extra):
        res = []
        for m in METHODS:
            ag = agreement(phi0[m], phi_r[m])
            ag_z = agreement(phi_r[m], z)                          # rastgele modelin açıklaması girdiye mi benziyor?
            res += summarize({"spearman_abs": ag["spearman_abs"], "spearman_işaretli": ag["spearman_işaretli"],
                              "ilk3": ag["ilk3"], "ort_uzaklık_ile_spearman": ag_z["spearman_abs"]},
                             {**extra, "yöntem": m})
        return res

    t0 = time.perf_counter()
    for s in range(n_seeds):
        lr = LeafRandomizer(booster, seed=100 + s)
        for frac in (0.25, 0.5, 0.75, 1.0):
            rb = lr.booster(frac)
            auc = roc_auc_score(yh, rb.predict(Xh, raw_score=True))
            aucs.append({"test": "kademeli", "adım": frac, "tohum": s, "AUC": auc})
            phi_r, _ = explain_all(rb, X, B, seeds, lime_samples)
            rows += compare(phi_r, {"test": "kademeli", "adım": frac, "tohum": s})
        print(f"  kademeli tohum {s + 1}/{n_seeds} ({time.perf_counter() - t0:.0f} sn)", flush=True)
    for s in range(n_seeds):
        yp = np.random.default_rng(200 + s).permutation(ydev)
        mdl = build_lgbm(params, seed=200 + s).fit(pd.DataFrame(Xdev, columns=FEATURES), yp)
        auc = roc_auc_score(yh, mdl.booster_.predict(Xh, raw_score=True))
        aucs.append({"test": "karışık_etiket", "adım": 1.0, "tohum": s, "AUC": auc})
        phi_r, _ = explain_all(mdl.booster_, X, B, seeds, lime_samples)
        rows += compare(phi_r, {"test": "karışık_etiket", "adım": 1.0, "tohum": s})
    # kontroller: modelden bağımsız açıklama hep aynı kalır (benzerlik 1); rastgele açıklama ≈ 0
    ctrl = [agreement(random_attr(*X.shape, 11 + 2 * k), random_attr(*X.shape, 12 + 2 * k)) for k in range(10)]
    rows += summarize({"spearman_abs": np.concatenate([c["spearman_abs"] for c in ctrl]),
                       "ilk3": np.concatenate([c["ilk3"] for c in ctrl])},
                      {"test": "kontrol", "adım": np.nan, "tohum": -1, "yöntem": "Rastgele"})
    rows += summarize({"spearman_abs": np.ones(len(X)), "ilk3": np.ones(len(X))},
                      {"test": "kontrol", "adım": np.nan, "tohum": -1, "yöntem": "Ort-uzaklık"})
    # orijinal açıklamanın Ort-uzaklık ile benzerliği (referans)
    for m in METHODS:
        rows += summarize({"ort_uzaklık_ile_spearman": agreement(phi0[m], z)["spearman_abs"]},
                          {"test": "orijinal", "adım": 0.0, "tohum": -1, "yöntem": m})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "sanity.csv", index=False)
    aucs = pd.DataFrame(aucs)
    aucs.to_csv(out / "sanity_auc.csv", index=False)
    # şekil
    f, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.5, 1]})
    steps = [0.0, 0.25, 0.5, 0.75, 1.0]
    auc0 = roc_auc_score(yh, booster.predict(Xh, raw_score=True))
    auc_by = aucs[aucs["test"] == "kademeli"].groupby("adım")["AUC"].mean()
    for m in METHODS:
        d = tab[(tab["test"] == "kademeli") & (tab["yöntem"] == m) & (tab["ölçüt"] == "spearman_abs")]
        g = d.groupby("adım")["ort"]
        mean = [1.0] + [g.mean().get(s_, np.nan) for s_ in steps[1:]]
        lo = [1.0] + [g.min().get(s_, np.nan) for s_ in steps[1:]]
        hi = [1.0] + [g.max().get(s_, np.nan) for s_ in steps[1:]]
        a1.plot(steps, mean, "o-", color=COL[m], ms=3, lw=1.5, label=m)
        a1.fill_between(steps, lo, hi, color=COL[m], alpha=0.15, lw=0)
    rnd = tab[(tab["test"] == "kontrol") & (tab["yöntem"] == "Rastgele") & (tab["ölçüt"] == "spearman_abs")]["ort"].iloc[0]
    a1.axhline(1.0, color=COL["Ort-uzaklık"], ls=":", lw=1.2, label="Ort-uzaklık (modelsiz)")
    a1.axhline(rnd, color=COL["Rastgele"], ls="--", lw=1.0, label="Rastgele")
    a1.set_xticks(steps, [f"{int(s_ * 100)}%\nAUC {auc0 if s_ == 0 else auc_by.get(s_, np.nan):.2f}" for s_ in steps], fontsize=6.5)
    a1.set_xlabel("rastgeleleştirilen ağaç oranı (sondan başa)")
    a1.set_ylabel("orijinalle |φ| sıra korelasyonu")
    a1.legend(fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.06))
    d = tab[(tab["test"] == "karışık_etiket") & (tab["ölçüt"] == "spearman_abs")].groupby("yöntem")["ort"]
    xm = np.arange(len(METHODS))
    a2.bar(xm, [d.mean()[m] for m in METHODS], yerr=[d.std()[m] if d.count()[m] > 1 else 0 for m in METHODS],
           color=[COL[m] for m in METHODS], width=0.6, error_kw={"lw": 0.8})
    a2.axhline(rnd, color=COL["Rastgele"], ls="--", lw=1.0)
    a2.set_xticks(xm, METHODS, fontsize=7)
    sh_auc = aucs[aucs["test"] == "karışık_etiket"]["AUC"].mean()
    a2.set_title(f"karıştırılmış etiket (AUC {sh_auc:.2f})", fontsize=8, color=INK)
    a2.set_ylim(min(-0.1, a2.get_ylim()[0]), 1.0)
    for a in (a1, a2):
        a.tick_params(labelsize=7)
    f.tight_layout()
    f.savefig(fig / "sanity.png")
    plt.close(f)
    return tab, aucs


# ------------------------------------------------------------------ 3) kararlılık
def step_stability(booster, phi0, X, B, seeds, lime_samples, mu, sdv, cut, eps_list, M, max_try,
                   out, fig, rng):
    predict = raw_predict(booster)
    n, d = X.shape
    cont = [FEATURES.index(c) for c in CONT]
    thr = split_thresholds(booster, d)
    fx = predict(X)
    px = 1 / (1 + np.exp(-fx))
    leaf_x = booster.predict(X, pred_leaf=True)
    rows, acc_rows = [], []
    phi_x_z = zscore(X, mu, sdv)
    for eps in eps_list:
        P, owner = [], []
        acc_rate = []
        for i in range(n):
            got = tries = 0
            while got < M and tries < max_try:
                tries += 1
                x2 = X[i].copy()
                x2[cont] += eps * sdv[cont] * rng.normal(size=len(cont))
                for c in ("LIMIT_BAL",) + tuple(f"PAY_AMT{k}" for k in range(1, 7)):
                    x2[FEATURES.index(c)] = max(x2[FEATURES.index(c)], 0.0)
                f2 = predict(x2[None])[0]
                p2 = 1 / (1 + np.exp(-f2))
                if (p2 >= cut) == (px[i] >= cut) and abs(p2 - px[i]) <= 0.01:
                    P.append(x2); owner.append(i); got += 1
            acc_rate.append(got / tries)
        P, owner = np.array(P), np.array(owner)
        phiP, _ = explain_all(booster, P, B, seeds[owner], lime_samples)
        phiP["Ort-uzaklık"] = zscore(P, mu, sdv)
        phiP["Rastgele"] = random_attr(len(P), d, 900 + int(eps * 1000))
        fP = predict(P)
        leafP = booster.predict(P, pred_leaf=True)
        cross = np.array([threshold_crossings(X[o], P[k], thr) for k, o in enumerate(owner)])
        leafchg = (leafP != leaf_x[owner]).sum(1)
        dx = np.linalg.norm((P - X[owner]) / sdv, axis=1)
        base = {"eps": eps, "örnek": owner, "Δf": np.abs(fP - fx[owner]),
                "Δp": np.abs(1 / (1 + np.exp(-fP)) - px[owner]), "eşik_geçişi": cross,
                "yaprağı_değişen_ağaç": leafchg, "‖Δx‖_std": dx}
        for m in METHODS + CONTROLS:
            ref = phi_x_z[owner] if m == "Ort-uzaklık" else (random_attr(len(P), d, 800 + int(eps * 1000)) if m == "Rastgele" else phi0[m][owner])
            ag = agreement(ref, phiP[m])
            rl = rel_l2(ref, phiP[m])
            dphi = np.linalg.norm(ref - phiP[m], axis=1)
            rows.append(pd.DataFrame({**base, "yöntem": m, "spearman_abs": ag["spearman_abs"], "ilk3": ag["ilk3"],
                                      "göreli_L2": rl, "‖Δφ‖": dphi,
                                      "‖Δφ‖/Δf": dphi / np.maximum(base["Δf"], 0.01),   # Δf tabanı 0.01 log-odds
                                      "‖Δφ‖/‖Δx‖": dphi / np.maximum(dx, 1e-12)}))
        acc_rows.append({"eps": eps, "kabul_edilen": len(P), "kabul_oranı_ort": float(np.mean(acc_rate)),
                         "örnek_başına_hedef_M": M, "en_çok_deneme": max_try})
        print(f"  ε={eps}: kabul edilen {len(P)} pertürbasyon (kabul oranı ort. {np.mean(acc_rate):.2f})", flush=True)
    st = pd.concat(rows, ignore_index=True)
    st.to_csv(out / "stability_rows.csv", index=False)
    pd.DataFrame(acc_rows).to_csv(out / "stability_acceptance.csv", index=False)
    st["eşik_grubu"] = pd.cut(st["eşik_geçişi"], [-0.5, 0.5, 5.5, 20.5, np.inf], labels=["0", "1–5", "6–20", ">20"])
    summ = st.groupby(["eps", "yöntem"])[["spearman_abs", "ilk3", "göreli_L2", "Δf", "Δp", "eşik_geçişi",
                                          "yaprağı_değişen_ağaç", "‖Δφ‖/Δf"]].agg(["mean", "std", "count"])
    summ.to_csv(out / "stability_summary.csv")
    bygrp = st[st["yöntem"].isin(METHODS)].groupby(["eps", "eşik_grubu", "yöntem"], observed=True)[
        ["göreli_L2", "spearman_abs", "Δf"]].agg(["mean", "count"])
    bygrp.to_csv(out / "stability_by_crossings.csv")
    # şekil
    f, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.6))
    names = METHODS + CONTROLS
    for k, eps in enumerate(eps_list):
        vals = [st[(st["eps"] == eps) & (st["yöntem"] == m)]["spearman_abs"].dropna() for m in names]
        pos = np.arange(len(names)) + (k - 0.5) * 0.36
        bp = a1.boxplot(vals, positions=pos, widths=0.3, showfliers=False, patch_artist=True)
        for patch, m in zip(bp["boxes"], names):
            patch.set_facecolor(COL[m]); patch.set_alpha(0.35 if k == 0 else 0.8); patch.set_edgecolor(COL[m])
        for med in bp["medians"]:
            med.set_color(INK)
    a1.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=6.8)
    a1.set_ylabel("|φ(x)| ile |φ(x̃)| sıra korelasyonu")
    a1.set_title(f"açık: ε={eps_list[0]}, koyu: ε={eps_list[-1]}", fontsize=7.5, color=MUTED)
    sub = st[(st["eps"] == eps_list[-1]) & st["yöntem"].isin(METHODS)]
    g = sub.groupby(["eşik_grubu", "yöntem"], observed=True)["göreli_L2"].mean().unstack()
    for m in METHODS:
        if m in g:
            a2.plot(range(len(g)), g[m].values, "o-", color=COL[m], ms=3.5, lw=1.4, label=m)
    a2.set_xticks(range(len(g)), [str(i) for i in g.index])
    a2.set_xlabel(f"geçilen bölünme eşiği sayısı (ε={eps_list[-1]})")
    a2.set_ylabel("göreli açıklama değişimi ‖Δφ‖/‖φ‖")
    a2.legend(fontsize=6.3, frameon=False)
    for a in (a1, a2):
        a.tick_params(labelsize=7)
    f.tight_layout()
    f.savefig(fig / "stability.png")
    plt.close(f)
    return summ


# ------------------------------------------------------------------ deneyler (uyuşmazlık hipotezleri)
def step_experiments(booster, phi0, X, B, seeds, lime_samples, Xdev, ydev, Xh, yh, params, out, fig, n_lime_seeds):
    predict = raw_predict(booster)
    rows = []
    # E1: toplamsal (etkileşimsiz) model — num_leaves = 2 → her ağaç tek bölünme
    add = build_lgbm({**params, "num_leaves": 2}, seed=SEED).fit(pd.DataFrame(Xdev, columns=FEATURES), ydev)
    auc_add = roc_auc_score(yh, add.booster_.predict(Xh, raw_score=True))
    phi_add, _ = explain_all(add.booster_, X, B, seeds, lime_samples)
    for i, a in enumerate(METHODS):
        for b in METHODS[i + 1:]:
            for model_name, ph in (("orijinal", phi0), ("toplamsal", phi_add)):
                rows += summarize({k: v for k, v in agreement(ph[a], ph[b]).items() if k in ("spearman_abs", "ilk1", "ilk3")},
                                  {"deney": "E1_toplamsal", "model": model_name, "A": a, "B": b})
    # E2: LIME çekirdeği (σ taraması ve Shapley çekirdeği), daha çok örnekle
    # çekirdek (σ taraması, Shapley çekirdeği) × örnekleme gürültüsü (tek arka plan satırı / 20 satır ortalaması)
    half = max(lime_samples // 2, 100)
    for label, kw in [("LIME σ=0.25", {"kernel": "lime", "sigma": 0.25}),
                      ("LIME σ=0.75", {"kernel": "lime", "sigma": 0.75}),
                      ("LIME σ=3", {"kernel": "lime", "sigma": 3.0}),
                      ("LIME Shapley-çekirdeği", {"kernel": "shapley"}),
                      ("LIME σ=0.75 + 20 satır ort.", {"kernel": "lime", "sigma": 0.75, "n_bg_avg": 20, "n_samples": half}),
                      ("LIME Shapley + 20 satır ort.", {"kernel": "shapley", "n_bg_avg": 20, "n_samples": half})]:
        kw = {"n_samples": lime_samples, **kw}
        ph = lime(predict, X, B, seeds=seeds, **kw)
        for ref in ("SHAP-müd", "Örtme", "SHAP-yol"):
            rows += summarize({k: v for k, v in agreement(phi0[ref], ph).items() if k in ("spearman_abs", "ilk1", "ilk3")},
                              {"deney": "E2_çekirdek", "model": label, "A": label, "B": ref})
    # E3: LIME'ın kendi tohum gürültüsü (aynı x, farklı örnekleme)
    lp = [lime(predict, X, B, n_samples=lime_samples, seeds=seeds + 7919 * (r + 1)) for r in range(n_lime_seeds)]
    for i in range(n_lime_seeds):
        for j in range(i + 1, n_lime_seeds):
            rows += summarize({k: v for k, v in agreement(lp[i], lp[j]).items() if k in ("spearman_abs", "ilk1", "ilk3")},
                              {"deney": "E3_LIME_tohum", "model": f"tohum{i}-tohum{j}", "A": "LIME", "B": "LIME"})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "experiments.csv", index=False)
    # şekil
    f, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
    e1 = tab[(tab["deney"] == "E1_toplamsal") & (tab["ölçüt"] == "spearman_abs")]
    pairs = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1:]]
    for k, (a, b) in enumerate(pairs):
        o = e1[(e1["A"] == a) & (e1["B"] == b) & (e1["model"] == "orijinal")]["ort"].iloc[0]
        t = e1[(e1["A"] == a) & (e1["B"] == b) & (e1["model"] == "toplamsal")]["ort"].iloc[0]
        a1.plot([o, t], [k, k], color=FAINT, lw=2.5, zorder=1)
        a1.scatter([o], [k], color=COL["Rastgele"], s=22, zorder=2, label="orijinal model" if k == 0 else None)
        a1.scatter([t], [k], color=COL["SHAP-müd"], s=22, zorder=3, label=f"toplamsal model (AUC {auc_add:.3f})" if k == 0 else None)
    a1.set_yticks(range(len(pairs)), [f"{a} – {b}" for a, b in pairs], fontsize=6.8)
    a1.set_xlabel("|φ| sıra korelasyonu (ort.)")
    a1.legend(fontsize=6.2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    a1.grid(axis="y", visible=False)
    e2 = tab[(tab["deney"] == "E2_çekirdek") & (tab["ölçüt"] == "spearman_abs")]
    labels = list(dict.fromkeys(e2["model"]))
    for ref in ("SHAP-müd", "Örtme"):
        v = [e2[(e2["model"] == lab) & (e2["B"] == ref)]["ort"].iloc[0] for lab in labels]
        a2.plot(range(len(labels)), v, "o-", color=COL[ref], ms=3.5, lw=1.4, label=f"LIME ile {ref}")
    e3 = tab[(tab["deney"] == "E3_LIME_tohum") & (tab["ölçüt"] == "spearman_abs")]["ort"].mean()
    a2.axhline(e3, color=COL["LIME"], ls="--", lw=1, label="LIME tohum–tohum")
    short = [l.replace("LIME ", "").replace("Shapley-çekirdeği", "Shapley").replace(" satır ort.", " ort.")
             for l in labels]
    a2.set_xticks(range(len(labels)), short, fontsize=6.2, rotation=35, ha="right", rotation_mode="anchor")
    a2.set_ylabel("|φ| sıra korelasyonu")
    a2.set_ylim(min(e3, a2.get_ylim()[0]) - 0.05, 1.0)
    a2.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    for a in (a1, a2):
        a.tick_params(labelsize=7)
    f.tight_layout()
    f.savefig(fig / "experiments.png")
    plt.close(f)
    return tab, auc_add


# ------------------------------------------------------------------ main
def main():
    import sys
    for stream in (sys.stdout, sys.stderr):         # Windows konsolunda φ, → vb. için
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = parse_args()
    sfx = "_fast" if args.fast else ""
    out = ROOT / "results" / f"odev6{sfx}"
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    cfg = dict(n_eval=30, n_sub=20, nb=50, lime_samples=300, R=5, n_rand=3, n_seeds=2, M=2, max_try=30,
               n_lime_seeds=3) if args.fast else \
          dict(n_eval=200, n_sub=100, nb=200, lime_samples=5000, R=10, n_rand=10, n_seeds=3, M=5, max_try=50,
               n_lime_seeds=5)
    rng = np.random.default_rng(SEED)

    data_kw = {"path": args.data} if args.data else {}
    Xd_df, yd, Xh_df, yh, _ = get_dev_holdout(**data_kw)
    Xd, Xh = Xd_df[FEATURES].to_numpy(float), Xh_df[FEATURES].to_numpy(float)
    yd, yh = np.asarray(yd), np.asarray(yh)
    model = joblib.load(ROOT / f"models{sfx}" / "odev2_LightGBM.joblib")
    booster = model.booster_
    params = {b["model"]: b["best_params"] for b in json.loads(
        (ROOT / "results" / f"odev2{sfx}" / "budget.json").read_text(encoding="utf-8"))}["LightGBM"]
    predict = raw_predict(booster)
    mu, sdv = Xd.mean(0), Xd.std(0)
    pdev = 1 / (1 + np.exp(-predict(Xd)))
    cut = float(np.quantile(pdev, 0.80))
    B = Xd[rng.choice(len(Xd), cfg["nb"], replace=False)]
    E = Xh[rng.choice(len(Xh), cfg["n_eval"], replace=False)]
    seeds = SEED * 1000 + np.arange(cfg["n_eval"])          # LIME: örnek başına sabit tohum
    sub = np.arange(cfg["n_sub"])                            # akıl sağlığı/kararlılık/deney alt kümesi
    setup = {"yöntemler": METHODS, "kontroller": CONTROLS, "arka_plan_n": cfg["nb"], "açıklanan_n": cfg["n_eval"],
             "alt_küme_n": cfg["n_sub"], "lime_örnek": cfg["lime_samples"], "dolgu_tekrar_R": cfg["R"],
             "rastgele_sıra_tekrar": cfg["n_rand"], "rastgeleleştirme_tohumu": cfg["n_seeds"],
             "kararlılık_pertürbasyon_M": cfg["M"], "ret_eşiği": cut, "çıktı": "log-odds",
             "AUC_holdout": float(roc_auc_score(yh, predict(Xh)))}
    print(json.dumps(setup, ensure_ascii=False))

    t0 = time.perf_counter()
    phi, bases = all_phi(booster, E, B, seeds, cfg["lime_samples"], mu, sdv, rnd_seed=SEED)
    np.savez_compressed(out / "phi_original.npz", X=E, **{k: v for k, v in phi.items()})
    print(f"[0] açıklamalar ({time.perf_counter() - t0:.0f} sn); tabanlar {bases}", flush=True)

    t0 = time.perf_counter()
    agr, worst = step_agreement(phi, out, fig)
    print(f"[4] uyuşma ({time.perf_counter() - t0:.0f} sn); en çok uyuşmayan çift: {worst}")
    print(agr[agr["ölçüt"] == "spearman_abs"][["A", "B", "ort", "sd", "n"]].round(3).to_string(index=False))

    t0 = time.perf_counter()
    fid, taus = step_fidelity(predict, phi, E, Xd, mu, sdv, cfg["R"], cfg["n_rand"], out, fig, rng)
    print(f"[2] sadakat ({time.perf_counter() - t0:.0f} sn)")
    print(fid.round(3).to_string())
    print(taus[["ölçüt", "dolgu_A", "dolgu_B", "tau", "tau_yalnız_yöntemler", "tau_önyükleme_ort", "tau_önyükleme_sd"]].round(3).to_string(index=False))

    t0 = time.perf_counter()
    phi_sub = {m: v[sub] for m, v in phi.items()}
    san, aucs = step_sanity(booster, phi_sub, E[sub], B, seeds[sub], cfg["lime_samples"], mu, sdv, Xd, yd, Xh, yh,
                            params, cfg["n_seeds"], out, fig)
    print(f"[1] akıl sağlığı ({time.perf_counter() - t0:.0f} sn)")
    print(aucs.groupby(["test", "adım"])["AUC"].agg(["mean", "std"]).round(3).to_string())
    print(san[san["ölçüt"] == "spearman_abs"].groupby(["test", "adım", "yöntem"], dropna=False)["ort"].mean().round(3).to_string())

    t0 = time.perf_counter()
    stab = step_stability(booster, phi_sub, E[sub], B, seeds[sub], cfg["lime_samples"], mu, sdv, cut,
                          [0.01, 0.05], cfg["M"], cfg["max_try"], out, fig, rng)
    print(f"[3] kararlılık ({time.perf_counter() - t0:.0f} sn)")
    print(stab.round(3).to_string())

    t0 = time.perf_counter()
    exp, auc_add = step_experiments(booster, phi_sub, E[sub], B, seeds[sub], cfg["lime_samples"], Xd, yd, Xh, yh,
                                    params, out, fig, cfg["n_lime_seeds"])
    print(f"[deney] ({time.perf_counter() - t0:.0f} sn); toplamsal model AUC {auc_add:.3f}")
    print(exp[exp["ölçüt"] == "spearman_abs"][["deney", "model", "A", "B", "ort", "sd"]].round(3).to_string(index=False))

    # ---------------------------------------------------------------- 5) özet tablo
    rows = []
    for m in METHODS:
        s = san[(san["yöntem"] == m) & (san["ölçüt"] == "spearman_abs")]
        a = agr[(agr["ölçüt"] == "spearman_abs") & (((agr["A"] == m) & agr["B"].isin(METHODS)) | ((agr["B"] == m) & agr["A"].isin(METHODS)))]
        a3 = agr[(agr["ölçüt"] == "ilk3") & (((agr["A"] == m) & agr["B"].isin(METHODS)) | ((agr["B"] == m) & agr["A"].isin(METHODS)))]
        row = {"yöntem": m,
               "sağlık_%100_rastgele_ρ": s[(s["test"] == "kademeli") & (s["adım"] == 1.0)]["ort"].mean(),
               "sağlık_karışık_etiket_ρ": s[s["test"] == "karışık_etiket"]["ort"].mean()}
        for fname in ("ortalama", "rastgele", "yakın-komşu"):
            row[f"sadakat_MoRF_{fname}"] = fid.loc[(fname, m), ("MoRF_alanı", "mean")]
            row[f"sadakat_PGI3_{fname}"] = fid.loc[(fname, m), ("PGI@3", "mean")]
        for eps in (0.01, 0.05):
            row[f"kararlılık_ρ_ε{eps}"] = stab.loc[(eps, m), ("spearman_abs", "mean")]
        row["uyuşma_ort_ρ"] = a["ort"].mean()
        row["uyuşma_ort_ilk3"] = a3["ort"].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary_table.csv", index=False)
    lime_noise = exp[(exp["deney"] == "E3_LIME_tohum") & (exp["ölçüt"] == "spearman_abs")]["ort"].mean()
    (out / "summary.json").write_text(json.dumps({"kurulum": setup, "en_çok_uyuşmayan": worst,
                                                  "toplamsal_model_AUC": auc_add,
                                                  "LIME_tohum_tohum_spearman": lime_noise},
                                                 ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(summary.round(3).T.to_string())
    print("bitti →", out)


if __name__ == "__main__":
    main()
