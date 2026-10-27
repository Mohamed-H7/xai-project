"""
Ödev 4 — Küresel açıklama yöntemlerinin karşılaştırmalı denetimi
---------------------------------------------------------------
Ödev 2'deki kara kutu (LightGBM) üzerinde; eğitim = dev kümesi, test = dokunulmamış holdout.

  1) Permütasyon önemi: eğitim ve test, 30 tekrar
  2) Korelasyon blokları (hiyerarşik kümeleme) + gruplanmış permütasyon önemi
     (+ Hooker & Mentch'in alternatifi: bloğu çıkarıp yeniden eğitme, maliyetiyle)
  3) En önemli 3 öznitelik: PDP + c-ICE, paralellik ölçütü, açıklayıcı alt grup
  4) ALE ve PDP; farkın özniteliğin diğerleriyle korelasyonuna karşı çizimi
  5) Küresel vekil ağaç: derinliğe göre sadakat, yerel sapmaların yapısı
  6) "En önemli 3 değişken" yöntemlere göre tablo
  7) Ödev 3 kararlılık analiziyle tutarlılık

Çalıştırma:
  python odev4_global.py           # önce odev2 (ve tercihen odev3) çalışmış olmalı
  python odev4_global.py --fast    # duman testi
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
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor, export_text

from src.data import FEATURES, PAY_COLS, ROOT, SEED, get_dev_holdout
from src.global_methods import (ale_1d, correlation_blocks, deviation_structure, feature_grid,
                                ice_matrix, ice_parallelism, largest_gap_threshold, make_predictors,
                                multiple_r2, pdp_ale_distance, pdp_on, permutation_importance,
                                subgroup_search, surrogate_curve, threshold_sensitivity)
from src.models import build_lgbm

warnings.filterwarnings("ignore")

C_TRAIN, C_TEST, C_ALT = "#2a78d6", "#eb6834", "#1baf7a"
HETERO_I2 = 0.05   # c-ICE uç farklarında I² ≥ 0.05 → "heterojen" (alt grup renklendirmesi yapılır)
from scipy.cluster.hierarchy import set_link_color_palette
set_link_color_palette(["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"])
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": FAINT, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--data", default=None)
    return ap.parse_args()


def money_axis(ax, f, lo, hi):
    if f.startswith("LIMIT"):
        ax.set_xscale("log")
        ticks = [t for t in (1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6) if lo <= t <= hi]
        ax.set_xticks(ticks, [f"{t / 1000:.0f}b" for t in ticks])
        ax.minorticks_off()
    elif f.startswith(("BILL", "PAY_AMT")):
        ax.set_xscale("symlog", linthresh=1000)
        ticks = [t for t in (-1e5, -1e4, 0, 1e3, 1e4, 1e5, 1e6) if lo <= t <= hi]
        ax.set_xticks(ticks, [f"{t:,.0f}".replace(",", ".") for t in ticks])
    else:
        return
    ax.set_xlim(lo, hi)


# ------------------------------------------------------------------ 1) permütasyon önemi
def step1(proba, X, y, Xh, yh, reps, out, fig):
    single = {f: [f] for f in FEATURES}
    tr = permutation_importance(proba, X, y, single, reps, SEED)
    te = permutation_importance(proba, Xh, yh, single, reps, SEED)
    m = tr[["grup", "önem", "sd", "sıra"]].merge(te[["grup", "önem", "sd", "sıra"]], on="grup",
                                                 suffixes=("_eğitim", "_test"))
    m["fark"] = m["önem_eğitim"] - m["önem_test"]
    m["oran"] = m["önem_eğitim"] / m["önem_test"].where(m["önem_test"].abs() > 1e-5)
    m = m.rename(columns={"grup": "öznitelik"}).sort_values("önem_test", ascending=False)
    m.to_csv(out / "pi_train_vs_test.csv", index=False)
    rho = spearmanr(m["önem_eğitim"], m["önem_test"]).statistic
    summary = {"base_auc_eğitim": tr.attrs["base_auc"], "base_auc_test": te.attrs["base_auc"],
               "spearman_sıra": float(rho), "tekrar": reps,
               "toplam_önem_eğitim": float(m["önem_eğitim"].sum()), "toplam_önem_test": float(m["önem_test"].sum())}
    # grafik
    t = m.iloc[::-1]
    yv = np.arange(len(t))
    f_, ax = plt.subplots(figsize=(3.6, 4.6))
    ax.errorbar(t["önem_eğitim"], yv + 0.15, xerr=t["sd_eğitim"], fmt="o", ms=3.5, color=C_TRAIN,
                lw=0.8, label=f"eğitim (AUC {summary['base_auc_eğitim']:.3f})")
    ax.errorbar(t["önem_test"], yv - 0.15, xerr=t["sd_test"], fmt="o", ms=3.5, color=C_TEST,
                lw=0.8, label=f"test (AUC {summary['base_auc_test']:.3f})")
    ax.set_yticks(yv, t["öznitelik"])
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel(f"AUC düşüşü (ort. ± sd, {reps} tekrar)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=7)
    f_.savefig(fig / "pi_train_vs_test.png")
    plt.close(f_)
    return m, summary


# ------------------------------------------------------------------ 2) bloklar
def step2(proba, X, y, Xh, yh, reps, lgbm_params, pi, out, fig, do_loco=True):
    rho, Z, _ = correlation_blocks(X, threshold=0.3)
    thr = largest_gap_threshold(Z)
    _, _, blocks = correlation_blocks(X, threshold=thr)
    sens = threshold_sensitivity(Z, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, round(thr, 3)]).drop_duplicates("eşik")
    sens.sort_values("eşik").to_csv(out / "block_threshold_sensitivity.csv", index=False)
    rho.to_csv(out / "spearman_corr.csv")
    # birincil eşik (en büyük boşluk) + duyarlılık için gevşek eşik 0.5 (ort. |ρ| ≥ 0.5)
    multi, block_thr, seen = {}, {}, set()
    for t in sorted({thr, 0.5}):
        _, _, bl = correlation_blocks(X, threshold=t)
        for k, v in bl.items():
            if len(v) > 1 and frozenset(v) not in seen:
                seen.add(frozenset(v))
                multi[k], block_thr[k] = v, t
    grp = permutation_importance(proba, Xh, yh, multi, reps, SEED + 7) if multi else pd.DataFrame()
    ind = pi.set_index("öznitelik")["önem_test"]
    rows = []
    base = roc_auc_score(yh, proba(Xh))
    for _, r in grp.iterrows():
        cols = r["sütunlar"].split(",")
        within = rho.loc[cols, cols].abs().to_numpy()
        row = {"blok": r["grup"], "eşik": block_thr[r["grup"]], "üye_sayısı": len(cols), "ort_iç_|ρ|": within[np.triu_indices(len(cols), 1)].mean(),
               "gruplanmış_önem": r["önem"], "sd": r["sd"], "tekil_toplam": ind[cols].sum(),
               "tekil_maks": ind[cols].max(), "en_önemli_üye": ind[cols].idxmax()}
        row["gruplanmış/toplam"] = row["gruplanmış_önem"] / max(row["tekil_toplam"], 1e-6)
        if do_loco:
            # Hooker & Mentch alternatifi: bloğu at, aynı ayarlarla yeniden eğit (maliyet = yeniden eğitim)
            t0 = time.perf_counter()
            keep = [c for c in FEATURES if c not in cols]
            m = build_lgbm(lgbm_params).fit(X[keep], y)
            row["çıkar-yeniden-eğit_önem"] = base - roc_auc_score(yh, m.predict_proba(Xh[keep])[:, 1])
            row["çıkar-yeniden-eğit_sn"] = time.perf_counter() - t0
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values("gruplanmış_önem", ascending=False)
    tab.to_csv(out / "grouped_pi.csv", index=False)
    (out / "blocks.json").write_text(json.dumps({"eşik": thr, "kriter": "1-|Spearman ρ|, ortalama bağlantı, en büyük boşluk",
                                                 "bloklar": blocks}, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    # grafik: dendrogram + gruplanmış vs tekil toplam
    f_, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    dendrogram(Z, labels=list(X.columns), orientation="right", ax=a1, color_threshold=thr,
               above_threshold_color=MUTED, leaf_font_size=6.5)
    a1.axvline(thr, color=C_TEST, ls="--", lw=1)
    a1.set_xlabel("1 − |Spearman ρ|  (kesikli: blok eşiği)")
    a1.grid(False)
    if len(tab):
        yv = np.arange(len(tab))[::-1]
        a2.barh(yv + 0.18, tab["gruplanmış_önem"], height=0.34, color=C_TEST, label="gruplanmış permütasyon")
        a2.barh(yv - 0.18, tab["tekil_toplam"], height=0.34, color=C_TRAIN, label="tekil önemlerin toplamı")
        if "çıkar-yeniden-eğit_önem" in tab:
            a2.plot(tab["çıkar-yeniden-eğit_önem"], yv, "D", color=INK, ms=3.5, label="çıkar + yeniden eğit")
        a2.set_yticks(yv, [f"{b}  (t={t:.2f})" for b, t in zip(tab["blok"], tab["eşik"])])
        a2.set_xlabel("test AUC düşüşü")
        a2.legend(fontsize=6.5, frameon=False, loc="lower right")
        a2.grid(axis="y", visible=False)
        a2.tick_params(labelsize=7)
    f_.tight_layout()
    f_.savefig(fig / "blocks_grouped_pi.png")
    plt.close(f_)
    return tab, blocks, thr, rho


# ------------------------------------------------------------------ 3) PDP + c-ICE
def step3(logit, Xh, X, top3, n_ice, out, fig):
    rng = np.random.default_rng(SEED)
    S = Xh.iloc[rng.choice(len(Xh), min(n_ice, len(Xh)), replace=False)].reset_index(drop=True)
    rows, curves = [], {}
    for f in top3:
        g = feature_grid(X, f)
        ice = ice_matrix(logit, S, f, g)
        par = ice_parallelism(ice, g)
        delta = ice[:, -1] - ice[:, 0]
        sub = subgroup_search(delta, S, f, seed=SEED)
        rows.append({"öznitelik": f, "ızgara": f"{g[0]:g} … {g[-1]:g} ({len(g)} nokta)", **par,
                     "heterojen": par["I2"] >= HETERO_I2, **sub})
        curves[f] = (g, ice, sub)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "ice_parallelism.csv", index=False)
    f_, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, f in zip(axes, top3):
        g, ice, sub = curves[f]
        c = ice - ice[:, [0]]
        idx = rng.choice(len(c), min(150, len(c)), replace=False)
        step = "steps-mid" if f in PAY_COLS else "default"
        hetero = tab.set_index("öznitelik").loc[f, "I2"] >= HETERO_I2
        if sub["kök_değişken"] and hetero:
            grp = S[sub["kök_değişken"]].to_numpy() <= sub["kök_eşik"]
            for i in idx:
                ax.plot(g, c[i], color=C_TRAIN if grp[i] else C_TEST, lw=0.4, alpha=0.35, drawstyle=step)
            ax.plot([], [], color=C_TRAIN, label=f"{sub['kök_değişken']} ≤ {sub['kök_eşik']:.0f}")
            ax.plot([], [], color=C_TEST, label=f"{sub['kök_değişken']} > {sub['kök_eşik']:.0f}")
        else:
            for i in idx:
                ax.plot(g, c[i], color=MUTED, lw=0.4, alpha=0.3, drawstyle=step)
        ax.plot(g, c.mean(axis=0), color=INK, lw=2, drawstyle=step, label="PDP (merkezlenmiş)")
        money_axis(ax, f, g[0], g[-1])
        r = tab.set_index("öznitelik").loc[f]
        ax.set_title(f"{f}   I²={r['I2']:.2f}", fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5.8, frameon=False, loc="best")
    axes[0].set_ylabel("c-ICE (log-odds)")
    f_.tight_layout()
    f_.savefig(fig / "pdp_cice_top3.png")
    plt.close(f_)
    return tab


# ------------------------------------------------------------------ 4) ALE vs PDP
def step4(logit, Xh, X, top3, n_pdp, rho, out, fig):
    rng = np.random.default_rng(SEED + 1)
    S = Xh.iloc[rng.choice(len(Xh), min(n_pdp, len(Xh)), replace=False)].reset_index(drop=True)
    r2 = multiple_r2(X)
    absr = rho.abs().where(~np.eye(len(rho), dtype=bool)).max()
    rows, store = [], {}
    for f in FEATURES:
        edges, ale = ale_1d(logit, Xh, f)
        pdp = pdp_on(logit, S, f, edges)
        dist = pdp_ale_distance(pdp, ale, edges, Xh[f].to_numpy(dtype=float))
        rows.append({"öznitelik": f, **dist, "R2_diğerleri": r2[f], "maks_|ρ|": absr[f],
                     "pdp_sd": float(np.interp(Xh[f], edges, pdp).std()),
                     "ale_sd": float(np.interp(Xh[f], edges, ale).std())})
        store[f] = (edges, pdp, ale)
    tab = pd.DataFrame(rows).sort_values("göreli_fark", ascending=False)
    tab.to_csv(out / "ale_vs_pdp.csv", index=False)
    # etkisi ihmal edilebilir öznitelikler göreli farkta gürültü üretir → korelasyon iki şekilde
    big = tab[tab["etki_sd"] >= 0.1 * tab["etki_sd"].max()]
    corr = {"spearman_tümü": float(spearmanr(tab["R2_diğerleri"], tab["göreli_fark"]).statistic),
            "spearman_etkili": float(spearmanr(big["R2_diğerleri"], big["göreli_fark"]).statistic),
            "n_etkili": int(len(big))}
    f_, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    for ax, f in zip(axes.ravel()[:3], top3):
        e, p, a = store[f]
        step = "steps-mid" if f in PAY_COLS else "default"
        ax.plot(e, p, color=C_TRAIN, lw=1.6, drawstyle=step, label="PDP")
        ax.plot(e, a, color=C_TEST, lw=1.6, drawstyle=step, label="ALE")
        money_axis(ax, f, e[0], e[-1])
        r = tab.set_index("öznitelik").loc[f]
        ax.set_title(f"{f}   göreli fark={r['göreli_fark']:.2f}", fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, frameon=False)
    axes[0, 0].set_ylabel("log-odds (merkezlenmiş)")
    axes[1, 0].set_ylabel("log-odds (merkezlenmiş)")
    ax = axes[1, 1]
    small = ~tab["öznitelik"].isin(big["öznitelik"])
    ax.scatter(tab.loc[~small, "R2_diğerleri"], tab.loc[~small, "göreli_fark"], s=18, color=C_TEST, label="etkili")
    ax.scatter(tab.loc[small, "R2_diğerleri"], tab.loc[small, "göreli_fark"], s=14, facecolor="none",
               edgecolor=MUTED, label="etkisi küçük (<%10)")
    for _, r in tab.iterrows():
        if r["öznitelik"] in top3 or r["göreli_fark"] >= tab["göreli_fark"].quantile(0.85):
            ax.annotate(r["öznitelik"], (r["R2_diğerleri"], r["göreli_fark"]), fontsize=6,
                        xytext=(3, 2), textcoords="offset points", color=INK)
    ax.set_xlabel("R² (öznitelik | diğerleri)")
    ax.set_ylabel("|PDP − ALE| / etki")
    ax.set_title(f"Spearman ρ = {corr['spearman_etkili']:.2f} (etkili, n={corr['n_etkili']})", fontsize=8.5, color=INK)
    ax.legend(fontsize=6.5, frameon=False)
    ax.tick_params(labelsize=7)
    f_.tight_layout()
    f_.savefig(fig / "ale_vs_pdp.png")
    plt.close(f_)
    return tab, corr


# ------------------------------------------------------------------ 5) vekil ağaç
def step5(proba, X, Xh, depths, chosen, n_null, out, fig):
    f_tr, f_te = proba(X), proba(Xh)
    curve = surrogate_curve(f_tr, X, f_te, Xh, depths, seed=SEED)
    curve.to_csv(out / "surrogate_fidelity.csv", index=False)
    s = DecisionTreeRegressor(max_depth=chosen, min_samples_leaf=50, random_state=SEED).fit(X, f_tr)
    (out / "surrogate_tree.txt").write_text(export_text(s, feature_names=list(X.columns), decimals=1),
                                            encoding="utf-8")
    resid = f_te - s.predict(Xh)
    q = np.quantile(resid, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    struct = deviation_structure(resid, Xh, seed=SEED, n_null=n_null)
    worst = struct.pop("worst_mask")
    prof = pd.DataFrame({"en_kötü_alt_grup": Xh[worst].median(), "diğerleri": Xh[~worst].median()})
    prof.loc["kara_kutu_olasılığı_ort"] = [f_te[worst].mean(), f_te[~worst].mean()]
    prof.loc["ort_|sapma|"] = [np.abs(resid[worst]).mean(), np.abs(resid[~worst]).mean()]
    prof.to_csv(out / "surrogate_worst_subgroup_profile.csv")
    summary = {"seçilen_derinlik": chosen, "sapma_nicelikleri": dict(zip(["p01", "p05", "p25", "p50", "p75", "p95", "p99"],
                                                                          map(float, q))),
               "ort_|sapma|": float(np.abs(resid).mean()), **struct,
               "|sapma|_vs_olasılık_spearman": float(spearmanr(np.abs(resid), f_te).statistic),
               "önem_vekil": dict(zip(X.columns, map(float, s.feature_importances_)))}
    f_, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.5))
    a1.plot(curve["derinlik"], curve["R2_eğitim"], "o-", color=C_TRAIN, ms=3, label="R² eğitim (dev)")
    a1.plot(curve["derinlik"], curve["R2_test"], "o-", color=C_TEST, ms=3, label="R² test (holdout)")
    a1.plot(curve["derinlik"], curve["karar_uyumu_test"], "s--", color=C_ALT, ms=3, label="%20 ret kararında uyum")
    a1.axvline(chosen, color=MUTED, ls=":", lw=1)
    a1.set_xlabel("vekil ağaç derinliği")
    a1.set_ylabel("sadakat")
    a1.legend(fontsize=6.5, frameon=False, loc="lower right")
    bins = np.linspace(q[0], q[-1], 50)
    a2.hist(resid[~worst], bins=bins, color=MUTED, alpha=0.6, label="diğerleri")
    a2.hist(resid[worst], bins=bins, color=C_TEST, alpha=0.8, label=f"en kötü yaprak (n={worst.sum()})")
    a2.set_xlabel(f"yerel sapma: kara kutu − vekil (derinlik {chosen})")
    a2.legend(fontsize=6.5, frameon=False)
    for a in (a1, a2):
        a.tick_params(labelsize=7)
    f_.tight_layout()
    f_.savefig(fig / "surrogate.png")
    plt.close(f_)
    return curve, summary


# ------------------------------------------------------------------ 6) sıra tablosu
def step6(pi, ale_tab, sur_imp, out):
    df = pi.set_index("öznitelik")[["önem_eğitim", "önem_test"]].rename(
        columns={"önem_eğitim": "PI_eğitim", "önem_test": "PI_test"})
    df = df.join(ale_tab.set_index("öznitelik")[["pdp_sd", "ale_sd"]].rename(
        columns={"pdp_sd": "PDP_sd", "ale_sd": "ALE_sd"}))
    df["vekil_ağaç"] = pd.Series(sur_imp)
    ranks = df.rank(ascending=False, method="min").astype(int)
    ranks["sıra_sd"] = ranks.std(axis=1)
    ranks["sıra_aralığı"] = ranks.drop(columns="sıra_sd").max(axis=1) - ranks.drop(columns="sıra_sd").min(axis=1)
    ranks = ranks.sort_values("PI_test")
    ranks.to_csv(out / "rank_table.csv")
    top3 = {m: list(df[m].sort_values(ascending=False).index[:3]) for m in df.columns}
    (out / "top3_by_method.json").write_text(json.dumps(top3, ensure_ascii=False, indent=2), encoding="utf-8")
    return ranks, top3


# ------------------------------------------------------------------ 7) Ödev 3 ile tutarlılık
def step7(ranks, out, fast):
    d3 = ROOT / "results" / ("odev3_fast" if fast else "odev3")
    p_lasso, p_ebm = d3 / "stability_lasso_by_variable.csv", d3 / "stability_ebm.csv"
    if not p_lasso.exists():
        return None
    lasso = pd.read_csv(p_lasso, index_col=0).iloc[:, 0]
    # seçilme sıklığı 0.5'e ne kadar yakınsa o kadar kararsız: 1 − |2p − 1|
    inst = pd.DataFrame({"LASSO_seçilme": lasso, "LASSO_kararsızlık": 1 - (2 * lasso - 1).abs()})
    if p_ebm.exists():
        inst["EBM_göreli_belirsizlik"] = pd.read_csv(p_ebm).set_index("değişken")["göreli_belirsizlik"]
    tab = ranks[["sıra_sd", "sıra_aralığı", "PI_test"]].join(inst, how="left")
    tab.to_csv(out / "consistency_with_odev3.csv")
    res = {}
    for c in ("LASSO_kararsızlık", "EBM_göreli_belirsizlik"):
        if c in tab:
            ok = tab[[c, "sıra_sd"]].dropna()
            res[f"spearman({c}, yöntemler_arası_sıra_sd)"] = float(spearmanr(ok[c], ok["sıra_sd"]).statistic)
    return res


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    sfx = "_fast" if args.fast else ""
    out = ROOT / "results" / f"odev4{sfx}"
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    reps = 5 if args.fast else 30
    n_ice, n_pdp = (300, 300) if args.fast else (1500, 1500)
    depths = range(1, 7) if args.fast else range(1, 11)
    n_null = 20 if args.fast else 200

    data_kw = {"path": args.data} if args.data else {}
    X, y, Xh, yh, _ = get_dev_holdout(**data_kw)
    model_path = ROOT / f"models{sfx}" / "odev2_LightGBM.joblib"
    if not model_path.exists():
        raise SystemExit(f"{model_path} yok — önce odev2_rashomon.py çalıştırın.")
    model = joblib.load(model_path)
    lgbm_params = {b["model"]: b["best_params"] for b in json.loads(
        (ROOT / "results" / f"odev2{sfx}" / "budget.json").read_text(encoding="utf-8"))}["LightGBM"]
    proba, logit = make_predictors(model)
    print(f"dev={len(X)} holdout={len(Xh)}  tekrar={reps}", flush=True)

    t0 = time.perf_counter()
    pi, s1 = step1(proba, X, y, Xh, yh, reps, out, fig)
    print(f"[1] PI bitti ({time.perf_counter() - t0:.0f} sn)  sıra korelasyonu eğitim-test = {s1['spearman_sıra']:.3f}")
    print(pi.head(8).round(4).to_string(index=False))

    grp, blocks, thr, rho = step2(proba, X, y, Xh, yh, reps, lgbm_params, pi, out, fig)
    print(f"[2] blok eşiği={thr}  bloklar: {[k for k, v in blocks.items() if len(v) > 1]}")
    print(grp.round(4).to_string(index=False))

    top3 = list(pi["öznitelik"].head(3))
    ice = step3(logit, Xh, X, top3, n_ice, out, fig)
    print("[3] c-ICE paralellik:\n", ice.round(3).to_string(index=False))

    ale_tab, corr = step4(logit, Xh, X, top3, n_pdp, rho, out, fig)
    print("[4] PDP–ALE farkı vs korelasyon:", corr)
    print(ale_tab.head(8).round(3).to_string(index=False))

    curve, s5 = step5(proba, X, Xh, depths, 4, n_null, out, fig)
    print("[5] vekil sadakat:\n", curve.round(3).to_string(index=False))
    print({k: v for k, v in s5.items() if k != "önem_vekil"})

    ranks, top3_m = step6(pi, ale_tab, s5["önem_vekil"], out)
    print("[6] yönteme göre ilk 3:", top3_m)

    cons = step7(ranks, out, args.fast)
    print("[7] Ödev 3 ile tutarlılık:", cons if cons is not None else "Ödev 3 sonuçları bulunamadı, atlandı")

    (out / "summary.json").write_text(json.dumps({
        "pi": s1, "blok_eşiği": thr, "ale_pdp_korelasyon": corr,
        "vekil": {k: v for k, v in s5.items()}, "ilk3": top3_m, "odev3_tutarlılık": cons,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("bitti →", out, f"({time.perf_counter() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
