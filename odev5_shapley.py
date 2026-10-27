"""
Ödev 5 — Kesin müdahaleci Shapley değerleri, arka plan seçimi ve TreeSHAP türleri
-------------------------------------------------------------------------------
Model: Ödev 2'deki LightGBM, açıklanan çıktı = log-odds (ham skor).
Öznitelikler: Ödev 4'teki test permütasyon önemine göre ilk 12; diğer 9 öznitelik dev kümesi
ortalamasında sabit (raporda belirtilir).

  1) Kendi kesin Shapley kodumuz + dört aksiyom testi
  2) shap.TreeExplainer (müdahaleci) ile aynı arka planda karşılaştırma (hedef ≤ 1e-6)
  3) 3 kişi × arka plan boyutu {1, 10, 100, 500} × 5 tohum → referanstan göreli sapma
  4) 3 kişi × 6 referans grubu → taban, φ, işaret değişimleri
  5) Yol bağımlı ve müdahaleci TreeSHAP karşılaştırması

Her SHAP tablosunda: explainer, değer fonksiyonu, arka plan, arka plan boyutu, taban sütunları.

Çalıştırma:
  python odev5_shapley.py           # önce odev2 (tercihen odev4) çalışmış olmalı
  python odev5_shapley.py --fast    # duman testi
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
from scipy.stats import spearmanr

from src.data import FEATURES, PAY_COLS, ROOT, SEED, TARGET, get_dev_holdout, get_split
from src.global_methods import permutation_importance
from src.shapley import (ReducedModel, axiom_additivity, axiom_dummy, axiom_efficiency,
                         axiom_symmetry, exact_shapley, linear_closed_form)

warnings.filterwarnings("ignore")

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#e6e5e0"
PERSON_COLORS = {"K1": C2, "K2": "#6250d6", "K3": C3}
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": FAINT, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

VALUE_FN = "müdahaleci: v(S)=ort_b f(x_S, b_S̄)"
VALUE_FN_PD = "yol bağımlı: v(S)=E[f | x_S] ağaç örtüsüyle"
OWN = "kendi kesin kodu (2^d numaralandırma)"
SIGN_EPS = 0.01        # işaret değişimi: iki değerden büyüğü ≥ 0.01 log-odds olmalı …
SIGN_MIN = 0.001       # … ve küçüğü ≥ 0.001 (sıfır civarındaki sayısal kıpırtı değişim sayılmaz)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n-features", type=int, default=12)
    ap.add_argument("--ref", choices=["treeshap", "own"], default="treeshap",
                    help="Adım 3 referansı (tüm dev arka planı): TreeSHAP (hızlı; Adım 2'de kendi kodla "
                         "eşitliği gösterilir) ya da kendi kodumuz (yavaş: 2^d × n_dev tahmin)")
    return ap.parse_args()


def interventional_explainer(model, B_full: np.ndarray):
    """
    DİKKAT: shap.TreeExplainer'a 100'den büyük bir arka plan doğrudan verilirse, kütüphane onu
    sessizce 100 satıra alt örnekler (maskers.Independent, max_samples=100). "Aynı arka plan"
    koşulu için max_samples arka plan boyutuna eşitlenir ve sonuç doğrulanır.
    """
    import shap
    masker = shap.maskers.Independent(B_full, max_samples=len(B_full))
    ex = shap.TreeExplainer(model, data=masker, feature_perturbation="interventional", model_output="raw")
    if ex.data.shape[0] != len(B_full):
        raise RuntimeError(f"shap arka planı değiştirdi: {len(B_full)} → {ex.data.shape[0]}")
    return ex


def long_table(rows, selected):
    """Kişi/grup başına φ'leri, raporlama meta verisiyle birlikte uzun biçime çevirir."""
    out = []
    for r in rows:
        for f, val in zip(selected, r["phi"]):
            out.append({k: v for k, v in r.items() if k != "phi"} | {"öznitelik": f, "phi": val})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ öznitelik seçimi
def select_features(model, Xh, yh, k, fast):
    p = ROOT / "results" / ("odev4_fast" if fast else "odev4") / "pi_train_vs_test.csv"
    if p.exists():
        pi = pd.read_csv(p).sort_values("önem_test", ascending=False)
        src = "Ödev 4 test permütasyon önemi"
    else:
        proba = lambda X: model.predict_proba(X)[:, 1]
        pi = permutation_importance(proba, Xh, yh, {f: [f] for f in FEATURES}, 5, SEED)
        pi = pi.rename(columns={"grup": "öznitelik", "önem": "önem_test"})
        src = "test permütasyon önemi (5 tekrar; Ödev 4 sonucu bulunamadı)"
    return list(pi["öznitelik"].head(k)), src


# ------------------------------------------------------------------ kişiler
def pick_persons(fm: ReducedModel, Xh, demo_h, sel, dev_prob):
    """Holdout'tan üç kişi: yüksek risk (p95), ret eşiğine yakın (dev p80), düşük risk (p20)."""
    logit = fm(Xh[sel].to_numpy(float))
    prob = 1 / (1 + np.exp(-logit))
    cut = np.quantile(dev_prob, 0.80)                     # %20 ret politikası eşiği
    targets = {"K1": np.quantile(prob, 0.95), "K2": cut, "K3": np.quantile(prob, 0.20)}
    labels = {"K1": "yüksek risk", "K2": "ret eşiğine yakın", "K3": "düşük risk"}
    rows = {}
    for k, t in targets.items():
        i = int(np.argmin(np.abs(prob - t)))
        rows[k] = {"etiket": labels[k], "holdout_sıra": i, "olasılık": prob[i], "log_odds": logit[i],
                   **demo_h.iloc[i].to_dict(), **Xh.iloc[i].to_dict()}
    return pd.DataFrame(rows).T, cut


# ------------------------------------------------------------------ 1) aksiyomlar
def step1(fm, booster, persons, Xh, Xd, sel, fixed, out, nb, seed):
    rng = np.random.default_rng(seed)
    B = Xd[sel].to_numpy(float)[rng.choice(len(Xd), nb, replace=False)]
    T = booster.num_trees()
    half = T // 2
    f1 = ReducedModel(booster, FEATURES, sel, fixed, start=0, num=half)
    f2 = ReducedModel(booster, FEATURES, sel, fixed, start=half)
    stems = [s.rstrip("0123456789").rstrip("_") for s in sel]
    pair = next(((i, k) for i in range(len(sel)) for k in range(i + 1, len(sel)) if stems[i] == stems[k]), (0, 1))
    rows = []
    for pid, p in persons.iterrows():
        x = Xh.iloc[int(p["holdout_sıra"])][sel].to_numpy(float)
        for r in (axiom_efficiency(fm, x, B), axiom_dummy(fm, x, B, 0, sel[0]),
                  axiom_symmetry(fm, x, B, pair[0], pair[1], sel),
                  axiom_additivity(f1, f2, x, B, f"ağaçlar 1–{half} ve {half + 1}–{T}: φ(f₁+f₂) = φ(f₁)+φ(f₂)"),
                  linear_closed_form(x, B, seed)):
            rows.append({"kişi": pid, **r, "arka_plan_n": nb})
    tab = pd.DataFrame(rows)
    tab["geçti (≤1e-9)"] = tab["ihlal"] <= 1e-9
    summary = (tab.groupby(["aksiyom", "test"], sort=False)
               .agg(maks_ihlal=("ihlal", "max"), geçti=("geçti (≤1e-9)", "all")).reset_index())
    tab.to_csv(out / "axiom_tests_detail.csv", index=False)
    summary.to_csv(out / "axiom_tests.csv", index=False)
    return summary


# ------------------------------------------------------------------ 2) shap ile karşılaştırma
def step2(fm, model, persons, Xh, Xd, sel, out, nb, n_extra, seed):
    import shap
    rng = np.random.default_rng(seed + 1)
    B = Xd[sel].to_numpy(float)[rng.choice(len(Xd), nb, replace=False)]
    extra = rng.choice(len(Xh), n_extra, replace=False)
    idx = [int(i) for i in persons["holdout_sıra"]] + [int(i) for i in extra if i not in persons["holdout_sıra"].values]
    names = list(persons.index) + [f"rastgele_{i}" for i in idx[len(persons):]]
    expl = interventional_explainer(model, fm.full_rows(B))
    rows, phi_rows = [], []
    t_own = t_lib = 0.0
    for name, i in zip(names, idx):
        x = Xh.iloc[i][sel].to_numpy(float)
        t0 = time.perf_counter()
        phi, base, fx = exact_shapley(fm, x, B)
        t_own += time.perf_counter() - t0
        t0 = time.perf_counter()
        sv = expl.shap_values(fm.full_rows(x[None, :]), check_additivity=False)
        t_lib += time.perf_counter() - t0
        sv = np.asarray(sv).reshape(-1)
        lib_sel = sv[fm.idx]
        lib_fixed = np.delete(sv, fm.idx)
        ev = float(np.ravel(expl.expected_value)[-1])
        fx_direct = float(fm(x[None, :])[0])
        rows.append({"kişi": name, "maks_|Δφ|": float(np.abs(phi - lib_sel).max()),
                     # farkın kaynağını ayırmak için her iki yöntemin kendi verimlilik hatası
                     "kendi_verimlilik_hatası": abs(phi.sum() + base - fx_direct),
                     "shap_verimlilik_hatası": abs(sv.sum() + ev - fx_direct),
                     "|Δtaban|": abs(base - ev), "sabit_özniteliklerde_maks_|φ_shap|": float(np.abs(lib_fixed).max()),
                     "en_büyük_farkın_özniteliği": sel[int(np.argmax(np.abs(phi - lib_sel)))]})
        for lab, vals, b, ex in [("kendi", phi, base, OWN),
                                 ("shap", lib_sel, ev, "shap.TreeExplainer(feature_perturbation='interventional')")]:
            phi_rows.append({"kişi": name, "kaynak": lab, "explainer": ex, "değer_fonksiyonu": VALUE_FN,
                             "arka_plan": "dev kümesinden rastgele", "arka_plan_n": nb, "taban": b, "phi": vals})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "own_vs_shap.csv", index=False)
    long_table(phi_rows, sel).to_csv(out / "own_vs_shap_phi.csv", index=False)
    summ = {"arka_plan_n": nb, "açıklanan_kişi": len(idx), "maks_|Δφ|": float(tab["maks_|Δφ|"].max()),
            "maks_|Δtaban|": float(tab["|Δtaban|"].max()),
            "en_büyük_fark_kişi": tab.loc[tab["maks_|Δφ|"].idxmax(), "kişi"],
            "en_büyük_fark_öznitelik": tab.loc[tab["maks_|Δφ|"].idxmax(), "en_büyük_farkın_özniteliği"],
            "maks_kendi_verimlilik_hatası": float(tab["kendi_verimlilik_hatası"].max()),
            "maks_shap_verimlilik_hatası": float(tab["shap_verimlilik_hatası"].max()),
            "süre_kendi_sn": round(t_own, 2), "süre_shap_sn": round(t_lib, 2)}
    return tab, summ


# ------------------------------------------------------------------ 3) arka plan boyutu
def reference_phi(fm, model, persons, Xh, Xd, sel, how):
    Bfull = Xd[sel].to_numpy(float)
    ref = {}
    if how == "own":
        for pid, p in persons.iterrows():
            x = Xh.iloc[int(p["holdout_sıra"])][sel].to_numpy(float)
            phi, base, _ = exact_shapley(fm, x, Bfull)
            ref[pid] = (phi, base)
        return ref, OWN
    import shap
    expl = interventional_explainer(model, fm.full_rows(Bfull))
    ev = float(np.ravel(expl.expected_value)[-1])
    for pid, p in persons.iterrows():
        x = Xh.iloc[int(p["holdout_sıra"])][sel].to_numpy(float)
        sv = np.asarray(expl.shap_values(fm.full_rows(x[None, :]), check_additivity=False)).reshape(-1)
        ref[pid] = (sv[fm.idx], ev)
    return ref, "shap.TreeExplainer(interventional) — Adım 2'de kendi kodla eşitliği doğrulandı"


def step3(fm, model, persons, Xh, Xd, sel, sizes, seeds, ref_how, out, fig):
    t0 = time.perf_counter()
    ref, ref_expl = reference_phi(fm, model, persons, Xh, Xd, sel, ref_how)
    t_ref = time.perf_counter() - t0
    rows, phi_rows = [], []
    for pid, p in persons.iterrows():
        x = Xh.iloc[int(p["holdout_sıra"])][sel].to_numpy(float)
        phi_ref, base_ref = ref[pid]
        phi_rows.append({"kişi": pid, "boyut": len(Xd), "tohum": -1, "explainer": ref_expl,
                         "değer_fonksiyonu": VALUE_FN, "arka_plan": "tüm dev kümesi (referans)",
                         "arka_plan_n": len(Xd), "taban": base_ref, "phi": phi_ref})
        for n in sizes:
            for s in range(seeds):
                rng = np.random.default_rng(1000 * n + s)          # tohum: (boyut, tekrar) çiftine bağlı
                B = Xd[sel].to_numpy(float)[rng.choice(len(Xd), n, replace=False)]
                phi, base, _ = exact_shapley(fm, x, B)
                rows.append({"kişi": pid, "boyut": n, "tohum": s,
                             "göreli_sapma": float(np.linalg.norm(phi - phi_ref) / np.linalg.norm(phi_ref)),
                             "maks_|Δφ|": float(np.abs(phi - phi_ref).max()),
                             "|Δtaban|": abs(base - base_ref),
                             "işaret_değişen": int(np.sum((np.sign(phi) != np.sign(phi_ref))
                                                          & (np.abs(phi_ref) >= SIGN_EPS)))})
                phi_rows.append({"kişi": pid, "boyut": n, "tohum": s, "explainer": OWN, "değer_fonksiyonu": VALUE_FN,
                                 "arka_plan": "dev kümesinden rastgele", "arka_plan_n": n, "taban": base, "phi": phi})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "background_size.csv", index=False)
    long_table(phi_rows, sel).to_csv(out / "background_size_phi.csv", index=False)
    agg = tab.groupby(["kişi", "boyut"])["göreli_sapma"].agg(["mean", "std", "min", "max"]).reset_index()
    agg.to_csv(out / "background_size_summary.csv", index=False)
    f_, ax = plt.subplots(figsize=(3.6, 2.6))
    for pid in persons.index:
        a = agg[agg["kişi"] == pid]
        ax.fill_between(a["boyut"], a["min"], a["max"], color=PERSON_COLORS[pid], alpha=0.15, lw=0)
        ax.plot(a["boyut"], a["mean"], "o-", color=PERSON_COLORS[pid], ms=3.5, lw=1.4,
                label=f"{pid} ({persons.loc[pid, 'etiket']})")
        t = tab[tab["kişi"] == pid]
        ax.scatter(t["boyut"], t["göreli_sapma"], s=6, color=PERSON_COLORS[pid], alpha=0.5, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("arka plan boyutu (rastgele, dev)")
    ax.set_ylabel("‖φ − φ_ref‖ / ‖φ_ref‖")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(f"{seeds} tohum; bant = min–maks; ref. n={len(Xd)}", fontsize=8, color=MUTED)
    f_.savefig(fig / "background_size.png")
    plt.close(f_)
    return agg, {"referans_explainer": ref_expl, "referans_süre_sn": round(t_ref, 1)}


# ------------------------------------------------------------------ 4) referans grupları
AGE_BANDS = [(21, 29), (30, 39), (40, 49), (50, 99)]


def reference_groups(person, demo_d, prob_d, y_d, X_d, cut):
    age = person["AGE"]
    lo, hi = next(b for b in AGE_BANDS if b[0] <= age <= b[1])
    q = pd.qcut(X_d["LIMIT_BAL"], 5, labels=False, duplicates="drop")
    edges = X_d["LIMIT_BAL"].quantile(np.linspace(0, 1, 6)).to_numpy()
    pq = int(np.clip(np.searchsorted(edges, person["LIMIT_BAL"], side="right") - 1, 0, 4))
    sex = {1: "erkek", 2: "kadın"}.get(int(person["SEX"]), str(person["SEX"]))
    return {
        "G1 tüm müşteriler": np.ones(len(X_d), bool),
        "G2 onaylananlar (model, %20 ret)": prob_d < cut,
        "G3 temerrüde düşmeyenler (y=0)": y_d == 0,
        f"G4 aynı yaş grubu ({lo}–{hi if hi < 99 else '+'})": demo_d["AGE"].between(lo, hi).to_numpy(),
        f"G5 aynı cinsiyet ({sex})": (demo_d["SEX"] == person["SEX"]).to_numpy(),
        f"G6 aynı limit dilimi (Q{pq + 1})": (q == pq).to_numpy(),
    }


def step4(fm, persons, Xh, Xd, demo_d, prob_d, y_d, sel, cut, nb, out, fig):
    rows, phi_rows = [], []
    store = {}
    for pid, p in persons.iterrows():
        x = Xh.iloc[int(p["holdout_sıra"])][sel].to_numpy(float)
        groups = reference_groups(p, demo_d, prob_d, y_d, Xd, cut)
        res = {}
        for g, mask in groups.items():
            pool = np.flatnonzero(mask)
            rng = np.random.default_rng(SEED + 31)
            take = pool if len(pool) <= nb else rng.choice(pool, nb, replace=False)
            B = Xd[sel].to_numpy(float)[take]
            phi, base, fx = exact_shapley(fm, x, B)
            res[g] = (phi, base, len(pool), len(take))
            phi_rows.append({"kişi": pid, "grup": g, "explainer": OWN, "değer_fonksiyonu": VALUE_FN,
                             "arka_plan": g, "arka_plan_n": len(take), "grup_büyüklüğü": len(pool),
                             "taban": base, "f(x)": fx, "phi": phi})
        phi_pop = res["G1 tüm müşteriler"][0]
        for g, (phi, base, npool, ntake) in res.items():
            flips = [f for f, a, b in zip(sel, phi, phi_pop)
                     if np.sign(a) != np.sign(b) and max(abs(a), abs(b)) >= SIGN_EPS and min(abs(a), abs(b)) >= SIGN_MIN]
            rows.append({"kişi": pid, "grup": g, "grup_büyüklüğü": npool, "arka_plan_n": ntake,
                         "taban": base, "Σφ": float(phi.sum()), "işaret_değişen_sayısı": len(flips),
                         "işaret_değişenler": ", ".join(flips)})
        store[pid] = res
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "reference_groups.csv", index=False)
    long_table(phi_rows, sel).to_csv(out / "reference_groups_phi.csv", index=False)
    # ısı haritaları: kişi başına (grup × öznitelik), işaret değişimleri "×"
    f_, axes = plt.subplots(len(persons), 1, figsize=(7.2, 1.8 * len(persons) + 0.8), sharex=True)
    f_.subplots_adjust(hspace=0.28)
    vmax = max(np.abs(r[0]).max() for res in store.values() for r in res.values())
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [C1, "#f1f0ec", C2])
    for ax, (pid, res) in zip(np.atleast_1d(axes), store.items()):
        G = list(res)
        M = np.vstack([res[g][0] for g in G])
        im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        pop = res[G[0]][0]
        for r, g in enumerate(G):
            for c in range(len(sel)):
                a, b = M[r, c], pop[c]
                if np.sign(a) != np.sign(b) and max(abs(a), abs(b)) >= SIGN_EPS and min(abs(a), abs(b)) >= SIGN_MIN:
                    ax.text(c, r, "×", ha="center", va="center", fontsize=9, color=INK, weight="bold")
        ax.set_yticks(range(len(G)), [f"{g}  | taban {res[g][1]:+.2f}" for g in G], fontsize=6.5)
        ax.set_xticks(range(len(sel)), sel, rotation=40, ha="right", fontsize=6.5)
        ax.tick_params(axis="x", labelbottom=ax is np.atleast_1d(axes)[-1])
        ax.set_title(f"{pid} ({persons.loc[pid, 'etiket']}), f(x) = {persons.loc[pid, 'log_odds']:+.2f} log-odds",
                     fontsize=8.5, color=INK, loc="left")
        ax.grid(False)
    cb = f_.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("φ (log-odds)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    f_.savefig(fig / "reference_groups.png")
    plt.close(f_)
    return tab


# ------------------------------------------------------------------ 5) TreeSHAP türleri
def step5(fm, model, persons, Xh, Xd, sel, n_explain, nb, out, fig, fast):
    import shap
    rng = np.random.default_rng(SEED + 5)
    E = Xh[sel].to_numpy(float)[rng.choice(len(Xh), n_explain, replace=False)]
    B = Xd[sel].to_numpy(float)[rng.choice(len(Xd), nb, replace=False)]
    E_full = fm.full_rows(E)
    ex_int = interventional_explainer(model, fm.full_rows(B))
    ex_pd = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    s_int = np.asarray(ex_int.shap_values(E_full, check_additivity=False))
    s_pd = np.asarray(ex_pd.shap_values(E_full, check_additivity=False))
    s_int, s_pd = s_int.reshape(len(E), -1), s_pd.reshape(len(E), -1)
    base_int, base_pd = float(np.ravel(ex_int.expected_value)[-1]), float(np.ravel(ex_pd.expected_value)[-1])
    fixed_idx = [i for i in range(len(FEATURES)) if i not in fm.idx]
    rows = []
    for j, f in enumerate(FEATURES):
        a, b = s_int[:, j], s_pd[:, j]
        rows.append({"öznitelik": f, "seçili": f in sel, "ort_|φ|_müdahaleci": np.abs(a).mean(),
                     "ort_|φ|_yol_bağımlı": np.abs(b).mean(), "ort_|fark|": np.abs(a - b).mean(),
                     "korelasyon": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan,
                     "işaret_uyuşmazlığı": float(((np.sign(a) != np.sign(b)) & (np.maximum(abs(a), abs(b)) >= SIGN_EPS)).mean())
                     if f in sel else np.nan})
    tab = pd.DataFrame(rows).sort_values("ort_|φ|_müdahaleci", ascending=False)
    p4 = ROOT / "results" / ("odev4_fast" if fast else "odev4") / "ale_vs_pdp.csv"
    rel = None
    if p4.exists():
        r2 = pd.read_csv(p4).set_index("öznitelik")["R2_diğerleri"]
        tab["R2_diğerleri(Ödev4)"] = tab["öznitelik"].map(r2)
        t = tab[tab["seçili"]]
        rel = float(spearmanr(t["R2_diğerleri(Ödev4)"], t["ort_|fark|"] / t["ort_|φ|_müdahaleci"].clip(lower=1e-9)).statistic)
    meta = {"explainer_müdahaleci": "shap.TreeExplainer(interventional)", "değer_fonksiyonu_müdahaleci": VALUE_FN,
            "arka_plan_n_müdahaleci": nb, "taban_müdahaleci": base_int,
            "explainer_yol": "shap.TreeExplainer(tree_path_dependent)", "değer_fonksiyonu_yol": VALUE_FN_PD,
            "arka_plan_yol": "yok (eğitim verisinin ağaç örtüsü)", "taban_yol": base_pd,
            "açıklanan_n": n_explain,
            "sabit_özniteliklere_atanan_pay_yol": float(np.abs(s_pd[:, fixed_idx]).sum(1).mean() / np.abs(s_pd).sum(1).mean()),
            "sabit_özniteliklere_atanan_pay_müdahaleci": float(np.abs(s_int[:, fixed_idx]).sum(1).mean() / np.abs(s_int).sum(1).mean()),
            "spearman(R2_diğerleri, göreli_fark)": rel}
    tab.to_csv(out / "treeshap_path_vs_interventional.csv", index=False)
    # kişiler için iki türün yan yana değerleri
    prow = []
    for pid, p in persons.iterrows():
        xf = fm.full_rows(Xh.iloc[[int(p["holdout_sıra"])]][sel].to_numpy(float))
        a = np.asarray(ex_int.shap_values(xf, check_additivity=False)).reshape(-1)
        b = np.asarray(ex_pd.shap_values(xf, check_additivity=False)).reshape(-1)
        for j, f in enumerate(FEATURES):
            prow.append({"kişi": pid, "öznitelik": f, "seçili": f in sel,
                         "phi_müdahaleci": a[j], "taban_müdahaleci": base_int, "arka_plan_n_müdahaleci": nb,
                         "phi_yol_bağımlı": b[j], "taban_yol_bağımlı": base_pd})
    pd.DataFrame(prow).to_csv(out / "treeshap_persons.csv", index=False)
    # grafik
    t = tab.head(15).iloc[::-1]
    f_, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
    yv = np.arange(len(t))
    a1.barh(yv + 0.18, t["ort_|φ|_müdahaleci"], height=0.34, color=C1, label=f"müdahaleci (n_B={nb})")
    a1.barh(yv - 0.18, t["ort_|φ|_yol_bağımlı"], height=0.34, color=C2, label="yol bağımlı")
    a1.set_yticks(yv, [f + ("" if s else " (sabit)") for f, s in zip(t["öznitelik"], t["seçili"])], fontsize=6.8)
    a1.set_xlabel(f"ort. |φ| (log-odds), {n_explain} holdout satırı")
    a1.legend(fontsize=6.5, frameon=False, loc="lower right")
    a1.grid(axis="y", visible=False)
    j = tab["öznitelik"].tolist().index(tab.iloc[0]["öznitelik"])
    jj = FEATURES.index(tab.iloc[0]["öznitelik"])
    a2.scatter(s_int[:, jj], s_pd[:, jj], s=5, color=C1, alpha=0.4, lw=0)
    lim = [min(s_int[:, jj].min(), s_pd[:, jj].min()), max(s_int[:, jj].max(), s_pd[:, jj].max())]
    a2.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
    a2.set_xlabel(f"φ müdahaleci — {tab.iloc[0]['öznitelik']}")
    a2.set_ylabel("φ yol bağımlı")
    for a in (a1, a2):
        a.tick_params(labelsize=7)
    f_.tight_layout()
    f_.savefig(fig / "treeshap_variants.png")
    plt.close(f_)
    return tab, meta


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    sfx = "_fast" if args.fast else ""
    out = ROOT / "results" / f"odev5{sfx}"
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    sizes = [1, 10, 50] if args.fast else [1, 10, 100, 500]
    seeds = 2 if args.fast else 5
    nb_main = 150 if args.fast else 500
    nb_axiom = 30 if args.fast else 100

    data_kw = {"path": args.data} if args.data else {}
    Xd, yd, Xh, yh, df = get_dev_holdout(**data_kw)
    dev_idx, hold_idx = get_split(df)
    demo_d = df.iloc[dev_idx][["SEX", "AGE", "MARRIAGE", "EDUCATION"]].reset_index(drop=True)
    demo_h = df.iloc[hold_idx][["SEX", "AGE", "MARRIAGE", "EDUCATION", TARGET]].reset_index(drop=True)
    model = joblib.load(ROOT / f"models{sfx}" / "odev2_LightGBM.joblib")
    booster = model.booster_

    sel, src = select_features(model, Xh, yh, args.n_features, args.fast)
    fixed = Xd.mean()                                         # sabit değerler: dev (arka plan) ortalaması
    fm = ReducedModel(booster, FEATURES, sel, fixed)
    dev_logit = fm(Xd[sel].to_numpy(float))
    dev_prob = 1 / (1 + np.exp(-dev_logit))
    persons, cut = pick_persons(fm, Xh, demo_h, sel, dev_prob)
    persons.to_csv(out / "persons.csv")
    setup = {"seçilen_öznitelikler": sel, "seçim_kaynağı": src,
             "sabit_öznitelikler": {f: float(fixed[f]) for f in FEATURES if f not in sel},
             "sabit_değer": "dev kümesi ortalaması", "açıklanan_çıktı": "log-odds (LightGBM ham skor)",
             "ret_eşiği_olasılık": float(cut), "ağaç_sayısı": booster.num_trees(),
             "koalisyon_sayısı": 2 ** len(sel)}
    print(json.dumps(setup, ensure_ascii=False, indent=1))
    print(persons[["etiket", "olasılık", "AGE", "SEX", TARGET]].to_string())

    t0 = time.perf_counter()
    ax_tab = step1(fm, booster, persons, Xh, Xd, sel, fixed, out, nb_axiom, SEED)
    print(f"[1] aksiyom testleri ({time.perf_counter() - t0:.0f} sn):\n", ax_tab.to_string(index=False))

    t0 = time.perf_counter()
    cmp_tab, cmp = step2(fm, model, persons, Xh, Xd, sel, out, nb_main, 3 if args.fast else 7, SEED)
    print(f"[2] kendi vs shap ({time.perf_counter() - t0:.0f} sn):", cmp)

    t0 = time.perf_counter()
    agg, s3 = step3(fm, model, persons, Xh, Xd, sel, sizes, seeds, args.ref, out, fig)
    print(f"[3] arka plan boyutu ({time.perf_counter() - t0:.0f} sn):", s3)
    print(agg.round(4).to_string(index=False))

    t0 = time.perf_counter()
    yd_arr = np.asarray(yd)
    rg = step4(fm, persons, Xh, Xd, demo_d, dev_prob, yd_arr, sel, cut, nb_main, out, fig)
    print(f"[4] referans grupları ({time.perf_counter() - t0:.0f} sn):\n",
          rg[["kişi", "grup", "grup_büyüklüğü", "taban", "Σφ", "işaret_değişen_sayısı", "işaret_değişenler"]].round(3).to_string(index=False))

    t0 = time.perf_counter()
    ts_tab, ts = step5(fm, model, persons, Xh, Xd, sel, 300 if args.fast else 1000, nb_main, out, fig, args.fast)
    print(f"[5] TreeSHAP türleri ({time.perf_counter() - t0:.0f} sn):", ts)
    print(ts_tab.head(12).round(4).to_string(index=False))

    (out / "summary.json").write_text(json.dumps({
        "kurulum": setup, "aksiyomlar": ax_tab.to_dict("records"), "kendi_vs_shap": cmp,
        "arka_plan_boyutu": s3, "treeshap": ts}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("bitti →", out)


if __name__ == "__main__":
    main()
