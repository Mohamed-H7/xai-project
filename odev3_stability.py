"""
Ödev 3 — Yorumlanabilir model tasarımı ve kararlılık analizi
-----------------------------------------------------------
Ödev 2 ile aynı veri, aynı holdout (dokunulmaz) ve Ödev 2'de seçilen EBM/LASSO ayarları.

  1) Ağaç modeli seçimi: GOSDT (eşik tahmini ile) ve budanmış CART, 3-katlı ÇD
  2) Ayrıklaştırma: GOSDT için eşik stratejilerinin sonuca etkisi
  3) Kararlılık: B=50 önyükleme (bootstrap)
       (i)  LASSO: seçilme sıklığı, katsayı aralığı
       (ii) GOSDT/CART: kök ve 1. düzey bölünme değişkenleri
       (iii) EBM: şekil fonksiyonlarının nokta bazında değişim aralığı
     + her tekrarda torba dışı (OOB) başarım
  4) Alan bilgisiyle çelişen örüntü araması (sistematik, önyüklemeye dayalı)
  5) Tek sayfalık risk puan kartı (EBM'den türetilmiş) + basitleştirmenin bedeli

Çalıştırma:
  python odev3_stability.py           # tam (önce odev2_rashomon.py çalışmış olmalı)
  python odev3_stability.py --fast    # duman testi
  python odev3_stability.py --rerun   # önbellekleri yok say
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data import FEATURES, PAY_COLS, ROOT, SEED, TARGET, get_dev_holdout
from src.models import build_ebm, build_lasso
from src.scorecard import EBMScorecard, interval_labels
from src.trees import GOSDTModel, build_cart, cart_splits, var_of

warnings.filterwarnings("ignore")

COLORS = {"EBM": "#eb6834", "LASSO": "#1baf7a", "GOSDT": "#6250d6", "CART": "#eda100"}
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": FAINT, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

# Alan bilgisine dayalı beklenen yön: +1 = değer arttıkça risk artar, -1 = azalır, None = ön bilgi yok
EXPECTED = {
    **{c: +1 for c in PAY_COLS},                         # daha uzun gecikme → daha riskli
    "LIMIT_BAL": -1,                                     # yüksek limit → banka daha güvenilir bulmuş
    **{f"PAY_AMT{k}": -1 for k in range(1, 7)},          # daha çok ödeme → daha az risk
    "EDUCATION": +1,                                     # 1 lisansüstü < 2 üniversite < 3 lise (4 'diğer' hariç)
    "AGE": None,
    **{f"BILL_AMT{k}": None for k in range(1, 7)},       # tutar tek başına belirsiz (limitle karışık)
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--data", default=None)
    return ap.parse_args()


def week2_params(fast: bool) -> dict:
    path = ROOT / "results" / ("odev2_fast" if fast else "odev2") / "budget.json"
    if not path.exists():
        raise SystemExit(f"{path} yok — önce odev2_rashomon.py çalıştırın.")
    return {b["model"]: b["best_params"] for b in json.loads(path.read_text(encoding="utf-8"))}


def cv_auc(make, X, y, folds):
    s = []
    for tr, te in folds:
        m = make().fit(X.iloc[tr], y[tr])
        s.append(roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(s)), float(np.std(s, ddof=1))


# ------------------------------------------------------------------ 1) ağaç seçimi
def select_trees(X, y, folds, fast):
    rows = []
    cart_grid = [{"max_depth": d, "min_samples_leaf": m} for d in (2, 3, 4) for m in (50, 200, 500)]
    gosdt_grid = [{"regularization": r, "balance": b}
                  for r in ((0.002, 0.005) if fast else (0.0005, 0.001, 0.002, 0.005))
                  for b in (False, True)]
    for p in cart_grid:
        auc, sd = cv_auc(lambda: build_cart(p, SEED), X, y, folds)
        rows.append({"model": "CART", "params": json.dumps(p), "auc": auc, "sd": sd})
    for p in gosdt_grid:
        t0 = time.perf_counter()
        auc, sd = cv_auc(lambda: GOSDTModel(depth_budget=4, strategy="tg", **p), X, y, folds)
        rows.append({"model": "GOSDT", "params": json.dumps(p), "auc": auc, "sd": sd,
                     "sec": time.perf_counter() - t0})
    df = pd.DataFrame(rows)
    best = {m: json.loads(df[df.model == m].sort_values("auc").iloc[-1]["params"]) for m in ("CART", "GOSDT")}
    return df, best


# ------------------------------------------------------------------ 2) ayrıklaştırma
def discretization_effect(X, y, folds, gosdt_params):
    strategies = [("tg", {"n_thresholds_est": 20}), ("tg", {"n_thresholds_est": 40}),
                  ("tg", {"n_thresholds_est": 80}), ("quantile", {"n_bins": 3}),
                  ("quantile", {"n_bins": 5})]
    rows, texts = [], {}
    for strat, kw in strategies:
        label = f"{strat}-" + str(list(kw.values())[0])
        mk = lambda: GOSDTModel(depth_budget=4, strategy=strat, **kw, **gosdt_params)
        t0 = time.perf_counter()
        auc, sd = cv_auc(mk, X, y, folds)
        full = mk().fit(X, y)
        rows.append({"strateji": label, "ikili_öznitelik": len(full.bin_columns_), "cv_auc": auc,
                     "cv_sd": sd, "yaprak": full.n_leaves(), "kök": full.root_split(),
                     "durum": full.status_, "süre_sn": round(time.perf_counter() - t0, 1)})
        texts[label] = full.to_text()
    return pd.DataFrame(rows), texts


# ------------------------------------------------------------------ 3) önyükleme
def ebm_grid(X: pd.DataFrame) -> dict:
    """Her değişken için şekil fonksiyonunun değerlendirileceği sabit ızgara."""
    grids = {}
    for f in FEATURES:
        v = X[f]
        if f in PAY_COLS or f == "EDUCATION":
            counts = v.value_counts()
            grids[f] = np.array(sorted(counts[counts >= 30].index), dtype=float)
        elif f == "AGE":
            grids[f] = np.arange(v.quantile(0.01), v.quantile(0.99) + 1, 2, dtype=float)
        else:
            grids[f] = np.unique(v.quantile(np.linspace(0.01, 0.99, 30)).to_numpy(dtype=float))
    return grids


def eval_shapes(ebm, X, grids) -> dict:
    base = X.median(numeric_only=True)
    out = {}
    term_of = {ebm.term_names_[i]: i for i, tf in enumerate(ebm.term_features_) if len(tf) == 1}
    for f, g in grids.items():
        Xg = pd.DataFrame([base] * len(g)).reset_index(drop=True)[FEATURES]
        Xg[f] = g
        out[f] = ebm.eval_terms(Xg)[:, term_of[f]]
    return out


def bootstrap(X, y, params, tree_params, B, out_dir):
    rng = np.random.default_rng(SEED)
    grids = ebm_grid(X)
    n = len(X)
    shapes = {f: [] for f in FEATURES}
    lasso_rows, tree_rows, perf_rows = [], [], []
    for b in range(B):
        idx = rng.integers(0, n, n)
        counts = np.bincount(idx, minlength=n)
        oob = counts == 0
        Xb, yb = X.iloc[idx].reset_index(drop=True), y[idx]
        t0 = time.perf_counter()

        # EBM: tekrar eden satırlar iç doğrulama bölmesine sızmasın diye ağırlıkla eğitilir
        keep = counts > 0
        ebm = build_ebm(params["EBM"], seed=SEED + b).fit(X.loc[keep], y[keep], sample_weight=counts[keep].astype(float))
        for f, v in eval_shapes(ebm, X, grids).items():
            shapes[f].append(v)

        lasso = build_lasso(params["LASSO"], seed=SEED + b).fit(Xb, yb)
        names = lasso.named_steps["feat"].get_feature_names_out()
        for nm, c in zip(names, lasso.named_steps["clf"].coef_.ravel()):
            lasso_rows.append({"boot": b, "sütun": nm, "katsayı": c})

        cart = build_cart(tree_params["CART"], seed=SEED + b).fit(Xb, yb)
        root, first = cart_splits(cart, FEATURES)
        tree_rows.append({"boot": b, "model": "CART", "kök": root, "birinci": first,
                          "yaprak": int(cart.get_n_leaves())})

        gosdt = GOSDTModel(depth_budget=4, strategy="tg", random_state=b, **tree_params["GOSDT"]).fit(Xb, yb)
        tree_rows.append({"boot": b, "model": "GOSDT", "kök": gosdt.root_split(),
                          "birinci": gosdt.first_level_splits(), "yaprak": gosdt.n_leaves(),
                          "durum": gosdt.status_})

        for name, m in [("EBM", ebm), ("LASSO", lasso), ("CART", cart), ("GOSDT", gosdt)]:
            perf_rows.append({"boot": b, "model": name,
                              "oob_auc": roc_auc_score(y[oob], m.predict_proba(X.loc[oob])[:, 1])})
        print(f"  önyükleme {b + 1}/{B}  ({time.perf_counter() - t0:.0f} sn)", flush=True)

    np.savez_compressed(out_dir / "boot_ebm_shapes.npz",
                        **{f"grid__{f}": g for f, g in grids.items()},
                        **{f"shape__{f}": np.vstack(v) for f, v in shapes.items()})
    pd.DataFrame(lasso_rows).to_csv(out_dir / "boot_lasso.csv", index=False)
    tr = pd.DataFrame(tree_rows)
    tr["birinci"] = tr["birinci"].apply(json.dumps, ensure_ascii=False)
    tr.to_csv(out_dir / "boot_trees.csv", index=False)
    pd.DataFrame(perf_rows).to_csv(out_dir / "boot_perf.csv", index=False)


def load_shapes(out_dir):
    z = np.load(out_dir / "boot_ebm_shapes.npz")
    return ({k[6:]: z[k] for k in z.files if k.startswith("grid__")},
            {k[7:]: z[k] for k in z.files if k.startswith("shape__")})


# ------------------------------------------------------------------ 3) özetler
def summarize_lasso(df, B):
    g = df.groupby("sütun")["katsayı"]
    tab = pd.DataFrame({
        "seçilme_sıklığı": g.apply(lambda s: (s != 0).mean()),
        "medyan": g.median(), "p05": g.quantile(0.05), "p95": g.quantile(0.95),
        "min": g.min(), "max": g.max(),
        # sıfır olmayan katsayıların işaret tutarlılığı
        "işaret_tutarlılığı": g.apply(lambda s: max((s > 0).sum(), (s < 0).sum()) / max((s != 0).sum(), 1)),
    }).sort_values("seçilme_sıklığı", ascending=False)
    # özgün değişken düzeyinde (PAY_1 → PAY_1_gecikme, PAY_1=-1, PAY_1=-2 birleşir)
    df = df.assign(değişken=df["sütun"].str.replace(r"(_gecikme|=-1|=-2)$", "", regex=True)
                   .str.replace(r"^EDUCATION=.*", "EDUCATION", regex=True))
    var = (df.groupby(["boot", "değişken"])["katsayı"].apply(lambda s: (s != 0).any())
           .groupby("değişken").mean().sort_values(ascending=False).rename("seçilme_sıklığı"))
    n_sel = df[df["katsayı"] != 0].groupby("boot")["sütun"].nunique()
    return tab, var, {"sıfırdan_farklı_katsayı_medyan": float(n_sel.median()),
                      "min": int(n_sel.min()), "max": int(n_sel.max()), "B": B}


def summarize_trees(df):
    rows = []
    for m in ("GOSDT", "CART"):
        d = df[df.model == m]
        B = len(d)
        root_var = Counter(var_of(r) if isinstance(r, str) else "(bölünme yok)" for r in d["kök"])
        root_full = Counter(r if isinstance(r, str) else "(bölünme yok)" for r in d["kök"])
        first = Counter(v for lst in d["birinci"] for v in {var_of(s) for s in json.loads(lst)})
        for k, v in root_var.most_common():
            rows.append({"model": m, "düzey": "kök (değişken)", "bölünme": k, "oran": v / B})
        for k, v in root_full.most_common(5):
            rows.append({"model": m, "düzey": "kök (değişken+eşik)", "bölünme": k, "oran": v / B})
        for k, v in first.most_common():
            rows.append({"model": m, "düzey": "1. düzey (değişken)", "bölünme": k, "oran": v / B})
        rows.append({"model": m, "düzey": "yaprak sayısı", "bölünme": f"{d['yaprak'].min()}–{d['yaprak'].max()}",
                     "oran": float(d["yaprak"].median())})
    return pd.DataFrame(rows)


def summarize_ebm(grids, shapes):
    rows = []
    for f, S in shapes.items():
        lo, med, hi = np.percentile(S, [5, 50, 95], axis=0)
        width = hi - lo
        rows.append({"değişken": f, "ızgara_nokta": len(grids[f]),
                     "şekil_genliği_medyan": float(med.max() - med.min()),
                     "bant_genişliği_ort": float(width.mean()), "bant_genişliği_maks": float(width.max()),
                     "maks_bant_konumu": float(grids[f][width.argmax()]),
                     "sıfırı_kesen_nokta_oranı": float(((lo < 0) & (hi > 0)).mean()),
                     "göreli_belirsizlik": float(width.mean() / max(med.max() - med.min(), 1e-9))})
    return pd.DataFrame(rows).sort_values("şekil_genliği_medyan", ascending=False)


# ------------------------------------------------------------------ 4) çelişki araması
def contradiction_search(grids, shapes, X, min_effect=0.05):
    """
    Beklenen yönü olan her değişken için tüm (a < b) ızgara çiftlerinde
    d = yön × (f(b) − f(a)) önyükleme dağılımına bakılır. Önyüklemelerin ≥%95'inde d < 0
    ise ve medyan büyüklük ≥ min_effect log-odds ise çelişki 'sağlam' sayılır.
    """
    rows = []
    for f, S in shapes.items():
        direction = EXPECTED.get(f)
        g = grids[f]
        if direction is None:
            continue
        if f == "EDUCATION":                       # 'diğer' (4) sıralamaya katılmaz
            keep = g <= 3
            g, S = g[keep], S[:, keep]
        best = None
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                d = direction * (S[:, j] - S[:, i])
                share = float((d < 0).mean())
                mag = float(-np.median(d))
                if share >= 0.95 and mag >= min_effect and (best is None or mag > best["büyüklük"]):
                    best = {"değişken": f, "beklenen_yön": "+" if direction > 0 else "−",
                            "a": g[i], "b": g[j], "büyüklük": mag, "önyükleme_oranı": share,
                            "n_a": int((X[f] == g[i]).sum()) if f in PAY_COLS + ["EDUCATION"] else None,
                            "n_b": int((X[f] == g[j]).sum()) if f in PAY_COLS + ["EDUCATION"] else None}
        rows.append(best or {"değişken": f, "beklenen_yön": "+" if direction > 0 else "−",
                             "a": None, "b": None, "büyüklük": 0.0, "önyükleme_oranı": None})
    return pd.DataFrame(rows).sort_values("büyüklük", ascending=False)


def diagnose(df_dev: pd.DataFrame, f: str) -> pd.DataFrame:
    """Bulunan örüntünün ham veride de olup olmadığı ve olası açıklayıcılar."""
    key = df_dev[f] if df_dev[f].nunique() <= 12 else pd.qcut(df_dev[f], 10, duplicates="drop")
    g = df_dev.groupby(key, observed=True)
    out = pd.DataFrame({
        "n": g.size(), "ham_temerrüt_oranı": g[TARGET].mean(),
        "medyan_LIMIT_BAL": g["LIMIT_BAL"].median(), "medyan_BILL_AMT1": g["BILL_AMT1"].median(),
        "medyan_PAY_AMT1": g["PAY_AMT1"].median(),
        "BILL_AMT1<=0_oranı": g["BILL_AMT1"].apply(lambda s: (s <= 0).mean()),
        "kullanım_oranı_medyan": g.apply(lambda d: (d["BILL_AMT1"] / d["LIMIT_BAL"]).median()),
    })
    if f in PAY_COLS and f != "PAY_6":
        nxt = f"PAY_{int(f[-1]) + 1}"
        out[f"önceki_ay_{nxt}>=1_oranı"] = g[nxt].apply(lambda s: (s >= 1).mean())
    return out


# ------------------------------------------------------------------ 5) puan kartı
def scorecard_experiment(X, y, params, folds, steps=(2, 3, 4, 6)):
    p_main = {**params["EBM"], "interactions": 0}
    rows = []
    for k, (tr, te) in enumerate(folds):
        ebm_full = build_ebm(params["EBM"]).fit(X.iloc[tr], y[tr])
        ebm_main = build_ebm(p_main).fit(X.iloc[tr], y[tr])
        rows.append({"kat": k, "model": "EBM (Ödev 2 ayarı)",
                     "auc": roc_auc_score(y[te], ebm_full.predict_proba(X.iloc[te])[:, 1])})
        rows.append({"kat": k, "model": "EBM (etkileşimsiz)",
                     "auc": roc_auc_score(y[te], ebm_main.predict_proba(X.iloc[te])[:, 1])})
        for s in steps:
            sc = EBMScorecard(top_k=8, max_steps=s).fit(ebm_main, X.iloc[tr], y[tr])
            rows.append({"kat": k, "model": f"puan kartı (8 değişken, ≤{s} basamak)",
                         "auc": roc_auc_score(y[te], sc.predict_proba(X.iloc[te])[:, 1])})
    df = pd.DataFrame(rows)
    return df.groupby("model", sort=False)["auc"].agg(["mean", "std"]).reset_index()


MONTHS = {1: "Eylül", 2: "Ağustos", 3: "Temmuz", 4: "Haziran", 5: "Mayıs", 6: "Nisan"}
TR_NAMES = {
    "LIMIT_BAL": "Kredi limiti (NT$)", "AGE": "Yaş", "EDUCATION": "Eğitim",
    **{f"PAY_{k}": f"Ödeme durumu — {MONTHS[k]}" for k in range(1, 7)},
    **{f"BILL_AMT{k}": f"Ekstre tutarı — {MONTHS[k]} (NT$)" for k in range(1, 7)},
    **{f"PAY_AMT{k}": f"Ödenen tutar — {MONTHS[k]} (NT$)" for k in range(1, 7)},
}


def render_scorecard(sc: EBMScorecard, X, y, ebm_main, cv_tab, out_base):
    """A4 tek sayfa: puan tabloları + puan→olasılık + çözümlü örnek."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.965, "Kredi kartı temerrüt riski — puan kartı", fontsize=15, weight="bold", color=INK)
    fig.text(0.06, 0.945, "Etkileşimsiz EBM'den türetilmiştir. Tayvan, 2005 verisi; yalnızca ders amaçlıdır.",
             fontsize=8.5, color=MUTED)
    fig.text(0.06, 0.922, "Kullanım: (1) Her değişkende başvurana uyan satırın puanını alın.  "
             "(2) Puanları toplayın.  (3) Toplamı sağ alttaki tablodan olasılığa çevirin.",
             fontsize=8.5, color=INK)
    # değişken tabloları: 2 sütun × 4 satır
    tab = sc.table()
    for k, f in enumerate(sc.features_):
        col, row = k % 2, k // 2
        x0, y0 = 0.06 + col * 0.46, 0.885 - row * 0.15
        fig.text(x0, y0, TR_NAMES.get(f, f), fontsize=9, weight="bold", color=INK)
        fig.text(x0, y0 - 0.016, f, fontsize=7.5, color=MUTED, family="monospace")
        fig.text(x0 + 0.38, y0 - 0.016, "puan", fontsize=7.5, color=MUTED, ha="right")
        t = tab[tab["değişken"] == f]
        for r, (_, line) in enumerate(t.iterrows()):
            yy = y0 - 0.036 - r * 0.018
            fig.text(x0 + 0.01, yy, line["aralık"], fontsize=8.5, color=INK)
            fig.text(x0 + 0.38, yy, f"{line['puan']:d}", fontsize=8.5, color=INK, ha="right")
    # puan → olasılık
    max_p = sc.max_points()
    pts = np.unique(np.linspace(0, max_p, 12).round().astype(int))
    probs = sc.prob_of_points(pts)
    yb = 0.235
    fig.text(0.52, yb + 0.03, "Toplam puan → temerrüt olasılığı", fontsize=9, weight="bold", color=INK)
    for r, (p, pr) in enumerate(zip(pts, probs)):
        cc, rr = r % 2, r // 2
        fig.text(0.53 + cc * 0.2, yb - rr * 0.018, f"{p:>3d} puan  →  %{100 * pr:4.1f}",
                 fontsize=8.5, family="monospace", color=INK)
    # çözümlü örnek: medyan civarında riskli bir başvuru
    total = sc.total_points(X)
    i = int(np.argsort(np.abs(total - np.percentile(total, 85)))[0])
    pf = sc.points_frame(X.iloc[[i]]).iloc[0]
    fig.text(0.06, yb + 0.03, "Çözümlü örnek (gerçek bir başvuru)", fontsize=9, weight="bold", color=INK)
    for r, f in enumerate(sc.features_):
        fig.text(0.07, yb - r * 0.018, f"{f} = {X.iloc[i][f]:,.0f}".replace(",", "."), fontsize=8.5, color=INK)
        fig.text(0.30, yb - r * 0.018, f"+{pf[f]:d}", fontsize=8.5, family="monospace", color=INK)
    yy = yb - len(sc.features_) * 0.018
    p_card = sc.prob_of_points([total[i]])[0]
    p_ebm = ebm_main.predict_proba(X.iloc[[i]])[0, 1]
    fig.text(0.07, yy - 0.006, f"Toplam = {total[i]} puan → %{100 * p_card:.1f}   "
             f"(tam EBM: %{100 * p_ebm:.1f})", fontsize=8.5, weight="bold", color=INK)
    # dipnot: başarım
    auc_line = "   ".join(f"{r['model']}: AUC {r['mean']:.3f}" for _, r in cv_tab.iterrows()
                           if "≤4" in r["model"] or "EBM" in r["model"])
    fig.text(0.06, 0.035, "5-katlı ÇD (dev kümesi): " + auc_line, fontsize=7.2, color=MUTED)
    fig.text(0.06, 0.02, "Ödeme durumu kodları: −2 harcama yok, −1 tam ödendi, 0 döner kredi (asgari ödeme), "
             "1…8 = o kadar ay gecikme.  Aralık 'a – b':  a < x ≤ b.",
             fontsize=7.2, color=MUTED)
    fig.savefig(f"{out_base}.png", dpi=200)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ görseller
def plot_lasso_stability(tab, out_path, top=20):
    t = tab.head(top).iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 0.2 * len(t) + 1.0), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1.6]})
    a1.barh(t.index, t["seçilme_sıklığı"], color=COLORS["LASSO"], height=0.6)
    a1.set_xlim(0, 1)
    a1.set_xlabel("seçilme sıklığı")
    a2.hlines(t.index, t["p05"], t["p95"], color=COLORS["LASSO"], lw=3, alpha=0.5)
    a2.hlines(t.index, t["min"], t["max"], color=COLORS["LASSO"], lw=0.8)
    a2.plot(t["medyan"], t.index, "o", color=INK, ms=3)
    a2.axvline(0, color=MUTED, lw=0.6)
    a2.set_xlabel("standart katsayı")
    a2.set_title("ince: min–maks · kalın: %5–95 · nokta: medyan", fontsize=7.5, color=MUTED)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
        a.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_tree_roots(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for ax, lvl in zip(axes, ["kök (değişken)", "1. düzey (değişken)"]):
        d = df[df["düzey"] == lvl]
        cats = list(dict.fromkeys(d["bölünme"]))
        yv = np.arange(len(cats))
        for k, m in enumerate(["GOSDT", "CART"]):
            dm = d[d.model == m].set_index("bölünme")["oran"].reindex(cats).fillna(0)
            ax.barh(yv + (k - 0.5) * 0.36, dm, height=0.34, color=COLORS[m], label=m)
        ax.set_yticks(yv, cats)
        ax.invert_yaxis()
        ax.set_title(lvl, fontsize=8.5, color=INK)
        ax.set_xlabel("önyüklemelerin oranı")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def money_axis(ax, f, g):
    """Para sütunları çarpık: LIMIT için log, diğerleri için simetrik log (0 ve negatifler)."""
    if f.startswith("LIMIT"):
        ax.set_xscale("log")
    elif f.startswith(("BILL", "PAY_AMT")):
        ax.set_xscale("symlog", linthresh=1000)
        ticks = [t for t in (-1e5, -1e4, 0, 1e3, 1e4, 1e5, 1e6) if g[0] <= t <= g[-1]]
        ax.set_xticks(ticks, [f"{t:,.0f}".replace(",", ".") for t in ticks])
    else:
        return
    ax.set_xlim(g[0], g[-1])


def plot_ebm_stability(grids, shapes, summary, out_path, feats=None):
    feats = feats or list(summary["değişken"].head(6))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    for ax, f in zip(axes.ravel(), feats):
        g, S = grids[f], shapes[f]
        lo, med, hi = np.percentile(S, [5, 50, 95], axis=0)
        step = f in PAY_COLS or f == "EDUCATION"
        for s in S:
            ax.plot(g, s, color=COLORS["EBM"], lw=0.4, alpha=0.15, drawstyle="steps-mid" if step else "default")
        ax.fill_between(g, lo, hi, color=COLORS["EBM"], alpha=0.25, lw=0, step="mid" if step else None)
        ax.plot(g, med, color=COLORS["EBM"], lw=1.6, drawstyle="steps-mid" if step else "default")
        ax.axhline(0, color=MUTED, lw=0.6)
        money_axis(ax, f, g)
        ax.set_title(f, fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("katkı (log-odds)")
    axes[1, 0].set_ylabel("katkı (log-odds)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_diagnostic(diag, f, out_path):
    fig, ax = plt.subplots(figsize=(3.6, 2.3))
    labels = [str(i) for i in diag.index]
    ax.bar(labels, diag["ham_temerrüt_oranı"], color=COLORS["EBM"], width=0.6)
    for k, (r, n) in enumerate(zip(diag["ham_temerrüt_oranı"], diag["n"])):
        ax.text(k, r + 0.01, f"n={n}", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xlabel(f)
    ax.set_ylabel("ham temerrüt oranı")
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=7)
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    suffix = "_fast" if args.fast else ""
    out_dir = ROOT / "results" / f"odev3{suffix}"
    fig_dir = out_dir / "figures"
    for d in (out_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    B = 5 if args.fast else 50

    data_kw = {"path": args.data} if args.data else {}
    X, y, _, _, df = get_dev_holdout(**data_kw)
    params = week2_params(args.fast)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y))
    print(f"dev={len(X)}  Ödev 2 ayarları yüklendi: EBM={params['EBM']}  LASSO={params['LASSO']}")

    # 1) ağaç seçimi
    sel_path = out_dir / "tree_selection.csv"
    if sel_path.exists() and not args.rerun:
        sel = pd.read_csv(sel_path)
        best = {m: json.loads(sel[sel.model == m].sort_values("auc").iloc[-1]["params"]) for m in ("CART", "GOSDT")}
    else:
        print("ağaç modeli seçimi…", flush=True)
        sel, best = select_trees(X, y, folds, args.fast)
        sel.to_csv(sel_path, index=False)
    (out_dir / "tree_params.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("seçilen ağaç ayarları:", best)

    # 2) ayrıklaştırma etkisi
    disc_path = out_dir / "discretization.csv"
    if not disc_path.exists() or args.rerun:
        print("ayrıklaştırma stratejileri…", flush=True)
        disc, texts = discretization_effect(X, y, folds, best["GOSDT"])
        disc.to_csv(disc_path, index=False)
        (out_dir / "gosdt_trees.txt").write_text(
            "\n\n".join(f"=== {k} ===\n{v}" for k, v in texts.items()), encoding="utf-8")
    print(pd.read_csv(disc_path).to_string())

    # 3) önyükleme
    if not (out_dir / "boot_perf.csv").exists() or args.rerun:
        print(f"önyükleme B={B}…", flush=True)
        bootstrap(X, y, params, best, B, out_dir)
    grids, shapes = load_shapes(out_dir)
    lasso_tab, lasso_var, lasso_n = summarize_lasso(pd.read_csv(out_dir / "boot_lasso.csv"), B)
    lasso_tab.to_csv(out_dir / "stability_lasso.csv")
    lasso_var.to_csv(out_dir / "stability_lasso_by_variable.csv")
    (out_dir / "stability_lasso_size.json").write_text(json.dumps(lasso_n, ensure_ascii=False), encoding="utf-8")
    trees = summarize_trees(pd.read_csv(out_dir / "boot_trees.csv"))
    trees.to_csv(out_dir / "stability_trees.csv", index=False)
    ebm_sum = summarize_ebm(grids, shapes)
    ebm_sum.to_csv(out_dir / "stability_ebm.csv", index=False)
    perf = pd.read_csv(out_dir / "boot_perf.csv").groupby("model")["oob_auc"].agg(["mean", "std"])
    perf.to_csv(out_dir / "oob_performance.csv")
    plot_lasso_stability(lasso_tab, fig_dir / "lasso_stability.png")
    plot_tree_roots(trees, fig_dir / "tree_roots.png")
    plot_ebm_stability(grids, shapes, ebm_sum, fig_dir / "ebm_stability.png")
    print("OOB AUC:\n", perf.round(4))
    print("LASSO seçilme (değişken düzeyi):\n", lasso_var.round(2).to_string())
    print(trees[trees["düzey"].str.startswith("kök (değişken)")].to_string())

    # 4) çelişki araması
    contra = contradiction_search(grids, shapes, X)
    contra.to_csv(out_dir / "contradictions.csv", index=False)
    print("alan bilgisiyle çelişki adayları:\n", contra.round(3).to_string())
    dev_df = X.assign(**{TARGET: y})
    top = contra.iloc[0]["değişken"] if contra.iloc[0]["büyüklük"] > 0 else "PAY_1"
    diag = diagnose(dev_df, top)
    diag.to_csv(out_dir / f"diagnostic_{top}.csv")
    plot_diagnostic(diag, top, fig_dir / f"diagnostic_{top}.png")
    print(f"teşhis tablosu ({top}):\n", diag.round(3).to_string())

    # 5) puan kartı
    sc_path = out_dir / "scorecard_cv.csv"
    if sc_path.exists() and not args.rerun:
        cv_tab = pd.read_csv(sc_path)
    else:
        print("puan kartı ÇD…", flush=True)
        cv_tab = scorecard_experiment(X, y, params, list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)))
        cv_tab.to_csv(sc_path, index=False)
    print(cv_tab.round(4).to_string())
    ebm_main = build_ebm({**params["EBM"], "interactions": 0}).fit(X, y)
    sc = EBMScorecard(top_k=8, max_steps=4).fit(ebm_main, X, y)
    sc.table().to_csv(out_dir / "scorecard_table.csv", index=False)
    (ROOT / f"models{suffix}").mkdir(exist_ok=True)
    joblib.dump({"ebm_main": ebm_main, "scorecard": sc},
                ROOT / f"models{suffix}" / "odev3_scorecard.joblib")
    render_scorecard(sc, X, y, ebm_main, cv_tab, fig_dir / "scorecard_one_page")
    print("bitti →", out_dir)


if __name__ == "__main__":
    main()
