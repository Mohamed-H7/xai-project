"""
Ödev 2 — Rashomon açığının ölçülmesi
-----------------------------------
Adımlar:
  1) Veri + tanımsız kod raporu
  2) Hiperparametre araması (Optuna/TPE), bütçe = deneme sayısı:
       LightGBM N_BB, EBM ve LASSO her biri N_BB/2  (kritik koşul: >= yarı)
     Her deneme aynı iç 3-katlı tabakalı ÇD ile (aynı katlar) değerlendirilir.
  3) Seçilen ayarlarla 5x10 tekrarlı tabakalı ÇD (eşleştirilmiş katlar)
  4) Fark + Nadeau-Bengio düzeltilmiş %95 GA
  5) Duyarlılık: korunan öznitelikler (SEX, MARRIAGE) eklenirse kara kutu ne kazanır?
  6) Nihai modeller dev kümesinin tamamında eğitilip kaydedilir (Ödev 3-5 için)
  7) Görseller: EBM şekil fonksiyonları, LASSO katsayıları, fark dağılımı

Çalıştırma:
  python odev2_rashomon.py            # tam deney
  python odev2_rashomon.py --fast     # hızlı duman testi
  python odev2_rashomon.py --retune   # önbelleği yok sayıp aramayı yeniden yap
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

from src.data import FEATURES, PROTECTED, ROOT, SEED, TARGET, get_dev_holdout, get_split, undocumented_code_report
from src.evaluation import corrected_ci, metrics
from src.models import MODELS, build_lgbm

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

COLORS = {"LightGBM": "#2a78d6", "EBM": "#eb6834", "LASSO": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--retune", action="store_true")
    ap.add_argument("--data", default=None, help="xls/csv yolu (varsayılan: data/...xls)")
    return ap.parse_args()


# ------------------------------------------------------------------ 2) arama
def tune(name, X, y, n_trials, inner_cv, out_dir):
    space, build = MODELS[name]

    def objective(trial):
        model = build(space(trial))
        return cross_val_score(model, X, y, cv=inner_cv, scoring="roc_auc", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    wall = time.perf_counter() - t0
    study.trials_dataframe().to_csv(out_dir / f"tuning_{name}.csv", index=False)
    return {
        "model": name, "n_trials": n_trials, "inner_folds": inner_cv.get_n_splits(),
        "n_fits": n_trials * inner_cv.get_n_splits(), "wall_sec": round(wall, 1),
        "best_inner_auc": study.best_value, "best_params": study.best_params,
    }


# ------------------------------------------------------------------ 3) tekrarlı ÇD
def repeated_cv(params, X, y, cv):
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        for name, (_, build) in MODELS.items():
            t0 = time.perf_counter()
            model = build(params[name]).fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
            rows.append({"fold": fold, "model": name, "fit_sec": time.perf_counter() - t0,
                         **metrics(y[te], p)})
        print(f"  kat {fold + 1}/{cv.get_n_splits()} tamam", flush=True)
    return pd.DataFrame(rows)


def compare(cv_df, n_train, n_test):
    wide = {m: cv_df.pivot(index="fold", columns="model", values=m)
            for m in ["auc", "logloss", "brier", "caught_at_20"]}
    rows = []
    for interp in ["EBM", "LASSO"]:
        # pozitif fark = kara kutu daha iyi (log-kayıp ve Brier'de işaret ters çevrildi)
        for metric, sign in [("auc", 1), ("logloss", -1), ("brier", -1), ("caught_at_20", 1)]:
            d = sign * (wide[metric]["LightGBM"] - wide[metric][interp])
            rows.append({"karşılaştırma": f"LightGBM − {interp}", "metrik": metric,
                         **corrected_ci(d.to_numpy(), n_train, n_test)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5) duyarlılık
def protected_sensitivity(df, dev_idx, lgbm_params, cv):
    """Aynı LightGBM ayarlarıyla, SEX+MARRIAGE eklenince AUC değişimi."""
    from sklearn.model_selection import cross_val_score
    dev = df.iloc[dev_idx].reset_index(drop=True)
    y = dev[TARGET].to_numpy()
    out = {}
    for label, cols in [("21 öznitelik", FEATURES), ("+SEX,MARRIAGE", FEATURES + PROTECTED)]:
        s = cross_val_score(build_lgbm(lgbm_params), dev[cols], y, cv=cv, scoring="roc_auc")
        out[label] = s
    d = out["+SEX,MARRIAGE"] - out["21 öznitelik"]
    return {"auc_21": out["21 öznitelik"].mean(), "auc_23": out["+SEX,MARRIAGE"].mean(),
            "diff_mean": d.mean(), "diff_sd": d.std(ddof=1), "n_folds": len(d)}


# ------------------------------------------------------------------ 7) görseller
def ebm_shape_data(ebm, term_idx):
    """EBM terim skorlarını (log-odds) çizime hazır hale getirir: kesim noktaları + bag std."""
    feat_idx = ebm.term_features_[term_idx]
    scores = ebm.term_scores_[term_idx]
    sds = ebm.standard_deviations_[term_idx]
    if len(feat_idx) != 1:
        return None
    f = feat_idx[0]
    if ebm.feature_types_in_[f] == "nominal":
        cats = list(ebm.bins_[f][0].keys())
        return {"kind": "nominal", "x": cats, "y": scores[1:len(cats) + 1],
                "sd": sds[1:len(cats) + 1]}
    cuts = np.asarray(ebm.bins_[f][0])
    lo, hi = ebm.feature_bounds_[f]
    edges = np.concatenate([[lo], cuts, [hi]])
    return {"kind": "continuous", "x": edges, "y": scores[1:len(edges)], "sd": sds[1:len(edges)]}


def plot_ebm_shapes(ebm, out_path, top_k=6):
    imp = ebm.term_importances()
    mains = [i for i in np.argsort(-imp) if len(ebm.term_features_[i]) == 1][:top_k]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    for ax, i in zip(axes.ravel(), mains):
        d = ebm_shape_data(ebm, i)
        name = ebm.term_names_[i]
        if d["kind"] == "nominal":
            ax.bar(range(len(d["x"])), d["y"], yerr=d["sd"], color=COLORS["EBM"],
                   width=0.6, error_kw={"ecolor": MUTED, "lw": 0.8})
            ax.set_xticks(range(len(d["x"])), d["x"])
        else:
            x, y, sd = d["x"], d["y"], d["sd"]
            ax.stairs(y + sd, x, baseline=y - sd, fill=True, color=COLORS["EBM"], alpha=0.25, lw=0)
            ax.stairs(y, x, baseline=None, color=COLORS["EBM"], lw=1.6)
            if name.startswith("LIMIT"):
                ax.set_xscale("log")
            elif name.startswith(("BILL", "PAY_AMT")):
                # para sütunları çarpık: 0 ve negatifleri de gösteren simetrik log ekseni
                ax.set_xscale("symlog", linthresh=1000)
                ticks = [t for t in (-1e5, 0, 1e3, 1e4, 1e5, 1e6) if x[0] <= t <= x[-1]]
                ax.set_xticks(ticks, [f"{t:,.0f}".replace(",", ".") for t in ticks])
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(f"{name}  (önem {imp[i]:.3f})", fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("katkı (log-odds)")
    axes[1, 0].set_ylabel("katkı (log-odds)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return [ebm.term_names_[i] for i in mains]


def plot_ebm_importance(ebm, out_path, top_k=12):
    imp = ebm.term_importances()
    order = np.argsort(-imp)[:top_k][::-1]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.barh([ebm.term_names_[i] for i in order], imp[order], color=COLORS["EBM"], height=0.6)
    ax.set_xlabel("ort. |katkı| (log-odds)")
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", visible=False)
    fig.savefig(out_path)
    plt.close(fig)


def lasso_table(lasso):
    names = lasso.named_steps["feat"].get_feature_names_out()
    coef = lasso.named_steps["clf"].coef_.ravel()
    scale = lasso.named_steps["scale"].scale_
    tab = pd.DataFrame({"öznitelik": names, "katsayı_std": coef,
                        "katsayı_ham": coef / scale, "odds_oranı_1sd": np.exp(coef)})
    tab = tab[tab["katsayı_std"] != 0].sort_values("katsayı_std", key=np.abs, ascending=False)
    return tab


def plot_lasso(tab, out_path):
    t = tab.iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.6, max(2.0, 0.18 * len(t) + 0.6)))
    ax.barh(t["öznitelik"], t["katsayı_std"], color=COLORS["LASSO"], height=0.6)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("standartlaştırılmış katsayı (log-odds / 1 sd)")
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", visible=False)
    fig.savefig(out_path)
    plt.close(fig)


def plot_diffs(cv_df, comp, out_path):
    wide = cv_df.pivot(index="fold", columns="model", values="auc")
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    rng = np.random.default_rng(0)
    for k, interp in enumerate(["EBM", "LASSO"]):
        d = wide["LightGBM"] - wide[interp]
        ax.scatter(d, k + rng.uniform(-0.12, 0.12, len(d)), s=10, color=COLORS[interp],
                   alpha=0.6, lw=0)
        r = comp[(comp["karşılaştırma"] == f"LightGBM − {interp}") & (comp["metrik"] == "auc")].iloc[0]
        ax.errorbar(r["mean"], k + 0.32, xerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
                    fmt="o", color=INK, ms=4, capsize=3, lw=1.2)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks([0, 1], ["LightGBM − EBM", "LightGBM − LASSO"])
    ax.set_xlabel("AUC farkı (kat başına; siyah: ort. ± düzeltilmiş %95 GA)")
    ax.grid(axis="y", visible=False)
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    args = parse_args()
    suffix = "_fast" if args.fast else ""      # duman testi gerçek sonuçların üzerine yazmasın
    out_dir = ROOT / "results" / f"odev2{suffix}"
    fig_dir = out_dir / "figures"
    model_dir = ROOT / f"models{suffix}"
    for d in (out_dir, fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_bb = 6 if args.fast else 60
    n_interp = n_bb // 2                                   # kritik koşul: >= yarı
    reps = 2 if args.fast else 10

    # 1) veri
    data_kw = {"path": args.data} if args.data else {}
    X, y, X_hold, y_hold, df = get_dev_holdout(**data_kw)
    dev_idx, _ = get_split(df)
    summary = {
        "n_total": len(df), "n_dev": len(X), "n_holdout": len(X_hold),
        "default_rate": float(df[TARGET].mean()), "features": FEATURES,
        "excluded_protected": PROTECTED,
        "undocumented_codes": undocumented_code_report(df).to_dict("records"),
    }
    (out_dir / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"dev={len(X)}  holdout={len(X_hold)}  temerrüt oranı={summary['default_rate']:.3f}")

    # 2) arama (önbellekli)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    budget_path = out_dir / "budget.json"
    if budget_path.exists() and not args.retune and not args.fast:
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
    else:
        budget = []
        for name, n in [("LightGBM", n_bb), ("EBM", n_interp), ("LASSO", n_interp)]:
            print(f"arama: {name} ({n} deneme)", flush=True)
            budget.append(tune(name, X, y, n, inner_cv, out_dir))
            print(f"  en iyi iç AUC={budget[-1]['best_inner_auc']:.4f}  süre={budget[-1]['wall_sec']}s")
        budget_path.write_text(json.dumps(budget, ensure_ascii=False, indent=2), encoding="utf-8")
    params = {b["model"]: b["best_params"] for b in budget}

    # 3) tekrarlı ÇD
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=SEED)
    cv_path = out_dir / "cv_folds.csv"
    if cv_path.exists() and not args.retune and not args.fast:
        cv_df = pd.read_csv(cv_path)
    else:
        print(f"tekrarlı ÇD: 5x{reps}", flush=True)
        cv_df = repeated_cv(params, X, y, cv)
        cv_df.to_csv(cv_path, index=False)

    # 4) fark + GA
    n_test = len(X) // 5
    comp = compare(cv_df, n_train=len(X) - n_test, n_test=n_test)
    comp.to_csv(out_dir / "comparison.csv", index=False)
    perf = cv_df.groupby("model")[["auc", "logloss", "brier", "caught_at_20", "fit_sec"]].agg(["mean", "std"])
    perf.to_csv(out_dir / "performance.csv")
    print(perf.round(4))
    print(comp[comp["metrik"] == "auc"].round(4).to_string())

    # 5) duyarlılık
    sens_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED + 1)
    sens = protected_sensitivity(df, dev_idx, params["LightGBM"], sens_cv)
    (out_dir / "protected_sensitivity.json").write_text(json.dumps(sens, indent=2), encoding="utf-8")
    print("korunan öznitelik duyarlılığı:", {k: round(v, 4) for k, v in sens.items()})

    # 6) nihai modeller (Ödev 3-5 bunları yükler)
    final = {name: build(params[name]).fit(X, y) for name, (_, build) in MODELS.items()}
    for name, m in final.items():
        joblib.dump(m, model_dir / f"odev2_{name}.joblib")

    # 7) görseller
    shown = plot_ebm_shapes(final["EBM"], fig_dir / "ebm_shapes.png")
    plot_ebm_importance(final["EBM"], fig_dir / "ebm_importance.png")
    ebm_terms = pd.DataFrame({"terim": final["EBM"].term_names_,
                              "önem": final["EBM"].term_importances()}).sort_values("önem", ascending=False)
    ebm_terms.to_csv(out_dir / "ebm_terms.csv", index=False)
    tab = lasso_table(final["LASSO"])
    tab.to_csv(out_dir / "lasso_coefficients.csv", index=False)
    plot_lasso(tab, fig_dir / "lasso_coefficients.png")
    plot_diffs(cv_df, comp, fig_dir / "auc_differences.png")
    print("EBM şekil fonksiyonları çizilen terimler:", shown)
    print(f"LASSO sıfır olmayan katsayı: {len(tab)}")
    print("bitti →", out_dir)


if __name__ == "__main__":
    main()
