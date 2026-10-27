"""
Ödev 5 raporu için şekiller (yalnızca results/odev5/*.csv dosyalarından üretilir, model gerekmez).

  python odev5_figures.py                       # results/odev5 → results/odev5/report_figures
  python odev5_figures.py --results <klasör>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

INK, MUTED, FAINT = "#1a1a19", "#52514e", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"
PERSON = {"K1": ORANGE, "K2": "#6250d6", "K3": "#1baf7a"}
SIGN_EPS, SIGN_MIN = 0.01, 0.001
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": FAINT, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def fig_background_size(res: Path, out: Path):
    bs = pd.read_csv(res / "background_size.csv")
    persons = pd.read_csv(res / "persons.csv", index_col=0)
    agg = bs.groupby(["kişi", "boyut"])["göreli_sapma"].agg(["mean", "min", "max"]).reset_index()
    sizes = sorted(bs["boyut"].unique())
    f, ax = plt.subplots(figsize=(3.3, 2.35))
    for pid in persons.index:
        a = agg[agg["kişi"] == pid]
        ax.fill_between(a["boyut"], a["min"], a["max"], color=PERSON[pid], alpha=0.14, lw=0)
        ax.plot(a["boyut"], a["mean"], "o-", color=PERSON[pid], ms=3.2, lw=1.5,
                label=f"{pid} — {persons.loc[pid, 'etiket']}")
        t = bs[bs["kişi"] == pid]
        ax.scatter(t["boyut"], t["göreli_sapma"], s=5, color=PERSON[pid], alpha=0.55, lw=0)
    # 1/√n eğimi için kılavuz çizgi
    x = np.array([1, 500.0])
    ax.plot(x, 0.9 * x ** -0.5, color=MUTED, lw=0.8, ls=":", label="∝ 1/√|B|")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes]); ax.minorticks_off()
    ax.set_xlabel("arka plan boyutu |B|")
    ax.set_ylabel("‖φ_B − φ_ref‖ / ‖φ_ref‖")
    ax.legend(fontsize=6.3, frameon=False, loc="lower left")
    f.savefig(out / "fig1_background_size.png")
    plt.close(f)


def fig_reference_groups(res: Path, out: Path):
    ph = pd.read_csv(res / "reference_groups_phi.csv")
    persons = pd.read_csv(res / "persons.csv", index_col=0)
    feats = list(dict.fromkeys(ph["öznitelik"]))
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f4f3ef", ORANGE])
    norm = Normalize(-0.6, 0.6)                       # ±0.6 dışı doygun; değerler hücrede yazılı
    pids = list(persons.index)
    f, axes = plt.subplots(len(pids), 1, figsize=(6.9, 1.62 * len(pids)), sharex=True)
    f.subplots_adjust(hspace=0.32)
    for ax, pid in zip(axes, pids):
        d = ph[ph["kişi"] == pid]
        groups = list(dict.fromkeys(d["grup"]))
        M = np.array([[d[(d["grup"] == g) & (d["öznitelik"] == ft)]["phi"].iloc[0] for ft in feats] for g in groups])
        base = [d[d["grup"] == g]["taban"].iloc[0] for g in groups]
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        pop = M[0]
        for r in range(len(groups)):
            for c in range(len(feats)):
                v = M[r, c]
                dark = abs(v) > 0.45
                ax.text(c, r, f"{v:+.3f}" if abs(v) < 0.01 else f"{v:+.2f}",
                        ha="center", va="center", fontsize=5.6, color="white" if dark else INK)
                a, b = v, pop[c]
                if r > 0 and np.sign(a) != np.sign(b) and max(abs(a), abs(b)) >= SIGN_EPS and min(abs(a), abs(b)) >= SIGN_MIN:
                    ax.add_patch(Rectangle((c - 0.48, r - 0.46), 0.96, 0.92, fill=False, ec=INK, lw=1.3))
        ax.set_yticks(range(len(groups)), [f"{g}   taban {b:+.2f}" for g, b in zip(groups, base)], fontsize=6.2)
        ax.set_xticks(range(len(feats)), feats, rotation=35, ha="right", fontsize=6.3)
        ax.tick_params(axis="x", labelbottom=ax is axes[-1], length=0)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        p = persons.loc[pid]
        ax.set_title(f"{pid} ({p['etiket']}):  f(x) = {p['log_odds']:+.2f} log-odds,  p = {p['olasılık']:.3f}",
                     fontsize=7.5, color=INK, loc="left", pad=3)
    f.savefig(out / "fig2_reference_groups.png")
    plt.close(f)


def fig_treeshap(res: Path, out: Path):
    t = pd.read_csv(res / "treeshap_path_vs_interventional.csv")
    t = t.sort_values(["seçili", "ort_|φ|_müdahaleci", "ort_|φ|_yol_bağımlı"], ascending=[True, True, True])
    f, ax = plt.subplots(figsize=(3.3, 3.4))
    y = np.arange(len(t))
    ax.barh(y + 0.19, t["ort_|φ|_müdahaleci"], height=0.36, color=BLUE, label="müdahaleci (|B| = 500)")
    ax.barh(y - 0.19, t["ort_|φ|_yol_bağımlı"], height=0.36, color=ORANGE, label="yol bağımlı (arka plansız)")
    ax.set_yticks(y, [f + ("" if s else "  (sabit)") for f, s in zip(t["öznitelik"], t["seçili"])], fontsize=6.3)
    n_fixed = int((~t["seçili"]).sum())
    ax.axhline(n_fixed - 0.5, color=MUTED, lw=0.6, ls="--")
    ax.set_xlabel("ortalama |φ| (log-odds), 1000 holdout satırı")
    ax.legend(fontsize=6.3, frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    f.savefig(out / "fig3_treeshap.png")
    plt.close(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results" / "odev5"))
    a = ap.parse_args()
    res = Path(a.results)
    out = res / "report_figures"
    out.mkdir(exist_ok=True)
    fig_background_size(res, out)
    fig_reference_groups(res, out)
    fig_treeshap(res, out)
    print("şekiller →", out)


if __name__ == "__main__":
    main()
