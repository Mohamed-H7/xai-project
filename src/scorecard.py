"""
EBM → elle hesaplanabilir risk puan kartı.

Adımlar (raporda "ayrıklaştırma stratejisi" olarak belgelenir):
  1) Etkileşimsiz (yalnızca ana etkili) bir EBM'den en önemli top_k değişken alınır.
  2) Her şekil fonksiyonu, en çok max_steps basamaklı bir fonksiyonla yaklaşıklanır:
     x → katkı ilişkisine, veri yoğunluğuyla ağırlıklı bir regresyon ağacı (yaprak sayısı
     ≤ max_steps, her basamakta en az %min_leaf_frac gözlem) uydurulur.
  3) Para eşikleri 2 anlamlı basamağa yuvarlanır (insanın okuyabileceği sayılar).
  4) Basamak değerleri tamsayı puana çevrilir (1 puan = 1/points_per_logit log-odds),
     her değişkende en düşük puan 0 olacak biçimde kaydırılır.
  5) Toplam puan → olasılık eşlemesi tek değişkenli lojistik regresyonla yeniden
     kalibre edilir (Platt). Böylece kart tek başına tutarlıdır.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from .data import CATEGORICAL, PAY_COLS

MONEY_PREFIX = ("LIMIT_BAL", "BILL_AMT", "PAY_AMT")


def _round_sig(x: float, sig: int = 2) -> float:
    if x == 0:
        return 0.0
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


class EBMScorecard:
    def __init__(self, top_k: int = 8, max_steps: int = 4, points_per_logit: float = 10.0,
                 min_leaf_frac: float = 0.03):
        self.top_k = top_k
        self.max_steps = max_steps
        self.points_per_logit = points_per_logit
        self.min_leaf_frac = min_leaf_frac

    def fit(self, ebm, X: pd.DataFrame, y):
        contrib = ebm.eval_terms(X)
        imp = ebm.term_importances()
        mains = [i for i in np.argsort(-imp) if len(ebm.term_features_[i]) == 1]
        self.features_, self.cuts_, self.points_ = [], {}, {}
        n = len(X)
        for i in mains:
            if len(self.features_) == self.top_k:
                break
            f = ebm.term_names_[i]
            x, c = X[f].to_numpy(dtype=float), contrib[:, i]
            if f in CATEGORICAL:
                cats = sorted(np.unique(x))
                cuts = [k + 0.5 for k in cats[:-1]]                    # her kategori ayrı
            else:
                tree = DecisionTreeRegressor(max_leaf_nodes=self.max_steps,
                                             min_samples_leaf=max(1, int(self.min_leaf_frac * n)))
                tree.fit(x.reshape(-1, 1), c)
                cuts = sorted(t for t in tree.tree_.threshold if t != -2)
                if f.startswith(MONEY_PREFIX):
                    cuts = sorted(set(_round_sig(t) for t in cuts))
                elif f in PAY_COLS or f == "AGE":
                    cuts = sorted(set(math.floor(t) + 0.5 for t in cuts))  # tamsayı sınırları
            bins = np.digitize(x, cuts, right=True)
            vals = np.array([c[bins == b].mean() if (bins == b).any() else 0.0
                             for b in range(len(cuts) + 1)])
            pts = np.round(vals * self.points_per_logit).astype(int)
            pts -= pts.min()
            if pts.max() == 0:              # yuvarlamadan sonra hiç puan taşımayan değişken karta girmez
                continue
            self.features_.append(f)
            self.cuts_[f] = cuts
            self.points_[f] = pts
        # toplam puan → olasılık (Platt yeniden kalibrasyonu)
        total = self.total_points(X)
        self.calib_ = LogisticRegression(C=1e6).fit(total.reshape(-1, 1), y)
        return self

    def points_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for f in self.features_:
            b = np.digitize(X[f].to_numpy(dtype=float), self.cuts_[f], right=True)
            out[f] = self.points_[f][b]
        return pd.DataFrame(out, index=X.index)

    def total_points(self, X: pd.DataFrame) -> np.ndarray:
        return self.points_frame(X).sum(axis=1).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calib_.predict_proba(self.total_points(X).reshape(-1, 1))

    def prob_of_points(self, pts) -> np.ndarray:
        return self.calib_.predict_proba(np.asarray(pts).reshape(-1, 1))[:, 1]

    # ---------- okunabilir tablo
    def table(self) -> pd.DataFrame:
        rows = []
        for f in self.features_:
            for label, p in zip(interval_labels(f, self.cuts_[f]), self.points_[f]):
                rows.append({"değişken": f, "aralık": label, "puan": int(p)})
        return pd.DataFrame(rows)

    def max_points(self) -> int:
        return int(sum(p.max() for p in self.points_.values()))


def _fmt_money(v: float) -> str:
    return f"{v:,.0f}".replace(",", ".")


EDU_NAMES = {1: "lisansüstü", 2: "üniversite", 3: "lise", 4: "diğer"}


def interval_labels(f: str, cuts: list[float]) -> list[str]:
    if not cuts:
        return ["tümü"]
    if f == "EDUCATION":
        codes = [math.floor(c) for c in cuts] + [math.floor(cuts[-1]) + 1]
        return [EDU_NAMES.get(c, str(c)) for c in codes]
    if f in PAY_COLS or f == "AGE" or f in CATEGORICAL:
        ints = [math.floor(c) for c in cuts]          # x ≤ ints[0], ints[0]+1 … ints[1], …
        labels, lo = [], None
        for k, hi in enumerate(ints + [None]):
            if lo is None and hi is not None:
                labels.append(f"≤ {hi}")
            elif hi is None:
                labels.append(f"≥ {lo}")
            else:
                labels.append(f"{lo}" if lo == hi else f"{lo} … {hi}")
            lo = None if hi is None else hi + 1
        return labels
    fmt = _fmt_money if f.startswith(MONEY_PREFIX) else (lambda v: f"{v:g}")
    labels = [f"≤ {fmt(cuts[0])}"] if cuts else ["tümü"]
    for a, b in zip(cuts[:-1], cuts[1:]):
        labels.append(f"{fmt(a)} – {fmt(b)}")
    if cuts:
        labels.append(f"> {fmt(cuts[-1])}")
    return labels
