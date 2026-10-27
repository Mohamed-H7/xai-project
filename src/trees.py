"""
Ağaç tabanlı yorumlanabilir modeller: GOSDT (optimal seyrek ağaç) ve budanmış CART.

GOSDT ikili öznitelik ister → sürekli değişkenler ayrıklaştırılır. İki eşik stratejisi:
  "tg"       : Threshold Guessing (McTavish ve ark., 2022). Küçük bir gradyan artırma
               modelinin (derinlik 1) kullandığı eşikler aday olarak alınır.
  "quantile" : her sürekli değişken için eşit frekanslı (çeyreklik vb.) eşikler;
               PAY_k için alan anlamlı kod eşikleri (-1, 0, 1, 2 gecikme).
GOSDT sınıf etiketi üretir; AUC hesaplamak için her yaprağın eğitim kümesindeki
temerrüt oranı yaprak olasılığı olarak kullanılır.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL, PAY_COLS

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------- ayrıklaştırma
class QuantileBinarizer(BaseEstimator):
    """Sürekli değişkenlere eşit frekanslı eşikler; sütun adları 'X <= t' biçiminde."""

    def __init__(self, n_bins: int = 4):
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame, y=None):
        self.thresholds_ = {}
        for c in X.columns:
            if c in PAY_COLS:
                cuts = [-1.5, -0.5, 0.5, 1.5]                       # -2 | -1 | 0 | 1 | 2+
            elif c in CATEGORICAL:
                cuts = [1.5, 2.5, 3.5]
            else:
                qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
                cuts = sorted(set(np.quantile(X[c], qs).round(0)))
            self.thresholds_[c] = cuts
        # eğitimde sabit kalan sütunlar fit sırasında bir kez belirlenir (test'te tutarlı kalsın)
        full = self._all(X)
        self.columns_ = list(full.columns[full.nunique() > 1])
        return self

    def _all(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = {f"{c} <= {t}": (X[c] <= t).astype(int)
                for c, ts in self.thresholds_.items() for t in ts}
        return pd.DataFrame(cols, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._all(X)[self.columns_]


class TGBinarizer(BaseEstimator):
    """gosdt.ThresholdGuessBinarizer sarmalayıcısı (sütun adlarını korur)."""

    def __init__(self, n_estimators: int = 40, max_depth: int = 1, random_state: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        from gosdt import ThresholdGuessBinarizer
        self.tg_ = ThresholdGuessBinarizer(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                           random_state=self.random_state)
        self.tg_.set_output(transform="pandas")
        self.tg_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.tg_.transform(X)


def make_binarizer(strategy: str, **kw):
    if strategy == "tg":
        return TGBinarizer(**kw)
    if strategy == "quantile":
        return QuantileBinarizer(**kw)
    raise ValueError(strategy)


# ---------------------------------------------------------------- GOSDT
class GOSDTModel(BaseEstimator, ClassifierMixin):
    """
    ayrıklaştırma → GBDT sıcak başlangıç etiketleri → GOSDT → yaprak olasılıkları.
    depth_budget: kök tek başına derinlik 1 sayılır (depth_budget=4 → en çok 3 bölünme düzeyi).
    """

    def __init__(self, regularization: float = 0.001, depth_budget: int = 4, balance: bool = False,
                 strategy: str = "tg", n_thresholds_est: int = 40, n_bins: int = 4,
                 time_limit: int = 120, random_state: int = 0):
        self.regularization = regularization
        self.depth_budget = depth_budget
        self.balance = balance
        self.strategy = strategy
        self.n_thresholds_est = n_thresholds_est
        self.n_bins = n_bins
        self.time_limit = time_limit
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        from gosdt import GOSDTClassifier
        y = np.asarray(y)
        kw = ({"n_estimators": self.n_thresholds_est, "random_state": self.random_state}
              if self.strategy == "tg" else {"n_bins": self.n_bins})
        self.binarizer_ = make_binarizer(self.strategy, **kw).fit(X, y)
        Xb = self.binarizer_.transform(X)
        self.bin_columns_ = list(Xb.columns)
        # alt sınır tahmini için referans model (GOSDT belgelerindeki öneri)
        ref = GradientBoostingClassifier(n_estimators=40, max_depth=1,
                                         random_state=self.random_state).fit(Xb, y)
        self.gosdt_ = GOSDTClassifier(regularization=self.regularization,
                                      depth_budget=self.depth_budget, balance=self.balance,
                                      time_limit=self.time_limit, similar_support=False,
                                      allow_small_reg=True, verbose=False)
        self.gosdt_.fit(Xb, y, y_ref=ref.predict(Xb))
        res = self.gosdt_.get_result()
        self.tree_ = json.loads(res["models_string"])[0]
        self.status_ = str(res.get("status")).replace("Status.", "")   # CONVERGED / NON_CONVERGENCE / TIMEOUT
        self.fit_time_ = float(res.get("time", np.nan))
        # yaprak olasılıkları (eğitim kümesindeki temerrüt oranı, hafif Laplace düzeltmesi)
        leaves = self._leaf_ids(Xb)
        self.leaf_prob_ = {}
        for leaf in np.unique(leaves):
            m = leaves == leaf
            self.leaf_prob_[leaf] = (y[m].sum() + 1) / (m.sum() + 2)
        self.classes_ = np.array([0, 1])
        return self

    # ağacı gez: her satır için yaprak kimliği (kökten yaprağa giden yol dizgisi)
    def _leaf_ids(self, Xb: pd.DataFrame) -> np.ndarray:
        arr = Xb[self.bin_columns_].to_numpy() if list(Xb.columns) != self.bin_columns_ else Xb.to_numpy()
        out = np.empty(len(arr), dtype=object)

        def walk(node, idx, path):
            if "prediction" in node:
                out[idx] = path or "root"
                return
            f = node["feature"]
            mask = arr[idx, f] == 1
            walk(node["true"], idx[mask], path + "T")
            walk(node["false"], idx[~mask], path + "F")

        walk(self.tree_, np.arange(len(arr)), "")
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xb = self.binarizer_.transform(X).reindex(columns=self.bin_columns_, fill_value=0)
        leaves = self._leaf_ids(Xb)
        base = np.mean(list(self.leaf_prob_.values()))
        p = np.array([self.leaf_prob_.get(l, base) for l in leaves])
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # --- yapı özetleri (kararlılık analizi için)
    def _name(self, node):
        return self.bin_columns_[node["feature"]] if "feature" in node else None

    def root_split(self) -> str | None:
        return self._name(self.tree_)

    def first_level_splits(self) -> list[str]:
        if "feature" not in self.tree_:
            return []
        return [n for n in (self._name(self.tree_["true"]), self._name(self.tree_["false"])) if n]

    def n_leaves(self) -> int:
        def count(node):
            return 1 if "prediction" in node else count(node["true"]) + count(node["false"])
        return count(self.tree_)

    def to_text(self) -> str:
        lines = []

        def rec(node, depth, path):
            pad = "    " * depth
            if "prediction" in node:
                # yaprak olasılığı fit sırasında hesaplanan yol kimliğinden okunur
                p = self.leaf_prob_.get(path or "root", np.nan)
                lines.append(f"{pad}→ temerrüt olasılığı {p:.2f}")
                return
            name = self._name(node)
            lines.append(f"{pad}EĞER {name}:")
            rec(node["true"], depth + 1, path + "T")
            lines.append(f"{pad}DEĞİLSE ({name.replace('<=', '>')}):")
            rec(node["false"], depth + 1, path + "F")

        rec(self.tree_, 0, "")
        return "\n".join(lines)


def var_of(split_name: str | None) -> str | None:
    """'PAY_1 <= 1.5' → 'PAY_1'"""
    return None if split_name is None else split_name.split(" <= ")[0].strip()


# ---------------------------------------------------------------- budanmış CART
def build_cart(params: dict, seed: int = 0) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=seed, **params)


def cart_splits(tree: DecisionTreeClassifier, feature_names) -> tuple[str | None, list[str]]:
    t = tree.tree_
    if t.feature[0] < 0:
        return None, []
    fmt = lambda i: f"{feature_names[t.feature[i]]} <= {t.threshold[i]:.1f}"
    root = fmt(0)
    first = [fmt(c) for c in (t.children_left[0], t.children_right[0]) if t.feature[c] >= 0]
    return root, first
