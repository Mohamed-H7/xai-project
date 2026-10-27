"""
Küresel açıklama yöntemleri (Ödev 4). Hepsi kütüphanesiz, açık uygulamalardır ki
her yöntemin varsayımı kodda görülebilsin.

  - permütasyon önemi (tekil ve gruplanmış, aynı permütasyon şeması)
  - korelasyon blokları: Spearman |ρ| → hiyerarşik kümeleme
  - PDP + merkezlenmiş ICE (c-ICE) ve paralellik ölçütü
  - ALE (Apley & Zhu, 2020), birinci derece
  - küresel vekil karar ağacı
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from .data import PAY_COLS


# ---------------------------------------------------------------- tahmin fonksiyonları
def make_predictors(model):
    """olasılık ve log-odds (ham skor). ICE paralelliği log-odds ölçeğinde ölçülür:
    olasılık ölçeğinde sigmoid eğriliği, etkileşim olmasa bile eğrileri paralellikten saptırır."""
    proba = lambda X: model.predict_proba(X)[:, 1]
    logit = lambda X: model.predict(X, raw_score=True)
    return proba, logit


# ---------------------------------------------------------------- permütasyon önemi
def permutation_importance(proba, X, y, groups: dict, n_repeats: int, seed: int) -> pd.DataFrame:
    """
    Önem = AUC(özgün) − AUC(permütasyonlu). Bir gruptaki sütunlar AYNI satır permütasyonuyla
    karıştırılır → grup içi korelasyon korunur, grubun geri kalanla ilişkisi kırılır.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, proba(X))
    rows = []
    n = len(X)
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(n)]
            drops.append(base - roc_auc_score(y, proba(Xp)))
        rows.append({"grup": name, "sütunlar": ",".join(cols), "önem": np.mean(drops),
                     "sd": np.std(drops, ddof=1), "p05": np.quantile(drops, 0.05),
                     "p95": np.quantile(drops, 0.95)})
    out = pd.DataFrame(rows).sort_values("önem", ascending=False).reset_index(drop=True)
    out["sıra"] = np.arange(1, len(out) + 1)
    out.attrs["base_auc"] = base
    return out


# ---------------------------------------------------------------- korelasyon blokları
def correlation_blocks(X: pd.DataFrame, threshold: float, method: str = "average"):
    """
    Uzaklık = 1 − |Spearman ρ| (para sütunları çarpık → sıra korelasyonu).
    Ortalama bağlantı: bloğun üyeleri ortalamada |ρ| ≥ 1 − threshold ile bağlı.
    """
    rho = spearmanr(X).statistic
    rho = pd.DataFrame(rho, index=X.columns, columns=X.columns)
    dist = 1 - rho.abs().to_numpy()
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method=method)
    labels = fcluster(Z, t=threshold, criterion="distance")
    blocks = {}
    for lab in np.unique(labels):
        cols = [c for c, l in zip(X.columns, labels) if l == lab]
        blocks[block_name(cols)] = cols
    return rho, Z, blocks


def block_name(cols: list[str]) -> str:
    """['BILL_AMT1', …, 'BILL_AMT6', 'LIMIT_BAL'] → 'BILL_AMT[1-6]+LIMIT_BAL'"""
    if len(cols) == 1:
        return cols[0]
    groups: dict[str, list[int]] = {}
    for c in cols:
        stem = c.rstrip("0123456789")
        num = c[len(stem):]
        groups.setdefault(stem.rstrip("_") if num else c, []).append(int(num) if num else -1)
    parts = []
    for stem, nums in groups.items():
        nums = sorted(n for n in nums if n >= 0)
        if not nums:
            parts.append(stem)
        elif nums == list(range(nums[0], nums[-1] + 1)) and len(nums) > 2:
            parts.append(f"{stem}[{nums[0]}-{nums[-1]}]")
        else:
            parts.append(f"{stem}[{','.join(map(str, nums))}]")
    return "+".join(parts)


def threshold_sensitivity(Z, thresholds) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        lab = fcluster(Z, t=t, criterion="distance")
        sizes = np.bincount(lab)[1:]
        rows.append({"eşik": t, "min_ort_|ρ|": 1 - t, "blok_sayısı": len(sizes),
                     "çok_üyeli_blok": int((sizes > 1).sum()), "en_büyük_blok": int(sizes.max())})
    return pd.DataFrame(rows)


def largest_gap_threshold(Z, lo=0.2, hi=0.5) -> float:
    """Dendrogramdaki birleşme yüksekliklerinde [lo, hi] aralığındaki en büyük boşluğun ortası."""
    h = np.sort(Z[:, 2])
    h = h[(h >= lo - 0.1) & (h <= hi + 0.1)]
    if len(h) < 2:
        return 0.3
    gaps = np.diff(h)
    mids = (h[:-1] + h[1:]) / 2
    ok = (mids >= lo) & (mids <= hi)
    if not ok.any():
        return 0.3
    k = np.argmax(np.where(ok, gaps, -1))
    return float(round(mids[k], 3))


def multiple_r2(X: pd.DataFrame) -> pd.Series:
    """Her özniteliğin diğerleri tarafından (sıra dönüşümlü) doğrusal açıklanma oranı R²_j."""
    R = X.rank(pct=True)
    out = {}
    for c in X.columns:
        others = R.drop(columns=c)
        out[c] = LinearRegression().fit(others, R[c]).score(others, R[c])
    return pd.Series(out, name="R2_diğerleri")


# ---------------------------------------------------------------- ızgaralar
def feature_grid(X_ref: pd.DataFrame, f: str, n: int = 20, min_count: int = 30) -> np.ndarray:
    v = X_ref[f]
    if f in PAY_COLS or f == "EDUCATION":
        cnt = v.value_counts()
        return np.array(sorted(cnt[cnt >= min_count].index), dtype=float)
    return np.unique(v.quantile(np.linspace(0.05, 0.95, n)).to_numpy(dtype=float))


# ---------------------------------------------------------------- PDP / ICE
def ice_matrix(predict, X: pd.DataFrame, f: str, grid: np.ndarray) -> np.ndarray:
    """(n_örnek, n_ızgara) ICE matrisi; tek bir toplu tahminle."""
    n, g = len(X), len(grid)
    big = pd.concat([X] * g, ignore_index=True)
    big[f] = np.repeat(grid, n)
    return predict(big).reshape(g, n).T


def ice_parallelism(ice: np.ndarray, grid: np.ndarray) -> dict:
    """
    c-ICE: her eğri ızgaranın ilk noktasına sabitlenir.
    Δ_i = c-ICE_i(son) → uçtan uca eğim s_i = Δ_i / (x_son − x_ilk).
    Uç eğimler: ilk ve son ızgara aralığındaki yerel eğimler.
    I² = Var(Δ) / E[Δ²] ∈ [0, 1]: 0 = tam paralel (toplamsal), 1 = ortalama etki yok, saf heterojenlik.
    """
    c = ice - ice[:, [0]]
    span = grid[-1] - grid[0]
    s = c[:, -1] / span
    left = (ice[:, 1] - ice[:, 0]) / (grid[1] - grid[0])
    right = (ice[:, -1] - ice[:, -2]) / (grid[-1] - grid[-2])
    d = c[:, -1]
    return {
        "Δ_ort": float(d.mean()), "Δ_sd": float(d.std(ddof=1)),
        "eğim_ort": float(s.mean()), "eğim_sd": float(s.std(ddof=1)),
        "eğim_DK": float(s.std(ddof=1) / max(abs(s.mean()), 1e-12)),
        "sol_uç_eğim_sd": float(left.std(ddof=1)), "sağ_uç_eğim_sd": float(right.std(ddof=1)),
        "I2": float(d.var() / max(np.mean(d ** 2), 1e-12)),
        "ters_işaret_oranı": float((np.sign(d) != np.sign(d.mean())).mean()),
    }


def subgroup_search(delta: np.ndarray, X: pd.DataFrame, f: str, seed: int = 0) -> dict:
    """ICE uç farkı Δ_i'yi diğer özniteliklerle açıklayan sığ ağaç; kökü alt grup değişkenidir."""
    Xo = X.drop(columns=f)
    tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=max(20, len(X) // 20), random_state=seed)
    r2 = cross_val_score(tree, Xo, delta, cv=5, scoring="r2").mean()
    tree.fit(Xo, delta)
    t = tree.tree_
    root_var = Xo.columns[t.feature[0]] if t.feature[0] >= 0 else None
    rho = {c: spearmanr(Xo[c], delta).statistic for c in Xo.columns}
    top_rho = max(rho, key=lambda c: abs(np.nan_to_num(rho[c])))
    return {"kök_değişken": root_var, "kök_eşik": float(t.threshold[0]) if root_var else None,
            "ağaç_cv_R2": float(r2), "en_yüksek_|ρ|_değişken": top_rho, "ρ": float(rho[top_rho])}


# ---------------------------------------------------------------- ALE
def ale_1d(predict, X: pd.DataFrame, f: str, n_bins: int = 20):
    """
    Birinci derece ALE. Kenarlar: nicelikler (ayrık PAY kodlarında benzersiz değerler).
    Her gözlem kendi aralığının iki kenarına taşınır → yalnızca veride gerçekleşen
    bölgelerde yerel fark alınır (PDP'nin aksine gerçekleşmeyen kombinasyon üretmez).
    """
    x = X[f].to_numpy(dtype=float)
    if f in PAY_COLS or f == "EDUCATION" or len(np.unique(x)) <= n_bins:
        edges = np.unique(x)
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    K = len(edges) - 1
    k = np.clip(np.searchsorted(edges, x, side="left"), 1, K)       # aralık (z_{k-1}, z_k]
    lo, hi = X.copy(), X.copy()
    lo[f], hi[f] = edges[k - 1], edges[k]
    d = predict(hi) - predict(lo)
    counts = np.bincount(k, minlength=K + 1)[1:]
    sums = np.bincount(k, weights=d, minlength=K + 1)[1:]
    local = np.divide(sums, counts, out=np.zeros(K), where=counts > 0)
    ale = np.concatenate([[0.0], np.cumsum(local)])
    # merkezleme: veri dağılımı üzerinde ortalama sıfır
    ale -= np.interp(x, edges, ale).mean()
    return edges, ale


def pdp_on(predict, X: pd.DataFrame, f: str, grid: np.ndarray) -> np.ndarray:
    pdp = ice_matrix(predict, X, f, grid).mean(axis=0)
    return pdp - np.interp(X[f].to_numpy(dtype=float), grid, pdp).mean()


def pdp_ale_distance(pdp, ale, grid, x) -> dict:
    """Veri ağırlıklı fark: sqrt(ort_i (PDP(x_i) − ALE(x_i))²) ve etki büyüklüğüne oranı."""
    p, a = np.interp(x, grid, pdp), np.interp(x, grid, ale)
    rms = float(np.sqrt(np.mean((p - a) ** 2)))
    scale = float((p.std() + a.std()) / 2)
    return {"rms_fark": rms, "etki_sd": scale, "göreli_fark": rms / max(scale, 1e-12)}


# ---------------------------------------------------------------- vekil ağaç
def surrogate_curve(target_train, X_train, target_test, X_test, depths, seed=0, min_leaf=50):
    """Sadakat = vekilin eğitilmediği kümede R²(kara kutu, vekil) ve %20 ret kararında uyum."""
    rows = []
    for d in depths:
        s = DecisionTreeRegressor(max_depth=d, min_samples_leaf=min_leaf, random_state=seed)
        s.fit(X_train, target_train)
        p_tr, p_te = s.predict(X_train), s.predict(X_test)
        rows.append({"derinlik": d, "yaprak": s.get_n_leaves(),
                     "R2_eğitim": r2(target_train, p_tr), "R2_test": r2(target_test, p_te),
                     "karar_uyumu_test": decision_agreement(target_test, p_te, 0.2)})
    return pd.DataFrame(rows)


def r2(a, b) -> float:
    return float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))


def decision_agreement(f, g, reject_rate) -> float:
    """En riskli %reject_rate reddedilirse, kara kutu ve vekil aynı kişileri mi reddediyor?"""
    k = int(round(reject_rate * len(f)))
    rf = np.zeros(len(f), bool); rf[np.argsort(-f)[:k]] = True
    rg = np.zeros(len(g), bool); rg[np.argsort(-g)[:k]] = True
    return float((rf == rg).mean())


def deviation_structure(resid: np.ndarray, X: pd.DataFrame, seed: int = 0, n_null: int = 200) -> dict:
    """
    Yerel sapmalar |r_i| rastgele mi, yapılı mı? Sığ ağaç |r|'yi özniteliklerden ne kadar
    öngörebiliyor (5-katlı ÇD R²)? Karşılaştırma: |r| karıştırılarak elde edilen sıfır dağılımı.
    """
    rng = np.random.default_rng(seed)
    a = np.abs(resid)
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(30, len(X) // 50), random_state=seed)
    obs = cross_val_score(tree, X, a, cv=5, scoring="r2").mean()
    null = np.array([cross_val_score(tree, X, rng.permutation(a), cv=5, scoring="r2").mean()
                     for _ in range(n_null)])
    tree.fit(X, a)
    leaf = tree.apply(X)
    stats = pd.DataFrame({"leaf": leaf, "a": a}).groupby("leaf")["a"].agg(["mean", "size"])
    worst = int(stats["mean"].idxmax())
    return {"cv_R2": float(obs), "null_ort": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
            "p_değeri": float((np.sum(null >= obs) + 1) / (n_null + 1)),
            "en_kötü_yaprak_kuralı": leaf_rule(tree, X.columns, worst),
            "en_kötü_yaprak_n": int(stats.loc[worst, "size"]),
            "en_kötü_yaprak_ort_|r|": float(stats.loc[worst, "mean"]),
            "genel_ort_|r|": float(a.mean()), "worst_mask": leaf == worst}


def leaf_rule(tree, names, leaf_id) -> str:
    t = tree.tree_
    parent = {}
    for i in range(t.node_count):
        for side, child in (("≤", t.children_left[i]), (">", t.children_right[i])):
            if child >= 0:
                parent[child] = (i, side)
    conds, node = [], leaf_id
    while node in parent:
        p, side = parent[node]
        conds.append(f"{names[t.feature[p]]} {side} {t.threshold[p]:.1f}")
        node = p
    return " VE ".join(reversed(conds))
