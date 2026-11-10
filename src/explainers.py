"""
Ödev 6 — değerlendirilen açıklama yöntemleri, kontroller ve yardımcılar.

Tüm açıklamalar log-odds (LightGBM ham skor) ölçeğindedir. SHAP-müd, Örtme ve LIME aynı
arka plan B ile "özniteliği kaldırır"; yöntemler kaldırma çerçevesinde (Covert vd., 2021)
yalnızca iki noktada ayrılır: özniteliğin NASIL kaldırıldığı ve HANGİ koalisyonların tartıldığı.

  SHAP-müd : müdahaleci TreeSHAP — tüm koalisyonlar, Shapley ağırlıkları, B'den marjinal kaldırma
  SHAP-yol : yol bağımlı TreeSHAP — tüm koalisyonlar, Shapley ağırlıkları, ağaç örtüsüyle kaldırma
  LIME     : kendi uygulamamız — rastgele koalisyonlar, üstel yakınlık çekirdeği, ağırlıklı ridge
  Örtme    : tek öznitelik kaldırma — φ_j = f(x) − ort_b f(x_{−j}, b_j)  (yalnız tam koalisyon)

Kontroller (modelden bağımsız):
  Rastgele    : φ ~ N(0, 1)
  Ort-uzaklık : φ_j = (x_j − μ_j) / σ_j   (ortalamadan uzaklık; modele hiç bakmaz)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

METHODS = ["SHAP-müd", "SHAP-yol", "LIME", "Örtme"]
CONTROLS = ["Rastgele", "Ort-uzaklık"]


# ---------------------------------------------------------------- tahmin
def raw_predict(booster):
    return lambda X: booster.predict(np.asarray(X, dtype=float), raw_score=True)


# ---------------------------------------------------------------- yöntemler
def shap_interventional(booster, X, B):
    """Müdahaleci TreeSHAP. Arka plan maskeleyiciyle verilir: aksi hâlde shap > 100 satırı
    sessizce 100'e indirir (Ödev 5'te bulundu) ve boyut doğrulanır."""
    import shap
    masker = shap.maskers.Independent(B, max_samples=len(B))
    ex = shap.TreeExplainer(booster, data=masker, feature_perturbation="interventional", model_output="raw")
    if ex.data.shape[0] != len(B):
        raise RuntimeError("shap arka planı değiştirdi")
    phi = np.asarray(ex.shap_values(np.asarray(X, float), check_additivity=False)).reshape(len(X), -1)
    return phi, float(np.ravel(ex.expected_value)[-1])


def shap_path(booster, X):
    import shap
    ex = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent", model_output="raw")
    phi = np.asarray(ex.shap_values(np.asarray(X, float), check_additivity=False)).reshape(len(X), -1)
    return phi, float(np.ravel(ex.expected_value)[-1])


def occlusion(predict, X, B):
    """φ_j(x) = f(x) − (1/|B|) Σ_b f(x, x_j := b_j)."""
    X, B = np.asarray(X, float), np.asarray(B, float)
    n, d = X.shape
    idx = np.arange(d)
    fx = predict(X)
    out = np.empty((n, d))
    for i in range(n):
        rows = np.broadcast_to(X[i], (d, len(B), d)).copy()
        rows[idx, :, idx] = B.T                       # j. blokta yalnızca j. sütun arka plandan
        out[i] = fx[i] - predict(rows.reshape(-1, d)).reshape(d, len(B)).mean(1)
    return out


def lime(predict, X, B, n_samples=5000, kernel="lime", sigma=0.75, seeds=None, ridge=1.0, n_bg_avg=1):
    """
    Tablo verisi için LIME (kendi uygulamamız, kütüphane kullanılmadı).
      örnekleme : kaldırılacak öznitelik sayısı m ~ U{1..d}, kaldırılan altküme rastgele;
                  kaldırılan öznitelikler B'den rastgele seçilen TEK bir satırdan alınır.
                  z_j = 1 ⇔ x_j korunmuş. İlk örnek orijinal x'tir (z = 1).
      çekirdek  : "lime"    → π(z) = exp(−D²/σ²), D² = m/d  (σ = 0.75: LIME'ın varsayılanı
                               √d·0.75 genişliğinin ikili uzaydaki karşılığı)
                  "shapley" → π ∝ 1/(m(d−m)); bu örnekleyicide Shapley çekirdeğine denk
                               (örnekleme olasılığıyla düzeltilmiş); m = 0 ve m = d büyük ağırlık
      uydurma   : log-odds ~ β0 + βᵀz, ağırlıklı ridge (α = 1, kesişim cezasız).  φ = β.
    seeds: örnek başına tohum dizisi (kararlılık testinde aynı tohum = ortak rastgele sayılar).
    n_bg_avg > 1: (yalnızca E2 deneyi) her z için kaldırılan öznitelikler n_bg_avg arka plan
                  satırıyla doldurulup tahminler ortalanır → örnekleme gürültüsü azalır.
    """
    X, B = np.asarray(X, float), np.asarray(B, float)
    n, d = X.shape
    seeds = np.arange(n) if seeds is None else np.asarray(seeds)
    out = np.empty((n, d))
    pen = np.eye(d + 1) * ridge
    pen[0, 0] = 0.0
    for i in range(n):
        rng = np.random.default_rng(int(seeds[i]))
        m = rng.integers(1, d + 1, n_samples)
        ranks = rng.random((n_samples, d)).argsort(1).argsort(1)
        Z = ranks >= m[:, None]                       # True = korunan öznitelik
        Z[0] = True
        m = d - Z.sum(1)
        brow = B[rng.integers(0, len(B), (n_samples, n_bg_avg))]          # (n, r, d)
        rows = np.where(Z[:, None, :], X[i], brow)
        y = predict(rows.reshape(-1, d)).reshape(n_samples, n_bg_avg).mean(1)
        if kernel == "lime":
            w = np.exp(-(m / d) / sigma ** 2)
            w = w / w.mean()
        elif kernel == "shapley":
            # ağırlıklar düzenli koalisyonlarda ortalama 1 olacak biçimde ölçeklenir; boş/tam
            # koalisyonlar (verimlilik kısıtı) çok büyük ağırlık alır. Ölçekleme ridge cezasının
            # düzenli örnekleri ezmemesi için gereklidir.
            reg = (m > 0) & (m < d)
            w = np.empty(n_samples)
            w[reg] = 1.0 / (m[reg] * (d - m[reg]))
            w[reg] /= w[reg].mean()
            w[~reg] = 1e4
        else:
            raise ValueError(kernel)
        A = np.column_stack([np.ones(n_samples), Z.astype(float)])
        Aw = A * w[:, None]
        beta = np.linalg.solve(A.T @ Aw + pen, Aw.T @ y)
        out[i] = beta[1:]
    return out


def zscore(X, mu, sd):
    return (np.asarray(X, float) - mu) / sd


def random_attr(n, d, seed):
    return np.random.default_rng(seed).normal(size=(n, d))


def explain_all(booster, X, B, lime_seeds, lime_samples=5000):
    """Dört yöntem; dönen: {ad: (n, d)}, {ad: taban}."""
    pred = raw_predict(booster)
    phi_int, base_int = shap_interventional(booster, X, B)
    phi_path, base_path = shap_path(booster, X)
    phi = {"SHAP-müd": phi_int, "SHAP-yol": phi_path,
           "LIME": lime(pred, X, B, n_samples=lime_samples, seeds=lime_seeds),
           "Örtme": occlusion(pred, X, B)}
    return phi, {"SHAP-müd": base_int, "SHAP-yol": base_path}


# ---------------------------------------------------------------- karşılaştırma ölçütleri
def _spearman_rows(A, B):
    out = np.full(len(A), np.nan)
    for i, (a, b) in enumerate(zip(A, B)):
        if np.ptp(a) > 0 and np.ptp(b) > 0:
            out[i] = spearmanr(a, b).statistic
    return out


def agreement(A, B, k=3):
    """Satır başına: ilk-1, ilk-k örtüşmesi, |φ| sıra korelasyonu, işaretli sıra korelasyonu,
    ortak ilk-k özniteliklerde işaret uyumu (Krishna vd., 2022 ölçütleri)."""
    A, B = np.asarray(A), np.asarray(B)
    aA, aB = np.abs(A), np.abs(B)
    top1 = (aA.argmax(1) == aB.argmax(1)).astype(float)
    tA, tB = np.argsort(-aA, 1)[:, :k], np.argsort(-aB, 1)[:, :k]
    topk = np.array([len(set(a) & set(b)) / k for a, b in zip(tA, tB)])
    sign = np.full(len(A), np.nan)
    for i, (a, b) in enumerate(zip(tA, tB)):
        common = list(set(a) & set(b))
        if common:
            sign[i] = np.mean(np.sign(A[i, common]) == np.sign(B[i, common]))
    return {"ilk1": top1, f"ilk{k}": topk, "spearman_abs": _spearman_rows(aA, aB),
            "spearman_işaretli": _spearman_rows(A, B), "işaret_uyumu": sign}


def rel_l2(A, B):
    """‖A − B‖ / ‖A‖ satır başına."""
    num = np.linalg.norm(np.asarray(A) - np.asarray(B), axis=1)
    den = np.linalg.norm(np.asarray(A), axis=1)
    return num / np.maximum(den, 1e-12)


# ---------------------------------------------------------------- model rastgeleleştirme
class LeafRandomizer:
    """
    Kademeli parametre rastgeleleştirme (Adebayo vd., 2018) — ağaç topluluğu karşılığı:
    çıkıştan girişe doğru (son ağaçtan başlayarak) giderek artan oranda ağacın yaprak değerleri,
    aynı ortalama ve standart sapmalı normal dağılımdan yeniden çekilir; ağaç yapısı (bölünme
    öznitelikleri, eşikler, örtüler) korunur. Rastgele değerler bir tohum için bir kez çekilir,
    böylece adımlar birbirinin üzerine eklenir (kümülatif).
    """

    def __init__(self, booster, seed):
        self.lines = [l for l in booster.model_to_string().split("\n") if not l.startswith("tree_sizes=")]
        self.leaf_line, cur = {}, None
        for k, line in enumerate(self.lines):
            if line.startswith("Tree="):
                cur = int(line[5:])
            elif line.startswith("leaf_value=") and cur is not None:
                self.leaf_line[cur] = k
        self.T = len(self.leaf_line)
        rng = np.random.default_rng(seed)
        self.new = {}
        for t, k in self.leaf_line.items():
            v = np.array(self.lines[k][11:].split(), float)
            sd = v.std() if len(v) > 1 and v.std() > 0 else max(abs(v[0]) * 0.1, 1e-6)
            self.new[t] = v.mean() + sd * rng.normal(size=len(v))

    def booster(self, frac):
        import lightgbm as lgb
        n_rand = int(round(frac * self.T))
        lines = list(self.lines)
        for t in range(self.T - n_rand, self.T):
            lines[self.leaf_line[t]] = "leaf_value=" + " ".join(repr(float(v)) for v in self.new[t])
        return lgb.Booster(model_str="\n".join(lines))


# ---------------------------------------------------------------- model duyarlılığı
def split_thresholds(booster, d):
    """Her öznitelik için topluluktaki tüm sayısal bölünme eşikleri (sıralı)."""
    thr = [[] for _ in range(d)]

    def walk(node):
        if "split_feature" in node:
            if node.get("decision_type", "<=") == "<=":
                thr[node["split_feature"]].append(float(node["threshold"]))
            walk(node["left_child"])
            walk(node["right_child"])

    for t in booster.dump_model()["tree_info"]:
        walk(t["tree_structure"])
    return [np.unique(np.array(t)) for t in thr]


def threshold_crossings(x, x2, thr):
    """x ile x2 arasında kalan eşik sayısı (öznitelikler toplamı): 0 ise hiçbir ağaçta hiçbir
    hibrit yol değişmez, yani ağaç tabanlı SHAP açıklaması birebir aynı kalmak zorundadır."""
    total = 0
    for j, t in enumerate(thr):
        lo, hi = min(x[j], x2[j]), max(x[j], x2[j])
        if hi > lo and len(t):
            total += int(np.searchsorted(t, hi, "left") - np.searchsorted(t, lo, "left"))
    return total
