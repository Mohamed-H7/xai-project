"""
Kesin müdahaleci (interventional / marjinal) Shapley değerleri — kütüphane kullanılmadan.

Oyun:  oyuncular = seçilen d öznitelik (d ≤ 12),  f = modelin log-odds çıktısı
       v(S) = (1/|B|) Σ_{b∈B} f(x_S, b_{S̄})          (arka plan B ile müdahaleci değer fonksiyonu)
       φ_i  = Σ_{S ⊆ N∖{i}} |S|!(d−|S|−1)!/d! · [v(S∪{i}) − v(S)]
Tüm 2^d alt küme için v(S) bir kez hesaplanır (alt küme = bit maskesi; bit i ↔ i. öznitelik),
sonra φ ağırlıklı farklardan bulunur. Maliyet: 2^d · |B| model değerlendirmesi.
"""
from __future__ import annotations

from math import factorial

import numpy as np
import pandas as pd


class ReducedModel:
    """
    f̃(x_seç) = f(x_seç, x_diğer = sabit). Seçilmeyen öznitelikler 'arka plan ortalamasında'
    sabittir; böylece oyun d oyunculudur ve sabit öznitelikler hiçbir koalisyonu etkilemez.
    start/num: ağaç aralığı (toplamsallık testinde modeli iki parçaya bölmek için).
    """

    def __init__(self, booster, features, selected, fixed: pd.Series, start=0, num=None, scale=1.0):
        self.booster, self.features, self.selected = booster, list(features), list(selected)
        self.idx = [self.features.index(f) for f in self.selected]
        self.base = fixed[self.features].to_numpy(dtype=float)
        self.start, self.num, self.scale = start, num, scale

    def full_rows(self, A: np.ndarray) -> np.ndarray:
        full = np.repeat(self.base[None, :], len(A), axis=0)
        full[:, self.idx] = A
        return full

    def __call__(self, A: np.ndarray) -> np.ndarray:
        kw = {"start_iteration": self.start}
        if self.num is not None:
            kw["num_iteration"] = self.num
        return self.scale * self.booster.predict(self.full_rows(np.asarray(A, float)), raw_score=True, **kw)


def subset_bits(d: int) -> np.ndarray:
    m = np.arange(2 ** d)
    return ((m[:, None] >> np.arange(d)) & 1).astype(bool)          # (2^d, d)


def coalition_values(predict, x: np.ndarray, B: np.ndarray, chunk_rows: int = 250_000) -> np.ndarray:
    """v(S), S = 0 … 2^d − 1. Her koalisyon için hibrit satırlar: S'deki sütunlar x'ten, diğerleri b'den."""
    d, nb = len(x), len(B)
    bits = subset_bits(d)
    v = np.empty(2 ** d)
    per = max(1, chunk_rows // nb)
    for s in range(0, 2 ** d, per):
        m = bits[s:s + per]
        rows = np.where(m[:, None, :], x[None, None, :], B[None, :, :])     # (c, nb, d)
        v[s:s + per] = predict(rows.reshape(-1, d)).reshape(len(m), nb).mean(axis=1)
    return v


def shapley_from_values(v: np.ndarray, d: int) -> np.ndarray:
    masks = np.arange(2 ** d)
    size = np.array([bin(m).count("1") for m in masks])
    w = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])
    phi = np.zeros(d)
    for i in range(d):
        without = masks[((masks >> i) & 1) == 0]
        phi[i] = np.sum(w[size[without]] * (v[without | (1 << i)] - v[without]))
    return phi


def exact_shapley(predict, x, B, chunk_rows: int = 250_000):
    """(φ, taban = v(∅), f(x) = v(N))"""
    x, B = np.asarray(x, float), np.asarray(B, float)
    v = coalition_values(predict, x, B, chunk_rows)
    return shapley_from_values(v, len(x)), float(v[0]), float(v[-1])


# ---------------------------------------------------------------- aksiyom testleri
def axiom_efficiency(predict, x, B):
    phi, base, fx = exact_shapley(predict, x, B)
    direct = float(predict(np.asarray(x, float)[None, :])[0])
    return {"aksiyom": "verimlilik", "test": "Σφ = f(x) − E_B[f]",
            "ihlal": abs(phi.sum() - (direct - base))}


def axiom_dummy(predict, x, B, j: int, name: str):
    """j. sütunu sabitleyen sarmalayıcı: f_d hiçbir koalisyonda j'ye bakmaz → φ_j = 0 olmalı."""
    c = float(np.mean(np.asarray(B)[:, j]))

    def f_d(A):
        A = np.array(A, float, copy=True)
        A[:, j] = c
        return predict(A)

    phi, _, _ = exact_shapley(f_d, x, B)
    return {"aksiyom": "kukla", "test": f"f, {name}'i yok sayacak biçimde sarıldı → φ_{name} = 0",
            "ihlal": abs(phi[j])}


def axiom_symmetry(predict, x, B, i: int, k: int, names):
    """
    f_sym(A) = ½[f(A) + f(A, i↔k yer değiştirmiş)] i ve k'ye göre simetriktir. x_k := x_i ve
    b_k := b_i yapılınca her S için v(S∪{i}) = v(S∪{k}) → φ_i = φ_k olmalı.
    """
    x2 = np.array(x, float, copy=True); x2[k] = x2[i]
    B2 = np.array(B, float, copy=True); B2[:, k] = B2[:, i]

    def f_sym(A):
        A = np.asarray(A, float)
        Asw = A.copy(); Asw[:, [i, k]] = A[:, [k, i]]
        return 0.5 * (predict(A) + predict(Asw))

    phi, _, _ = exact_shapley(f_sym, x2, B2)
    return {"aksiyom": "simetri", "test": f"simetrik sarmalayıcı ({names[i]}↔{names[k]}) → φ_i = φ_k",
            "ihlal": abs(phi[i] - phi[k])}


def axiom_additivity(pred1, pred2, x, B, label: str):
    """f = f₁ + f₂ (ağaçların ilk ve ikinci yarısı) → φ(f) = φ(f₁) + φ(f₂)."""
    p1, _, _ = exact_shapley(pred1, x, B)
    p2, _, _ = exact_shapley(pred2, x, B)
    p12, _, _ = exact_shapley(lambda A: pred1(A) + pred2(A), x, B)
    return {"aksiyom": "toplamsallık", "test": label, "ihlal": float(np.abs(p12 - (p1 + p2)).max())}


def linear_closed_form(x, B, seed=0):
    """Ek doğrulama: doğrusal f(z)=βᵀz için müdahaleci Shapley kapalı formu φ_i = β_i (x_i − ort_B z_i)."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=len(x))
    phi, _, _ = exact_shapley(lambda A: np.asarray(A) @ beta, x, B)
    return {"aksiyom": "(ek) doğrusal kapalı form", "test": "φ_i = β_i (x_i − ort_B x_i)",
            "ihlal": float(np.abs(phi - beta * (np.asarray(x) - np.asarray(B).mean(0))).max())}
