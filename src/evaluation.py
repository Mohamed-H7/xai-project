"""
Değerlendirme yardımcıları: metrikler ve düzeltilmiş tekrarlı-ÇD güven aralığı.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def metrics(y_true, p) -> dict:
    return {
        "auc": roc_auc_score(y_true, p),
        "logloss": log_loss(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "caught_at_20": caught_at_reject_rate(y_true, p, 0.20),
    }


def caught_at_reject_rate(y_true, p, rate: float) -> float:
    """
    Başvuruların en riskli %rate'i reddedilirse, gerçek temerrütlerin kaçta kaçı
    yakalanır (recall)? AUC farkını alan diline çevirmek için kullanılır.
    """
    y_true = np.asarray(y_true)
    k = int(round(rate * len(p)))
    top = np.argsort(-np.asarray(p))[:k]
    return y_true[top].sum() / max(y_true.sum(), 1)


def corrected_ci(diffs, n_train: int, n_test: int, alpha: float = 0.05) -> dict:
    """
    Nadeau & Bengio (2003) düzeltilmiş tekrarlı-ÇD t aralığı.
    Katlar eğitim verisini paylaştığı için farklar bağımsız değildir; naif varyans
    gerçek varyansı küçümser. Düzeltme:  var = s^2 * (1/J + n_test/n_train)
    """
    d = np.asarray(diffs, dtype=float)
    J = len(d)
    mean, s2 = d.mean(), d.var(ddof=1)
    se_corr = np.sqrt(s2 * (1 / J + n_test / n_train))
    se_naive = np.sqrt(s2 / J)
    t = stats.t.ppf(1 - alpha / 2, df=J - 1)
    t_stat = mean / se_corr if se_corr > 0 else np.inf
    return {
        "mean": mean, "sd": np.sqrt(s2), "J": J,
        "ci_low": mean - t * se_corr, "ci_high": mean + t * se_corr,
        "naive_ci_low": mean - t * se_naive, "naive_ci_high": mean + t * se_naive,
        "p_value": 2 * stats.t.sf(abs(t_stat), df=J - 1),
        "share_positive": float((d > 0).mean()),
    }
