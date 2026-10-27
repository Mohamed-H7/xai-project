"""
Model kurucuları ve hiperparametre arama uzayları.
  - Kara kutu : LightGBM (gradyan artırma)
  - Yorumlanabilir 1 : EBM (interpret)
  - Yorumlanabilir 2 : L1-cezalı lojistik regresyon (LASSO)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import BILL_COLS, CATEGORICAL, FEATURES, PAY_AMT_COLS, PAY_COLS, SEED

N_JOBS = -1


# ---------------------------------------------------------------- kara kutu
def lgbm_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def build_lgbm(params: dict, seed: int = SEED):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**params, subsample_freq=1, random_state=seed,
                          n_jobs=N_JOBS, verbose=-1)


# ---------------------------------------------------------------- EBM
EBM_OUTER_BAGS = 8   # tüm EBM'lerde sabit (hesaplama bütçesi; raporda belirtilir)


def ebm_space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_bins": trial.suggest_categorical("max_bins", [32, 64, 128, 256]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50, log=True),
        "max_leaves": trial.suggest_int("max_leaves", 2, 3),
        "interactions": trial.suggest_categorical("interactions", [0, 5, 10]),
        "smoothing_rounds": trial.suggest_categorical("smoothing_rounds", [0, 75, 200]),
    }


def build_ebm(params: dict, seed: int = SEED):
    from interpret.glassbox import ExplainableBoostingClassifier
    ftypes = ["nominal" if f in CATEGORICAL else "continuous" for f in FEATURES]
    return ExplainableBoostingClassifier(
        feature_names=FEATURES, feature_types=ftypes, outer_bags=EBM_OUTER_BAGS,
        random_state=seed, n_jobs=N_JOBS, **params,
    )


# ---------------------------------------------------------------- LASSO
class LinearFeatures(BaseEstimator, TransformerMixin):
    """
    Doğrusal model için öznitelik temsili. Temsil seçimi de aramanın parçasıdır
    (Rudin: yorumlanabilir modele emek vermek, temsil üzerinde de çalışmak demektir).
      pay_encoding = "raw"   : PAY_k olduğu gibi (-2..8 tek sayı)
                     "split" : gecikme ay sayısı max(PAY_k,0) + "tam ödedi" (-1)
                               ve "harcama yok" (-2) göstergeleri
      log_amounts  = True    : para sütunlarına log1p (BILL için işaretli log)
    """

    def __init__(self, pay_encoding: str = "split", log_amounts: bool = True):
        self.pay_encoding = pay_encoding
        self.log_amounts = log_amounts

    def fit(self, X, y=None):
        self.feature_names_out_ = list(self.transform(X.iloc[:5]).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        lim = X["LIMIT_BAL"].astype(float)
        out["LIMIT_BAL"] = np.log1p(lim) if self.log_amounts else lim
        out["AGE"] = X["AGE"].astype(float)
        for lvl, name in [(1, "lisansüstü"), (2, "üniversite"), (3, "lise")]:
            out[f"EDUCATION={name}"] = (X["EDUCATION"] == lvl).astype(float)  # referans: 4=diğer
        for c in PAY_COLS:
            if self.pay_encoding == "raw":
                out[c] = X[c].astype(float)
            else:
                out[f"{c}_gecikme"] = X[c].clip(lower=0).astype(float)
                out[f"{c}=-1"] = (X[c] == -1).astype(float)
                out[f"{c}=-2"] = (X[c] == -2).astype(float)
        for c in BILL_COLS:
            v = X[c].astype(float)
            out[c] = np.sign(v) * np.log1p(np.abs(v)) if self.log_amounts else v
        for c in PAY_AMT_COLS:
            v = X[c].astype(float)
            out[c] = np.log1p(v) if self.log_amounts else v
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


def _l1_logreg(C: float, seed: int) -> LogisticRegression:
    # scikit-learn 1.8+ 'penalty' parametresini kullanımdan kaldırdı; eski sürümlerle uyumluluk
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear",
                                  max_iter=2000, random_state=seed)
    return LogisticRegression(C=C, penalty="l1", solver="liblinear",
                              max_iter=2000, random_state=seed)


def lasso_space(trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "pay_encoding": trial.suggest_categorical("pay_encoding", ["raw", "split"]),
        "log_amounts": trial.suggest_categorical("log_amounts", [True, False]),
    }


def build_lasso(params: dict, seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("feat", LinearFeatures(params["pay_encoding"], params["log_amounts"])),
        ("scale", StandardScaler()),
        ("clf", _l1_logreg(params["C"], seed)),
    ])


MODELS = {
    "LightGBM": (lgbm_space, build_lgbm),
    "EBM": (ebm_space, build_ebm),
    "LASSO": (lasso_space, build_lasso),
}
