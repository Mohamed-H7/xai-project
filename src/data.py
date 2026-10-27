"""
Veri yükleme, temizleme ve sabit bölme.
Ödev 2-5 boyunca TÜM ödevler bu modülü kullanır; böylece temizleme kararları ve
holdout kümesi tüm dönem boyunca aynı kalır.

Veri: UCI "Default of Credit Card Clients" (Yeh & Lien, 2009), id=350
      https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Dosyayı  data/default of credit card clients.xls  konumuna koyun.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
XLS_PATH = DATA_DIR / "default of credit card clients.xls"
SPLIT_PATH = DATA_DIR / "split_indices.json"

TARGET = "default"
SEED = 42

# Korunan öznitelikler: modele GİRMEZ, ama tabloda kalır
# (Ödev 5'teki referans grupları ve olası adalet analizleri için).
PROTECTED = ["SEX", "MARRIAGE"]

PAY_COLS = [f"PAY_{k}" for k in range(1, 7)]        # 1 = Eylül 2005 ... 6 = Nisan 2005
BILL_COLS = [f"BILL_AMT{k}" for k in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{k}" for k in range(1, 7)]

# Modele giren 21 öznitelik
FEATURES = ["LIMIT_BAL", "EDUCATION", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS
CATEGORICAL = ["EDUCATION"]


def load_raw(path: Path | str = XLS_PATH) -> pd.DataFrame:
    """Ham .xls dosyasını okur; ilk satır X1..Y, ikinci satır gerçek adlardır."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, header=1)
    df = df.drop(columns=["ID"], errors="ignore")
    # Orijinal dosyada PAY_0 → PAY_2 atlaması var; PAY_1..PAY_6 olarak düzeltildi
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})
    return df


def undocumented_code_report(df: pd.DataFrame) -> pd.DataFrame:
    """Veri sözlüğünde tanımlı olmayan kodların sayımı (rapora girer)."""
    rows = [
        ("EDUCATION", "0, 5, 6", int(df["EDUCATION"].isin([0, 5, 6]).sum())),
        ("MARRIAGE", "0", int((df["MARRIAGE"] == 0).sum())),
        ("PAY_1", "-2, 0", int(df["PAY_1"].isin([-2, 0]).sum())),
    ]
    return pd.DataFrame(rows, columns=["öznitelik", "tanımsız kod", "satır sayısı"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temizleme kararları (raporda belgelenir):
      - EDUCATION 0/5/6 (sözlükte yok)  → 4 ("diğer")
      - MARRIAGE 0 (sözlükte yok)       → 3 ("diğer")
      - PAY_k kodları (-2, -1, 0, 1..8) OLDUĞU GİBİ bırakılır: -2 ve 0 sözlükte
        tanımsızdır ama yaygın yoruma göre -2 = harcama yok, 0 = döner kredi
        kullanımı. Bu kodlar Ödev 3'teki "veri toplama artefaktı" incelemesine adaydır.
      - Para birimi sütunları dönüştürülmez (ağaçlar ve EBM için gerekmez;
        doğrusal model kendi dönüşümünü pipeline içinde yapar).
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def load_clean(path: Path | str = XLS_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def get_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    %20 holdout bir kez ayrılır ve indeksleri diske yazılır.
    Ödev 2 holdout'a HİÇ dokunmaz; Ödev 4 (test kümesinde permütasyon önemi,
    vekil modelin sadakati) için saklanır.
    """
    if SPLIT_PATH.exists():
        saved = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        if saved["n_rows"] == len(df):
            return np.array(saved["dev_idx"]), np.array(saved["holdout_idx"])
    idx = np.arange(len(df))
    dev_idx, hold_idx = train_test_split(
        idx, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    dev_idx, hold_idx = np.sort(dev_idx), np.sort(hold_idx)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps({
        "n_rows": len(df), "seed": seed, "test_size": test_size,
        "dev_idx": dev_idx.tolist(), "holdout_idx": hold_idx.tolist(),
    }), encoding="utf-8")
    return dev_idx, hold_idx


def get_dev_holdout(path: Path | str = XLS_PATH):
    """(X_dev, y_dev, X_hold, y_hold, df_tam) döndürür."""
    df = load_clean(path)
    dev_idx, hold_idx = get_split(df)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    return (X.iloc[dev_idx].reset_index(drop=True), y[dev_idx],
            X.iloc[hold_idx].reset_index(drop=True), y[hold_idx], df)
