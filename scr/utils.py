import numpy as np
import pandas as pd

import argparse, json, os, math, joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, Dict, Any, List

from sklearn.model_selection import GroupKFold, StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.metrics import ndcg_score

# Utils
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed)

def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)

def mrr_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    ranks = []
    order = np.argsort(-prob, axis=1)
    for i in range(prob.shape[0]):
        label = y_true[i]
        pos = np.where(order[i] == label)[0]
        if len(pos) == 0:
            ranks.append(0.0)
        else:
            ranks.append(1.0 / (pos[0] + 1))
    return float(np.mean(ranks))

def load_data(X_path: Path, y_path: Path, meta_path: Path, features_csv: Optional[Path]):
    X = np.load(X_path)
    y = np.load(y_path) if y_path.exists() else None

    meta = json.loads(Path(meta_path).read_text())
    labels_vocab = meta.get("label_info", {}).get("labels_vocab", None)

    groups = None
    if features_csv is not None and features_csv.exists():
        df = pd.read_csv(features_csv)
        if "iedb_receptor_id" in df.columns:
            groups = df["iedb_receptor_id"].astype(str).values
        else:
            groups = np.arange(len(df))
    else:
        groups = np.arange(X.shape[0])

    return X, y, groups, labels_vocab, meta

def load_json(p:Path):
    with p.open("r") as f:
        return json.load(f)

def load_npy(p:Path):
    return np.load(p, allow_pickle=False)

def human(n: int) -> str:
    for unit in ["", "K","M","B"]:
        if abs(n) < 1000:
            return f"{n: .0f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"

