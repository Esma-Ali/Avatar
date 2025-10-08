"""
train_jax_rf.py 
deep-learning-prediction/JAX/train_jax_rf.py
JAX Random Forest training (CPU-only, no JIT) + artifact export
Produces: metrics.json, confusion_matrix.png, rf_jax_model.pkl
Author: Esma Ali
"""

import os, json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from jax_rf import RandomForest

# Force CPU & no JIT
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_DISABLE_JIT"] = "1"

ROOT     = Path(__file__).resolve().parents[2]              # repo root (…/Avatar)
CSV_PATH = ROOT / "data" / "brainwaves.csv"
OUT_DIR  = Path(__file__).resolve().parent / "artifacts"

N_ESTIMATORS = 120
MAX_DEPTH    = 10
SEED         = 42

# ---------- utilities (no sklearn) ----------

def load_csv_xy(csv_path: Path, target_col: str = "label"):
    df = pd.read_csv(csv_path)
    y = df[target_col].astype("int32").to_numpy()
    X = df.drop(columns=[target_col]).astype("float32").to_numpy()
    return X, y

def stratified_split(X, y, test_size=0.2, val_size=0.0, seed=0):
    key = jrand.key(seed)
    X = np.asarray(X); y = np.asarray(y, dtype=np.int32)
    classes = np.unique(y)
    idx_tr, idx_val, idx_te = [], [], []
    for c in classes:
        c_idx = np.where(y == c)[0]
        key, sk = jrand.split(key)
        shuffled = np.array(jrand.permutation(sk, c_idx))
        n = shuffled.size
        n_test = int(round(n * test_size))
        n_val  = int(round(n * val_size))
        n_train = n - n_test - n_val
        idx_tr.append(shuffled[:n_train])
        idx_val.append(shuffled[n_train:n_train+n_val])
        idx_te.append(shuffled[n_train+n_val:])
    tr = np.concatenate(idx_tr); va = np.concatenate(idx_val); te = np.concatenate(idx_te)
    key, k1, k2, k3 = jrand.split(key, 4)
    tr = np.array(jrand.permutation(k1, tr))
    va = np.array(jrand.permutation(k2, va))
    te = np.array(jrand.permutation(k3, te))
    return X[tr], y[tr], X[va], y[va], X[te], y[te]

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def f1_macro(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    K = max(int(y_true.max()), int(y_pred.max())) + 1
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def plot_confusion(cm: np.ndarray, out_path: Path):
    plt.figure(figsize=(8,7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def classification_report(y_true, y_pred) -> str:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    lines = []
    lines.append(f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    acc = accuracy_score(y_true, y_pred)
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        sup = np.sum(y_true == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
        lines.append(f"{c:12d} {prec:10.2f} {rec:10.2f} {f1:10.2f} {sup:10d}")
    lines.append(f"\naccuracy  {acc:.4f}")
    return "\n".join(lines)

def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    X, y = load_csv_xy(CSV_PATH, "label")
    Xtr, ytr, Xval, yval, Xte, yte = stratified_split(X, y, test_size=0.2, val_size=0.0, seed=SEED)

    rf = RandomForest(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=None,
        num_classes=int(y.max()) + 1,
        max_thresholds=16,
        max_samples_per_tree=20000,
        seed=SEED,
    )
    rf.fit(Xtr, ytr)

    yhat = rf.predict(Xte)
    acc  = accuracy_score(yte, yhat)
    f1m  = f1_macro(yte, yhat)
    cm   = confusion_matrix(yte, yhat)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_macro": f1m,
                   "n_estimators": N_ESTIMATORS, "max_depth": MAX_DEPTH}, f, indent=2)
    plot_confusion(cm, OUT_DIR / "confusion_matrix.png")

    print(f"accuracy={acc:.4f}  f1_macro={f1m:.4f}")
    print(f"Artifacts → {OUT_DIR.resolve()}")
    print("\nClassification report:\n")
    print(classification_report(yte, yhat))

if __name__ == "__main__":
    main()
