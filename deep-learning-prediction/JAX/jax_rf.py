"""
jax_rf.py
deep-learning-prediction/JAX/jax_rf.py
Author: Esma Ali
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand

@dataclass
class TreeNode:
    is_leaf: bool
    prediction: Optional[int] = None
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional[int] = None
    right: Optional[int] = None

def gini_impurity_np(y_np: np.ndarray, num_classes: int) -> float:
    if y_np.size == 0:
        return 0.0
    counts = np.bincount(y_np.astype(np.int32), minlength=num_classes)
    p = counts / y_np.size
    return float(1.0 - np.sum(p * p))

def candidate_thresholds(x_np: np.ndarray, max_thresholds: int) -> np.ndarray:
    x_np = x_np.astype(np.float32, copy=False)
    uniq = np.unique(x_np)
    if uniq.size <= 1:
        return np.array([], dtype=np.float32)
    if uniq.size <= max_thresholds:
        return ((uniq[:-1] + uniq[1:]) * 0.5).astype(np.float32, copy=False)
    qs = np.linspace(0.05, 0.95, num=max_thresholds, dtype=np.float32)
    thr = np.quantile(x_np, qs, method="linear")
    return np.unique(thr).astype(np.float32, copy=False)

def best_split(
    X: np.ndarray,
    y: np.ndarray,
    features: np.ndarray,
    num_classes: int,
    min_leaf: int,
    max_thresholds: int,
) -> Tuple[Optional[int], Optional[float], float]:
    n = X.shape[0]
    base_imp = gini_impurity_np(y, num_classes)
    best_gain = -1.0
    best_feat: Optional[int] = None
    best_thr: Optional[float] = None

    for f in features.tolist():
        x = X[:, f]
        thrs = candidate_thresholds(x, max_thresholds)
        if thrs.size == 0:
            continue
        for t in thrs:
            left = (x <= t)
            nL = int(left.sum())
            nR = n - nL
            if nL < min_leaf or nR < min_leaf:
                continue
            gL = gini_impurity_np(y[left], num_classes)
            gR = gini_impurity_np(y[~left], num_classes)
            gain = base_imp - (nL / n) * gL - (nR / n) * gR
            if gain > best_gain:
                best_gain, best_feat, best_thr = float(gain), int(f), float(t)

    return best_feat, best_thr, best_gain

class DecisionTree:
    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        max_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        max_thresholds: int = 16,
        seed: int = 0,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.num_classes = num_classes
        self.max_thresholds = max_thresholds
        self.key = jrand.key(seed)
        self.nodes: List[TreeNode] = []

    def _majority_np(self, y_np: np.ndarray) -> int:
        return int(np.argmax(np.bincount(y_np.astype(np.int32), minlength=self.num_classes)))

    def fit(self, X_in, y_in):
        X_full = np.asarray(X_in)
        y_full = np.asarray(y_in).astype(np.int32)
        if self.num_classes is None:
            self.num_classes = int(y_full.max()) + 1
        n, d = X_full.shape
        if self.max_features is None:
            self.max_features = max(1, int(np.sqrt(d)))
        self.nodes = []

        def grow(idx_np: np.ndarray, depth: int) -> int:
            y_sub = y_full[idx_np]
            X_sub = X_full[idx_np, :]

            if (
                depth >= self.max_depth
                or idx_np.size < self.min_samples_split
                or np.all(y_sub == y_sub[0])
            ):
                pred = self._majority_np(y_sub)
                i = len(self.nodes)
                self.nodes.append(TreeNode(True, prediction=pred))
                return i

            self.key, subk = jrand.split(self.key)
            feat_idx = np.array(jrand.choice(subk, a=jnp.arange(d),
                                             shape=(self.max_features,), replace=False))
            f, thr, gain = best_split(
                X_sub, y_sub, feat_idx, self.num_classes, self.min_samples_leaf, self.max_thresholds
            )
            if f is None or gain <= 0.0:
                pred = self._majority_np(y_sub)
                i = len(self.nodes)
                self.nodes.append(TreeNode(True, prediction=pred))
                return i

            left_mask = (X_sub[:, f] <= thr)
            left_idx  = idx_np[left_mask]
            right_idx = idx_np[~left_mask]

            i = len(self.nodes)
            self.nodes.append(TreeNode(False, feature=f, threshold=thr))
            L = grow(left_idx,  depth + 1)
            R = grow(right_idx, depth + 1)
            self.nodes[i].left = L
            self.nodes[i].right = R
            return i

        grow(np.arange(n), 0)

    def _predict_one_np(self, x_np: np.ndarray) -> int:
        i = 0
        while True:
            node = self.nodes[i]
            if node.is_leaf:
                return node.prediction
            i = node.left if float(x_np[node.feature]) <= float(node.threshold) else node.right

    def predict(self, X_in) -> np.ndarray:
        X_np = np.asarray(X_in)
        return np.array([self._predict_one_np(X_np[i, :]) for i in range(X_np.shape[0])], dtype=np.int32)

class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 8,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        max_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        max_thresholds: int = 16,
        max_samples_per_tree: Optional[int] = 20000,
        seed: int = 0,
    ):
        self.n_estimators = n_estimators
        self.params = dict(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            num_classes=num_classes,
            max_thresholds=max_thresholds,
        )
        self.max_samples_per_tree = max_samples_per_tree
        self.key = jrand.key(seed)
        self.trees: List[DecisionTree] = []

    def fit(self, X_in, y_in):
        X = np.asarray(X_in)
        y = np.asarray(y_in).astype(np.int32)
        n = X.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            self.key, k_boot, k_tree = jrand.split(self.key, 3)
            m = min(n, self.max_samples_per_tree) if self.max_samples_per_tree else n
            idx = np.array(jrand.choice(k_boot, a=jnp.arange(n), shape=(m,), replace=True))
            seed = int(jrand.randint(k_tree, shape=(), minval=0, maxval=2**31-1, dtype=jnp.int32))
            t = DecisionTree(seed=seed, **self.params)
            t.fit(X[idx, :], y[idx])
            self.trees.append(t)

    def predict(self, X_in) -> np.ndarray:
        X = np.asarray(X_in)
        preds = np.stack([t.predict(X) for t in self.trees], axis=1)  # [N, T]
        K = max(t.num_classes for t in self.trees)
        out = np.zeros((preds.shape[0],), dtype=np.int32)
        for i in range(preds.shape[0]):
            out[i] = int(np.argmax(np.bincount(preds[i], minlength=K)))
        return out
