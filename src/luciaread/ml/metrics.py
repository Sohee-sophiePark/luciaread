"""Classification and calibration metrics (numpy only)."""

import numpy as np


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))))


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    """Expected calibration error with equal-width confidence bins, plus per-bin rows."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        acc, cf = float((pred[m] == y[m]).mean()), float(conf[m].mean())
        total += m.mean() * abs(acc - cf)
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "acc": acc, "conf": cf})
    return float(total), rows


def auc(scores: np.ndarray, pos: np.ndarray) -> float:
    """Binary ROC AUC via the rank-sum statistic (ties averaged)."""
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for s in np.unique(scores):
        tie = scores == s
        ranks[tie] = ranks[tie].mean()
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def macro_auc(probs: np.ndarray, y: np.ndarray) -> float:
    """Binary: AUC of class 1. Multiclass: mean one-vs-rest AUC."""
    if probs.shape[1] == 2:
        return auc(probs[:, 1], y == 1)
    return float(np.mean([auc(probs[:, k], y == k) for k in range(probs.shape[1])]))


def sens_spec(p_pos: np.ndarray, pos: np.ndarray, threshold: float) -> tuple[float, float]:
    hit = p_pos >= threshold
    return float(hit[pos].mean()), float((~hit[~pos]).mean())


def threshold_for_sensitivity(p_pos: np.ndarray, pos: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity on these scores is at least `target`."""
    return float(np.sort(p_pos[pos])[int(np.floor((1 - target) * pos.sum()))])


def cluster_bootstrap(fn, groups: np.ndarray, n: int = 1000, seed: int = 7) -> tuple[float, float]:
    """95% CI of fn(index_array) resampling whole groups (patients) with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = [
            fn(np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])) for _ in range(n)
        ]
    lo, hi = np.nanpercentile(np.array(vals, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)
