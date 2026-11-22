"""Feature-selection stability metrics."""
from itertools import combinations
import numpy as np


def mean_pairwise_jaccard(selected_sets):
    """Mean Jaccard similarity over all pairs of selected-feature sets."""
    values = []
    for a, b in combinations(selected_sets, 2):
        a, b = set(a), set(b)
        union = a | b
        values.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(values)) if values else float("nan")


def nogueira_stability(selected_sets, n_features):
    """
    Nogueira chance-corrected stability estimate for a binary
    feature-selection matrix. Returns NaN if the denominator is zero.
    """
    if len(selected_sets) < 2:
        return float("nan")

    z = np.zeros((len(selected_sets), n_features), dtype=float)
    for row, selected in enumerate(selected_sets):
        z[row, list(selected)] = 1.0

    p_hat = z.mean(axis=0)
    k_bar = z.sum(axis=1).mean()
    sample_variances = z.var(axis=0, ddof=1)
    denominator = (k_bar / n_features) * (1.0 - k_bar / n_features)

    if denominator == 0:
        return float("nan")

    return float(1.0 - (sample_variances.mean() / denominator))
