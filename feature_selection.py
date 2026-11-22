"""Feature selectors used by the SmartFeature AI research prototype."""
from time import perf_counter

import numpy as np
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif


def select_features(X, y, method, k, random_state=42):
    """
    Fit one selector on training data and return selected column indices,
    the full score vector, and selector execution time in milliseconds.
    """
    started = perf_counter()

    if method == "ANOVA F-Test":
        scores, _ = f_classif(X, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    elif method == "Kruskal-Wallis":
        classes = np.unique(y)
        scores = np.zeros(X.shape[1], dtype=float)
        for j in range(X.shape[1]):
            groups = [X[y == cls, j] for cls in classes]
            try:
                scores[j] = kruskal(*groups).statistic
            except ValueError:
                scores[j] = 0.0
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    elif method == "RF Gini":
        selector_model = RandomForestClassifier(
            n_estimators=100, criterion="gini", random_state=random_state,
            n_jobs=1
        )
        selector_model.fit(X, y)
        scores = selector_model.feature_importances_

    else:
        raise ValueError(f"Unknown selector: {method}")

    selected = np.argsort(scores)[::-1][:k]
    elapsed_ms = (perf_counter() - started) * 1000.0
    return selected, scores, elapsed_ms
