"""Repeated, fold-isolated cross-validation for multiple benchmark datasets."""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from feature_selection import select_features
from metrics import mean_pairwise_jaccard, nogueira_stability

SELECTORS = ["ANOVA F-Test", "Kruskal-Wallis", "RF Gini"]
CLASSIFIERS = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
DATASET_NAMES = ["Breast Cancer Wisconsin", "Wine Recognition"]
N_SPLITS = 5


def load_dataset(dataset_name="Breast Cancer Wisconsin"):
    """Return (X DataFrame, y Series, target class names) for a supported dataset."""
    if dataset_name == "Breast Cancer Wisconsin":
        bunch = load_breast_cancer()
    elif dataset_name == "Wine Recognition":
        bunch = load_wine()
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose one of: {', '.join(DATASET_NAMES)}"
        )

    X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    y = pd.Series(bunch.target, name="target")
    return X, y, np.asarray(bunch.target_names)


def load_paper_dataset():
    """Backward-compatible loader for the original paper dataset."""
    return load_dataset("Breast Cancer Wisconsin")


def make_classifier(name, random_state=42):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100, criterion="gini", random_state=random_state, n_jobs=1
        )
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=random_state
        )
    raise ValueError(f"Unknown classifier: {name}")


def run_experiment(k=10, repeats=10, random_state=42,
                   dataset_name="Breast Cancer Wisconsin"):
    """Run repeated stratified 5-fold CV; fit preprocessing/selectors on train folds only."""
    X_df, y_series, target_names = load_dataset(dataset_name)
    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy()
    feature_names = np.asarray(X_df.columns)
    n_features = X.shape[1]
    baseline_label = f"All {n_features} features"

    if not 1 <= k <= n_features:
        raise ValueError(f"K must be between 1 and {n_features} for {dataset_name}.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=repeats, random_state=random_state
    )
    rows = []
    selected_by_selector_repeat = {
        selector: defaultdict(list) for selector in SELECTORS
    }
    selected_feature_rows = []

    for split_number, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        repeat_id = split_number // N_SPLITS
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Unfiltered baseline using every feature in the selected dataset.
        for classifier_name in CLASSIFIERS:
            model = make_classifier(classifier_name, random_state)
            if classifier_name == "Logistic Regression":
                train_data, test_data = X_train_scaled, X_test_scaled
            else:
                train_data, test_data = X_train, X_test
            model.fit(train_data, y_train)
            predictions = model.predict(test_data)
            rows.append({
                "dataset": dataset_name,
                "split": split_number + 1,
                "repeat": repeat_id + 1,
                "selector": baseline_label,
                "classifier": classifier_name,
                "k": n_features,
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1": f1_score(y_test, predictions, average="macro"),
                "selection_time_ms": 0.0,
            })

        for selector_name in SELECTORS:
            selected, _scores, selection_ms = select_features(
                X_train, y_train, selector_name, k, random_state
            )
            selected_by_selector_repeat[selector_name][repeat_id].append(selected)
            selected_feature_rows.append({
                "dataset": dataset_name,
                "split": split_number + 1,
                "repeat": repeat_id + 1,
                "selector": selector_name,
                "selected_features": ", ".join(feature_names[selected]),
            })

            for classifier_name in CLASSIFIERS:
                model = make_classifier(classifier_name, random_state)
                if classifier_name == "Logistic Regression":
                    train_data = X_train_scaled[:, selected]
                    test_data = X_test_scaled[:, selected]
                else:
                    train_data = X_train[:, selected]
                    test_data = X_test[:, selected]
                model.fit(train_data, y_train)
                predictions = model.predict(test_data)
                rows.append({
                    "dataset": dataset_name,
                    "split": split_number + 1,
                    "repeat": repeat_id + 1,
                    "selector": selector_name,
                    "classifier": classifier_name,
                    "k": k,
                    "accuracy": accuracy_score(y_test, predictions),
                    "macro_f1": f1_score(y_test, predictions, average="macro"),
                    "selection_time_ms": selection_ms,
                })

    stability_rows = []
    for selector_name in SELECTORS:
        jaccard_values, nogueira_values = [], []
        for repeat_id in range(repeats):
            selected_sets = selected_by_selector_repeat[selector_name][repeat_id]
            jaccard_values.append(mean_pairwise_jaccard(selected_sets))
            nogueira_values.append(nogueira_stability(selected_sets, n_features))
        stability_rows.append({
            "dataset": dataset_name,
            "selector": selector_name,
            "jaccard": float(np.nanmean(jaccard_values)),
            "nogueira": float(np.nanmean(nogueira_values)),
        })

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["selector", "classifier"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            selection_time_mean_ms=("selection_time_ms", "mean"),
            selection_time_std_ms=("selection_time_ms", "std"),
        )
    )
    stability = pd.DataFrame(stability_rows)
    selected_features = pd.DataFrame(selected_feature_rows)

    return {
        "dataset_name": dataset_name,
        "raw": raw,
        "summary": summary,
        "stability": stability,
        "selected_features": selected_features,
        "dataset_shape": X.shape,
        "target_names": list(target_names),
        "n_splits": repeats * N_SPLITS,
        "k": k,
        "repeats": repeats,
        "baseline_label": baseline_label,
    }
