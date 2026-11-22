# Code Appendix — Multi-Dataset SmartFeature AI Prototype

The complete executable source code is provided in `app.py`, `experiment.py`, `feature_selection.py`, and `metrics.py`.

## Dataset loading

`experiment.py` provides `load_dataset(dataset_name)` for the Breast Cancer Wisconsin Diagnostic and Wine Recognition datasets bundled with scikit-learn. Both datasets are loaded as a feature DataFrame and target Series with their feature and class names.

## Experimental procedure

`run_experiment(k=10, repeats=10, dataset_name=...)` uses repeated stratified 5-fold cross-validation. With 10 repetitions, each dataset produces 50 train/test splits. For every split, the standard scaler is fitted only on the training partition. Each feature selector is also fitted only on the training partition. The held-out partition is used for prediction and evaluation.

## Feature selection and classifiers

The selectors are ANOVA F-Test, Kruskal-Wallis, and Random Forest Gini importance. The classifiers are Logistic Regression, Random Forest, and Gradient Boosting. An all-features baseline is also evaluated for each classifier.

## Evaluation measures

The prototype reports accuracy, macro-F1, feature-selection execution time, mean pairwise Jaccard similarity, and Nogueira stability. Split-level results, summaries, stability values, and selected feature names can be exported as CSV files from the Streamlit dashboard.

## Reproducibility

The default random seed is 42. For a fair dataset comparison, use the same number of repetitions, selected-feature count K where valid for both datasets, and random seed. Record the Python and package versions and the execution hardware when reporting timing results.
