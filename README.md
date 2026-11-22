# SmartFeature AI — Multi-Dataset Prototype

This version evaluates the same feature-selection and classification workflow on:

1. **Breast Cancer Wisconsin Diagnostic** — original dataset bundled with scikit-learn (569 observations, 30 numeric features, 2 classes).
2. **Wine Recognition** — additional benchmark dataset bundled with scikit-learn (178 observations, 13 numeric features, 3 classes).

## Run on Windows / VS Code

Activate your existing virtual environment from the project folder:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app.py
```

## Experiment procedure

1. Choose **Breast Cancer Wisconsin** in the sidebar.
2. Set **K = 10** and **CV repetitions = 10**; click **Run experiment for selected dataset**.
3. Download the CSV results.
4. Choose **Wine Recognition**. Use the same K=10, repetitions=10 and random seed; click Run.
5. The app retains both result sets during the same Streamlit session and displays a cross-dataset comparison after both runs complete.

Each dataset uses repeated stratified 5-fold cross-validation. Thus 10 repetitions produce 50 train/test splits per dataset. The scaler and feature selector are fitted on the training fold only.

## Methods

Feature selectors: ANOVA F-Test, Kruskal-Wallis, Random Forest Gini.

Classifiers: Logistic Regression, Random Forest, Gradient Boosting.

Metrics: accuracy, macro-F1, feature-selection time, mean pairwise Jaccard stability, and Nogueira stability.

## Important research note

The Wine Recognition dataset is an additional benchmark dataset, not a dataset newly collected by this project. Describe it accurately in the paper. Compare methods within each dataset and discuss cross-dataset differences carefully because sample sizes, feature counts, and class counts differ. Do not claim that the prototype proves generalization to all datasets.
