# SmartFeature-AI
### Hypothesis Testing for Feature Selection in Machine Learning Models

## 1. Project Overview

SmartFeature-AI is a Python-based research prototype developed to study and compare different feature selection techniques in Machine Learning.

The project evaluates feature selection methods based on predictive performance, feature selection stability, and computational cost.

The application is developed using **Python and Streamlit**.

## 2. Research Objectives

- Compare different feature selection techniques.
- Evaluate the effect of feature selection on machine learning performance.
- Analyze the stability of selected features across repeated data splits.
- Measure the computational time required by different methods.
- Test the methods on multiple datasets.

## 3. Datasets

The prototype supports two datasets:

| Dataset | Samples | Features | Classes |
|---|---:|---:|---:|
| Breast Cancer Wisconsin | 569 | 30 | 2 |
| Wine Recognition | 178 | 13 | 3 |

Both datasets are used to evaluate the feature selection techniques under different classification problems.

## 4. Feature Selection Techniques

The prototype compares three techniques:

1. **ANOVA F-Test** – Measures the relationship between individual features and class labels.
2. **Kruskal-Wallis Test** – A non-parametric statistical test used to compare feature distributions across classes.
3. **Random Forest Gini Importance** – Uses feature importance scores from a Random Forest model.

## 5. Evaluation Metrics

The project evaluates:

- Accuracy
- Macro-F1 Score
- Jaccard Stability
- Nogueira Stability
- Computational Time

These metrics help analyze predictive performance, consistency of selected features, and computational cost.

## 6. Technologies Used

- Python
- Streamlit
- Scikit-learn
- NumPy
- Pandas
- SciPy
- Matplotlib

## 7. Project Structure

```text
SmartFeature-AI/
│
├── app.py
├── experiment.py
├── feature_selection.py
├── metrics.py
├── requirements.txt
├── README.md
├── CODE_APPENDIX.md
└── Code_Appendix_for_Research_Paper.docx
```

## 8. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/bambulgeprajwal-sketch/SmartFeature-AI.git
```

### Step 2: Open the project folder

```bash
cd SmartFeature-AI
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 9. How to Use

1. Launch the Streamlit application.
2. Select a dataset.
3. Configure the experiment settings.
4. Select or run the feature selection methods.
5. Execute the experiment.
6. Compare the evaluation results.
7. Download the generated results if available.

## 10. Expected Research Contribution

This prototype provides an experimental framework for comparing statistical and model-based feature selection methods across multiple datasets.

It aims to study the trade-offs between predictive performance, selection stability, and computational cost.

## 11. Future Improvements

- Support additional datasets.
- Include more feature selection techniques.
- Expand statistical comparisons.
- Improve result visualization.
- Conduct experiments with additional configurations.

## 12. Author

**Prajwal Bambulge**

B.Tech – Artificial Intelligence and Data Science

Presidency University, Bengaluru

## 13. Disclaimer

This repository contains a research prototype developed for academic and experimental purposes. Experimental findings should be independently verified before being used as research conclusions.
