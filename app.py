import pandas as pd
import plotly.express as px
import streamlit as st

from experiment import DATASET_NAMES, SELECTORS, load_dataset, run_experiment

st.set_page_config(page_title="SmartFeature AI", page_icon="🧪", layout="wide")
st.title("🧪 SmartFeature AI")
st.subheader("Statistical Feature Selection: Accuracy, Stability and Cost")
st.write(
    "A multi-dataset prototype comparing ANOVA F-Test, Kruskal–Wallis and "
    "Random Forest Gini feature selection on two classification benchmarks."
)

with st.sidebar:
    st.header("Experiment settings")
    dataset_name = st.selectbox("Choose dataset", DATASET_NAMES)
    X, y, target_names = load_dataset(dataset_name)
    valid_k = [value for value in [3, 5, 10, 15, 20] if value <= X.shape[1]]
    if X.shape[1] not in valid_k:
        valid_k.append(X.shape[1])
    valid_k = sorted(set(valid_k))
    default_k = 10 if 10 in valid_k else valid_k[-1]
    k = st.select_slider("Number of selected features (K)", options=valid_k, value=default_k)
    repeats = st.selectbox("CV repetitions", options=[1, 2, 5, 10], index=3)
    st.caption("Each repetition uses stratified 5-fold cross-validation.")
    run = st.button("Run experiment for selected dataset", type="primary", use_container_width=True)

if "dataset_results" not in st.session_state:
    st.session_state["dataset_results"] = {}

if run:
    with st.spinner(f"Running {repeats} repetitions × 5 folds on {dataset_name}... This may take a few minutes."):
        try:
            result = run_experiment(
                k=k, repeats=repeats, dataset_name=dataset_name
            )
            st.session_state["dataset_results"][dataset_name] = result
            st.success(f"Completed {dataset_name}.")
        except Exception as exc:
            st.exception(exc)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset", dataset_name)
c2.metric("Observations", f"{X.shape[0]:,}")
c3.metric("Original features", f"{X.shape[1]}")
c4.metric("Classes", f"{y.nunique()}")

with st.expander("Dataset information", expanded=False):
    descriptions = {
        "Breast Cancer Wisconsin": "Breast Cancer Wisconsin Diagnostic dataset (scikit-learn bundled copy): 569 observations and 30 numeric features.",
        "Wine Recognition": "Wine Recognition dataset (scikit-learn bundled copy): 178 observations and 13 numeric features.",
    }
    st.write(descriptions[dataset_name])
    class_counts = y.map(lambda value: str(target_names[int(value)])).value_counts()
    st.write("Class counts:")
    st.dataframe(class_counts.rename_axis("class").reset_index(name="count"), hide_index=True)
    st.write("Data preview:")
    st.dataframe(X.head(), use_container_width=True)

st.info(
    "Evaluation safeguard: scaling and feature selection are fitted on each "
    "training fold only. The held-out fold is used only for prediction and scoring."
)

results = st.session_state["dataset_results"].get(dataset_name)
if results is None:
    st.markdown("### How to use")
    st.markdown(
        "1. Select a dataset in the sidebar.\n"
        "2. Choose K and the number of CV repetitions.\n"
        "3. Click **Run experiment for selected dataset**.\n"
        "4. Repeat for the other dataset. Results are retained during this app session."
    )
else:
    summary = results["summary"]
    st.success(
        f"Experiment completed: {results['dataset_name']} — "
        f"{results['repeats']} repetitions × 5 folds = {results['n_splits']} train/test splits."
    )

    st.header("1. Predictive performance")
    performance = summary[summary["selector"] != results["baseline_label"]].copy()
    performance["Macro-F1 (%)"] = performance["macro_f1_mean"] * 100
    fig = px.bar(
        performance, x="classifier", y="Macro-F1 (%)", color="selector",
        barmode="group", error_y=performance["macro_f1_std"] * 100,
        title=f"Macro-F1 by selector and classifier — {dataset_name}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance summary")
    display = summary.copy()
    for column in ["accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]:
        display[column] = display[column] * 100
    display = display.rename(columns={
        "selector": "Feature selector", "classifier": "Classifier",
        "accuracy_mean": "Accuracy mean (%)", "accuracy_std": "Accuracy SD (pp)",
        "macro_f1_mean": "Macro-F1 mean (%)", "macro_f1_std": "Macro-F1 SD (pp)",
        "selection_time_mean_ms": "Selection time mean (ms)",
        "selection_time_std_ms": "Selection time SD (ms)",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.header("2. Feature-selection execution time")
    timing = summary[summary["selector"].isin(SELECTORS)]
    timing = timing.groupby("selector", as_index=False)["selection_time_mean_ms"].mean()
    fig_time = px.bar(
        timing, x="selector", y="selection_time_mean_ms",
        title=f"Mean feature-selection time — {dataset_name} (ms)",
        labels={"selector": "Selector", "selection_time_mean_ms": "Time (ms)"}
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.header("3. Feature-selection stability")
    st.caption("Stability is calculated across the five folds within each repetition, then averaged over repetitions.")
    st.dataframe(results["stability"], use_container_width=True, hide_index=True)
    stability_columns = st.columns(2)
    with stability_columns[0]:
        fig_j = px.bar(results["stability"], x="selector", y="jaccard", title="Mean pairwise Jaccard index")
        st.plotly_chart(fig_j, use_container_width=True)
    with stability_columns[1]:
        fig_n = px.bar(results["stability"], x="selector", y="nogueira", title="Nogueira stability index")
        st.plotly_chart(fig_n, use_container_width=True)

    st.header("4. Selected features across splits")
    selector_choice = st.selectbox("Choose selector", SELECTORS)
    feature_table = results["selected_features"]
    feature_table = feature_table[feature_table["selector"] == selector_choice]
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    st.header("5. Download experiment results")
    for label, key in [
        ("Split-level metrics", "raw"),
        ("Performance summary", "summary"),
        ("Stability metrics", "stability"),
        ("Selected features per split", "selected_features"),
    ]:
        csv_data = results[key].to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Download {label} CSV", data=csv_data,
            file_name=f"smartfeature_{dataset_name.lower().replace(' ', '_')}_{key}.csv",
            mime="text/csv", key=f"download_{dataset_name}_{key}"
        )

# Show a combined comparison once both datasets have been run in this app session.
all_results = st.session_state["dataset_results"]
if all(name in all_results for name in DATASET_NAMES):
    st.header("6. Cross-dataset comparison")
    comparison_frames = []
    for name in DATASET_NAMES:
        frame = all_results[name]["summary"].copy()
        frame = frame[frame["selector"] != all_results[name]["baseline_label"]]
        frame["dataset"] = name
        comparison_frames.append(frame)
    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison["Macro-F1 (%)"] = comparison["macro_f1_mean"] * 100
    comparison["Accuracy (%)"] = comparison["accuracy_mean"] * 100
    st.dataframe(
        comparison[["dataset", "selector", "classifier", "Accuracy (%)", "Macro-F1 (%)"]],
        use_container_width=True, hide_index=True
    )
    fig_compare = px.bar(
        comparison, x="selector", y="Macro-F1 (%)", color="dataset",
        barmode="group", facet_col="classifier",
        title="Feature-selection performance across datasets"
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    st.download_button(
        "Download cross-dataset comparison CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="smartfeature_cross_dataset_comparison.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Research note: execution times depend on hardware and software versions. "
    "Run both datasets with the same K, repetitions and random seed before comparing results. "
    "The datasets differ in sample count, feature count and number of classes."
)
