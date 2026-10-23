
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="OncoLens | ML Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: #f5f7fb;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101b35 0%, #172744 100%);
}
[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}
.hero {
    padding: 28px 30px;
    border-radius: 20px;
    background: linear-gradient(115deg, #101b35, #176b78);
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 32px;
    margin: 0 0 8px 0;
}
.hero p {
    color: #dbeafe;
    margin: 0;
    font-size: 14px;
}
.metric-card {
    background: white;
    border: 1px solid #e7ebf2;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, .04);
}
.metric-label {
    color: #64748b;
    font-size: 13px;
    margin-bottom: 8px;
}
.metric-value {
    color: #14213d;
    font-size: 28px;
    font-weight: 800;
}
.metric-note {
    color: #0f766e;
    font-size: 12px;
    margin-top: 6px;
}
.section-title {
    color: #14213d;
    font-size: 21px;
    font-weight: 700;
    margin: 20px 0 12px 0;
}
.info-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e7ebf2;
}
div[data-testid="stForm"] {
    background: white;
    padding: 20px;
    border: 1px solid #e7ebf2;
    border-radius: 16px;
}
.stButton > button, .stFormSubmitButton > button {
    border-radius: 10px;
    font-weight: 600;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.joblib")
    features = joblib.load("models/feature_names.joblib")
    return model, features

@st.cache_data
def load_dataset():
    return load_breast_cancer(as_frame=True)

if not os.path.exists("models/best_model.joblib"):
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()

model, feature_names = load_artifacts()
dataset = load_dataset()

X = dataset.data
y = dataset.target

class_names = ["Malignant", "Benign"]
class_colors = {
    "Malignant": "#e05d6f",
    "Benign": "#0d9488"
}

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🔬 OncoLens")
    st.caption("MACHINE LEARNING ANALYTICS")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Prediction Studio",
            "Model Analytics",
            "Dataset Explorer",
            "About Project"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("### Project status")
    st.success("Model loaded")
    st.caption("Algorithm: Logistic Regression")
    st.caption("Dataset: Wisconsin Diagnostic")
    st.caption("Educational use only")

# --------------------------------------------------
# REUSABLE UI HELPERS
# --------------------------------------------------
def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def make_chart_layout(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", color="#334155"),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", y=-0.2)
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf1f7")
    return fig

# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------
if page == "Overview":
    hero(
        "Breast Cancer Analytics",
        "Explore dataset insights and machine learning performance in one workspace."
    )

    # Use the original held-out split for reported performance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions, pos_label=0)
    precision = precision_score(y_test, predictions, pos_label=0)
    f1 = f1_score(y_test, predictions, pos_label=0)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("TOTAL SAMPLES", f"{len(X):,}", "Wisconsin dataset")
    with c2:
        metric_card("INPUT FEATURES", f"{X.shape[1]}", "Numerical features")
    with c3:
        metric_card("TEST ACCURACY", f"{accuracy:.1%}", "Held-out test split")
    with c4:
        metric_card("MALIGNANT RECALL", f"{recall:.1%}", "Positive class: malignant")

    st.markdown('<div class="section-title">Dataset intelligence</div>',
                unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        counts = (
            y.map({0: "Malignant", 1: "Benign"})
            .value_counts()
            .rename_axis("Diagnosis")
            .reset_index(name="Samples")
        )

        fig = px.pie(
            counts,
            names="Diagnosis",
            values="Samples",
            hole=0.62,
            color="Diagnosis",
            color_discrete_map=class_colors,
            title="Class distribution"
        )
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(make_chart_layout(fig), use_container_width=True)

    with right:
        feature = st.selectbox(
            "Explore feature distribution",
            options=list(X.columns),
            index=0
        )

        plot_data = X[[feature]].copy()
        plot_data["Diagnosis"] = y.map({
            0: "Malignant",
            1: "Benign"
        })

        fig = px.histogram(
            plot_data,
            x=feature,
            color="Diagnosis",
            barmode="overlay",
            opacity=0.70,
            color_discrete_map=class_colors,
            title=f"Distribution of {feature}"
        )
        st.plotly_chart(make_chart_layout(fig), use_container_width=True)

    st.markdown('<div class="section-title">Model summary</div>',
                unsafe_allow_html=True)

    st.write(
        f"The selected model achieved **{accuracy:.1%} accuracy** "
        f"on the held-out test split. Malignant-class precision is "
        f"{precision:.1%}, recall is {recall:.1%}, and F1-score is {f1:.1%}."
    )

    st.info(
        "Educational demonstration only. These metrics do not establish "
        "clinical validity or suitability for patient care."
    )

# --------------------------------------------------
# PAGE 2: PREDICTION STUDIO
# --------------------------------------------------
elif page == "Prediction Studio":
    hero(
        "Prediction Studio",
        "Explore model outputs using example measurements from the dataset."
    )

    st.warning(
        "This tool is for ML education, not medical diagnosis. "
        "Use only example data; never use these predictions for healthcare decisions."
    )

    mode = st.radio(
        "Input method",
        ["Choose a dataset sample", "Enter measurements manually"],
        horizontal=True
    )

    if mode == "Choose a dataset sample":
        sample_index = st.number_input(
            "Dataset sample index",
            min_value=0,
            max_value=len(X) - 1,
            value=0,
            step=1
        )
        sample = X.iloc[[int(sample_index)]].copy()

        actual_class = class_names[int(y.iloc[int(sample_index)])]
        st.caption(f"Reference dataset label: {actual_class}")

    else:
        st.write(
            "Enter numerical values within the ranges shown in the dataset. "
            "Defaults are feature medians and are not a real patient's measurements."
        )

        medians = X.median()
        sample_values = {}

        with st.form("manual_prediction_form"):
            for group_name, prefix in [
                ("Mean features", "mean "),
                ("Standard error features", " "),
                ("Worst features", "worst ")
            ]:
                if prefix == " ":
                    group_features = [
                        f for f in feature_names
                        if f.endswith(" error")
                    ]
                else:
                    group_features = [
                        f for f in feature_names
                        if f.startswith(prefix)
                    ]

                if not group_features:
                    continue

                with st.expander(group_name, expanded=(group_name == "Mean features")):
                    cols = st.columns(2)

                    for i, feature in enumerate(group_features):
                        lo = float(X[feature].min())
                        hi = float(X[feature].max())
                        default = float(medians[feature])

                        # Avoid invalid ranges if a feature is constant.
                        if lo == hi:
                            sample_values[feature] = default
                        else:
                            sample_values[feature] = cols[i % 2].number_input(
                                feature,
                                min_value=lo,
                                max_value=hi,
                                value=default,
                                step=max((hi - lo) / 100, 0.0001),
                                format="%.5f",
                                key=f"manual_{feature}"
                            )

            submitted = st.form_submit_button(
                "Analyze example measurements",
                use_container_width=True
            )

        if not submitted:
            st.stop()

        sample = pd.DataFrame([sample_values], columns=feature_names)

    # Keep feature order identical to training.
    sample = sample[feature_names]
    prediction = int(model.predict(sample)[0])
    label = class_names[prediction]

    st.divider()
    st.subheader("Model output")

    if label == "Malignant":
        st.error("Model classification: Malignant")
    else:
        st.success("Model classification: Benign")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)[0]
        probability_df = pd.DataFrame({
            "Class": class_names,
            "Model-estimated probability": probs
        })

        c1, c2 = st.columns([1, 1])

        with c1:
            fig = px.bar(
                probability_df,
                x="Class",
                y="Model-estimated probability",
                color="Class",
                color_discrete_map=class_colors,
                range_y=[0, 1],
                title="Model-estimated class probabilities"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(make_chart_layout(fig), use_container_width=True)

        with c2:
            st.markdown("#### Probability details")
            for _, row in probability_df.iterrows():
                st.write(
                    f"**{row['Class']}** — "
                    f"{row['Model-estimated probability']:.2%}"
                )
                st.progress(float(row["Model-estimated probability"]))

    st.download_button(
        "Download example prediction",
        data=pd.concat([
            sample.reset_index(drop=True),
            pd.DataFrame({
                "Model classification": [label]
            })
        ], axis=1).to_csv(index=False).encode("utf-8"),
        file_name="example_prediction.csv",
        mime="text/csv"
    )

    st.caption(
        "Model probabilities are not calibrated clinical risk estimates. "
        "The output must not be used to diagnose or rule out cancer."
    )

# --------------------------------------------------
# PAGE 3: MODEL ANALYTICS
# --------------------------------------------------

elif page == "Model Analytics":
    hero(
        "Model Analytics",
        "Compare classification algorithms and inspect the selected model's performance."
    )

    comparison_path = "models/model_comparison.csv"

    if os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path)

        st.markdown("### Algorithm leaderboard")
        st.dataframe(
            comparison.style.format({
                col: "{:.2%}"
                for col in comparison.select_dtypes(include="number").columns
            }),
            use_container_width=True,
            hide_index=True
        )

        numeric_cols = comparison.select_dtypes(
            include="number"
        ).columns.tolist()

        if "Model" in comparison.columns and numeric_cols:
            metric = st.selectbox(
                "Choose a metric to compare",
                numeric_cols,
                index=0
            )

            fig = px.bar(
                comparison.sort_values(metric, ascending=False),
                x="Model",
                y=metric,
                color="Model",
                text=comparison.sort_values(
                    metric, ascending=False
                )[metric].map(lambda value: f"{value:.2%}"),
                title=f"Model comparison — {metric}",
                range_y=[0, 1]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(
                make_chart_layout(fig),
                use_container_width=True
            )

            best_row = comparison.loc[comparison[metric].idxmax()]
            st.success(
                f"Highest {metric}: {best_row['Model']} "
                f"({best_row[metric]:.2%})"
            )

    else:
        st.warning(
            "Model comparison file not found. Run train_model.py "
            "to generate models/model_comparison.csv."
        )

    st.divider()
    st.markdown("### Selected model evaluation")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    predictions = model.predict(X_test)

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Score": [
            accuracy_score(y_test, predictions),
            precision_score(y_test, predictions, pos_label=0),
            recall_score(y_test, predictions, pos_label=0),
            f1_score(y_test, predictions, pos_label=0)
        ]
    })

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            metrics,
            x="Metric",
            y="Score",
            text=metrics["Score"].map(lambda value: f"{value:.1%}"),
            title="Selected model metrics",
            range_y=[0, 1]
        )
        st.plotly_chart(
            make_chart_layout(fig),
            use_container_width=True
        )

    with right:
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted malignant", "Predicted benign"],
            y=["Actual malignant", "Actual benign"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Teal"
        ))
        fig.update_layout(title="Confusion matrix")
        st.plotly_chart(
            make_chart_layout(fig),
            use_container_width=True
        )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 0]
        fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label=0)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Selected model (AUC = {roc_auc:.3f})"
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(dash="dash")
        ))
        fig.update_layout(
            title="ROC curve — malignant class",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(
            make_chart_layout(fig),
            use_container_width=True
        )

    st.download_button(
        "Download test metrics",
        data=metrics.to_csv(index=False).encode("utf-8"),
        file_name="test_metrics.csv",
        mime="text/csv"
    )

    st.caption(
        "Comparisons use the saved training results. The selected model's "
        "metrics are calculated on a held-out split. Educational use only."
    )


# --------------------------------------------------
# PAGE 4: DATASET EXPLORER
# --------------------------------------------------
elif page == "Dataset Explorer":
    hero(
        "Dataset Explorer",
        "Inspect the original dataset, feature summaries, and sample records."
    )

    df = X.copy()
    df["Diagnosis"] = y.map({
        0: "Malignant",
        1: "Benign"
    })

    search = st.text_input("Search feature names")
    filtered_features = [
        f for f in X.columns
        if search.lower() in f.lower()
    ]

    selected_features = st.multiselect(
        "Choose features to display",
        options=filtered_features,
        default=filtered_features[:5]
    )

    if selected_features:
        st.dataframe(
            df[selected_features + ["Diagnosis"]],
            use_container_width=True,
            height=350
        )

    st.markdown("### Descriptive statistics")
    st.dataframe(X.describe().T, use_container_width=True)

    st.download_button(
        "Download full dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="breast_cancer_dataset.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# PAGE 5: ABOUT
# --------------------------------------------------
elif page == "About Project":
    hero(
        "About OncoLens",
        "A machine learning internship project for learning classification and evaluation."
    )

    st.markdown("""
    ### Project objectives
    - Understand a real-world medical research dataset.
    - Train a classification model using scikit-learn.
    - Evaluate performance with multiple metrics.
    - Visualize predictions, distributions, and model performance.
    - Build an interactive Streamlit application.

    ### Technology stack
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Plotly
    - Streamlit
    - Joblib

    ### Dataset
    Breast Cancer Wisconsin Diagnostic dataset, distributed through
    scikit-learn. The data contains numerical features computed from
    digitized images of breast cell nuclei.

    ### Important limitation
    OncoLens is an educational software demonstration, not a validated
    medical device. It is not suitable for screening, diagnosis, treatment,
    or any real patient-care decision.
    """)

    st.markdown("### Project details")
    st.write("Models evaluated: Logistic Regression, Decision Tree, Random Forest")
    st.write("Selected saved model: Logistic Regression")
    st.write("Input features:", len(feature_names))

st.divider()
st.caption("OncoLens • Educational ML project • Not for clinical use")
