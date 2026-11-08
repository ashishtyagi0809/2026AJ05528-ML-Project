"""
Adult Income Classification - Streamlit App
Student: 2026AJ05528 | ML Assignment 2

Loads the 5 pre-trained pipelines from model/ (no retraining here),
lets the user upload test data, pick a model, and see the evaluation.
"""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

MODEL_DIR = "model"
TARGET = "income"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Adult Income Classifier", layout="wide")
st.title("Adult Income Classification")
st.caption("Predicts whether a person earns more than $50K/year "
           "(UCI Adult dataset) | ML Assignment 2 | 2026AJ05528")


# ---------- Helpers ----------------------------------------------------------
@st.cache_resource
def load_model(file_name):
    """Load a saved pipeline once and reuse it (cached)."""
    return joblib.load(os.path.join(MODEL_DIR, file_name))


def clean_columns(df):
    """Same column-name cleaning as in train_models.py."""
    df.columns = [c.strip().replace(".", "_").replace("-", "_") for c in df.columns]
    return df


def encode_target(series):
    """Accepts 0/1 or the original '>50K' / '<=50K' labels."""
    if series.dtype == object:
        s = series.astype(str).str.strip().str.rstrip(".")
        return (s == ">50K").astype(int)
    return series.astype(int)


def positive_scores(model, X):
    """Probability (or decision score) of the >50K class, used for AUC."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


# ---------- Sidebar: inputs -----------------------------------------------------
st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader("CSV file (must include 'income' column)",
                                    type="csv")

if os.path.exists("test_data.csv"):
    with open("test_data.csv", "rb") as f:
        st.sidebar.download_button("Download sample test_data.csv", f,
                                   file_name="test_data.csv")

st.sidebar.header("2. Choose a model")
model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

if uploaded is None:
    st.info("Upload a CSV from the sidebar to begin. "
            "You can download the sample test_data.csv there.")
    st.stop()

# ---------- Data ----------------------------------------------------------------
df = clean_columns(pd.read_csv(uploaded, skipinitialspace=True))
df = df.replace("?", pd.NA)

if TARGET not in df.columns:
    st.error(f"The uploaded file has no '{TARGET}' column, "
             "so the model cannot be evaluated.")
    st.stop()

y_true = encode_target(df[TARGET])
X = df.drop(columns=[TARGET])

with st.expander(f"Preview uploaded data ({df.shape[0]} rows, {X.shape[1]} features)"):
    st.dataframe(df.head(20))

# ---------- Predict + evaluate --------------------------------------------------
model = load_model(MODEL_FILES[model_name])
y_pred = model.predict(X)
y_score = positive_scores(model, X)

st.subheader(f"Evaluation metrics: {model_name}")
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "AUC": roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else float("nan"),
    "Precision": precision_score(y_true, y_pred, zero_division=0),
    "Recall": recall_score(y_true, y_pred, zero_division=0),
    "F1": f1_score(y_true, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_true, y_pred),
}
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, labels=[0, 1],
                                   target_names=["<=50K", ">50K"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))

# ---------- All-model comparison (from training run) ---------------------------
metrics_path = os.path.join(MODEL_DIR, "metrics.csv")
if os.path.exists(metrics_path):
    st.subheader("Comparison of all 5 models (test set from training)")
    st.dataframe(pd.read_csv(metrics_path), hide_index=True)