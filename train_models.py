"""
Adult Income Classification - Train 5 models and evaluate them.
Run this in the BITS Virtual Lab (screenshot it running!).

Output:
  model/<name>.joblib   -> 5 saved pipelines (preprocessing + model)
  model/metrics.csv     -> comparison table for README
  test_data.csv         -> held-out test split (features + target) for the Streamlit app
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------- Settings (change these to make the work your own) ----------
DATA_FILE = "adult.csv"   # Kaggle: uciml/adult-census-income
TARGET = "income"
RANDOM_STATE = 2026         # TODO: pick your own number
TEST_SIZE = 0.2
MODEL_DIR = "model"

# ---------------- 1. Load and clean -----------------------------------------
df = pd.read_csv(DATA_FILE, skipinitialspace=True)
df.columns = [c.strip().replace(".", "_").replace("-", "_") for c in df.columns]
df = df.replace("?", np.nan)

# Target: ">50K" -> 1, "<=50K" -> 0 (strip trailing '.' found in UCI test file)
df[TARGET] = df[TARGET].astype(str).str.strip().str.rstrip(".")
df[TARGET] = (df[TARGET] == ">50K").astype(int)

print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("Class balance:\n", df[TARGET].value_counts(normalize=True))

X = df.drop(columns=[TARGET])
y = df[TARGET]

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print("Numeric:", num_cols)
print("Categorical:", cat_cols)

# ---------------- 2. Split ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

# Save raw test splizst (features + target) for the Streamlit app
pd.concat([X_test, y_test], axis=1).to_csv("test_data.csv", index=False)

# ---------------- 3.a Preprocessing ------------------------------------------
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat_cols),
])

# ---------------- 4. Models -------------------------------------------------
# TODO: try different hyperparameters and note what changes
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0),  # AUC uses decision_function
}


def get_scores(pipe, X):
    """Score for the positive class, used for AUC."""
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return pipe.decision_function(X)


# ---------------- 5. Train, evaluate, save ----------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
rows = []
for name, clf in models.items():
    t0 = time.time()
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_score = get_scores(pipe, X_test)

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_score),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join(MODEL_DIR, fname), compress=3)
    print(f"{name}: done in {time.time() - t0:.1f}s -> {fname}")

results = pd.DataFrame(rows).round(4)
results.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
print("\n", results.to_string(index=False))
print("\nMarkdown table for README:\n")
print(results.to_markdown(index=False))