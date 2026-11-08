# Adult Income Classification – ML Assignment 2

- **Student ID:** 2026AJ05528
- **GitHub Repository:** https://github.com/ashishtyagi0809/2026AJ05528-ML-Project
- **Live Streamlit App:** https://2026aj05528-ml-project-ey78liuhpxipfw59mcdena.streamlit.app/

---

## a. Problem Statement

The goal is to predict whether a person's annual income is **more than $50K** or **$50K or less**, using demographic and employment information from the US Census. This is a **binary classification** problem. Five classification models are trained on the same data and compared using six evaluation metrics. The best model is chosen based on these results, and all models are demonstrated through an interactive Streamlit web app.

## b. Dataset Description

| Item           | Details                                                              |
| -------------- | -------------------------------------------------------------------- |
| Name           | Adult (Census Income)                                                |
| Source         | UCI Machine Learning Repository / Kaggle (uciml/adult-census-income) |
| Instances      | 48,842                                                               |
| Features       | 14 (6 numeric, 8 categorical)                                        |
| Target         | `income` → `>50K` (1) or `<=50K` (0)                                 |
| Class balance  | 76.1% `<=50K`, 23.9% `>50K` (imbalanced)                             |
| Missing values | `workclass`, `occupation`, `native_country` (marked as `?`)          |

**Features**

| Type        | Features                                                                                  |
| ----------- | ----------------------------------------------------------------------------------------- |
| Numeric     | age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week                    |
| Categorical | workclass, education, marital_status, occupation, relationship, race, sex, native_country |

**Preprocessing (done inside each saved pipeline)**

- `?` values replaced with missing values.
- Numeric features: missing values filled with the median, then scaled with StandardScaler.
- Categorical features: missing values filled with the most frequent value, then one-hot encoded.
- Train/test split: 80/20, stratified on the target, `random_state=2026`. The test set has 9,769 rows and is saved as `test_data.csv`.

## c. GitHub Repository Link

https://github.com/ashishtyagi0809/2026AJ05528-ML-Project

```
2026AJ05528-ML-Project/
├── app.py                  # Streamlit app (loads saved models, no retraining)
├── train_models.py         # Training + evaluation script (run on BITS Virtual Lab)
├── requirements.txt
├── README.md
├── adult.csv               # Full dataset
├── test_data.csv           # 20% held-out test split (features + target)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics.csv
```

## d. Models Used

Models implemented (all on the same dataset and the same train/test split):

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine

Evaluation metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC).

All five models are trained on the same training split. Each model is saved as a scikit-learn Pipeline that includes the preprocessing steps.

| Model                  | Settings          |
| ---------------------- | ----------------- |
| Logistic Regression    | max_iter=1000     |
| Decision Tree          | max_depth=10      |
| KNN                    | n_neighbors=7     |
| Naive Bayes            | GaussianNB        |
| Support Vector Machine | RBF kernel, C=1.0 |

### Comparison Table (evaluation metrics on test set, 9,769 rows)

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ---------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression    | 0.8538   | 0.9062 | 0.7395    | 0.6009 | 0.6630 | 0.5760 |
| Decision Tree          | 0.8602   | 0.9017 | 0.7768    | 0.5834 | 0.6663 | 0.5897 |
| KNN                    | 0.8379   | 0.8764 | 0.6839    | 0.5997 | 0.6390 | 0.5370 |
| Naive Bayes            | 0.5697   | 0.7987 | 0.3511    | 0.9410 | 0.5114 | 0.3504 |
| Support Vector Machine | 0.8571   | 0.9019 | 0.7582    | 0.5915 | 0.6646 | 0.5827 |

### Observations on the performance of each model

<!-- WRITE THESE IN YOUR OWN WORDS (3 marks). 2-3 sentences each. -->

| ML Model Name                         | Observation about model performance |
| ------------------------------------- | ----------------------------------- |
| Logistic Regression                   | TODO                                |
| Decision Tree                         | TODO                                |
| KNN                                   | TODO                                |
| Naive Bayes                           | TODO                                |
| Support Vector Machine                | TODO                                |
| **Overall, Winner for your dataset?** | TODO                                |

## How to Run Locally

```bash
pip install -r requirements.txt
python train_models.py      # optional: retrains and saves models to model/
streamlit run app.py
```

Upload `test_data.csv` in the app sidebar, then choose a model to see its metrics, confusion matrix and classification report.
