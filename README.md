# Student Performance Risk Predictor

A machine learning project for predicting whether a student is at risk of low final academic performance using tabular educational data.

The goal of this project was not only to train a classification model, but to go through a complete small ML workflow: data loading, target creation, exploratory analysis, baseline comparison, preprocessing, model training, evaluation, and decision-threshold tuning.

## Project Overview

This project uses the Student Performance dataset and frames the task as a binary classification problem.

The original dataset contains students' demographic, social, school-related, and academic features. The final grade is stored in the `G3` column and ranges from 0 to 20.

In this project, a student is considered **at risk** if their final grade is below 10:

```text
G3 < 10  →  at_risk = 1
G3 >= 10 →  at_risk = 0
```

This gives a simple and interpretable target variable for a student risk prediction task.

## Problem Statement

The project answers two related questions:

1. Can we predict whether a student is at risk of low final performance if previous grades are available?
2. Can we make an earlier prediction without using previous grades `G1` and `G2`?

This distinction is important because previous grades are highly predictive of the final grade. A model that uses `G1` and `G2` may perform very well, but it is less useful as a strict early-warning system.

## Dataset

Dataset used: **Student Performance Data Set**

The dataset contains information about students, including:

- demographic attributes;
- family and social background;
- study time;
- number of past class failures;
- absences;
- school support;
- family support;
- internet access;
- previous grades `G1` and `G2`;
- final grade `G3`.

The target variable `at_risk` was created from `G3`.

## Dataset Summary

After loading the `student-mat.csv` file:

```text
Rows: 395
Columns before target creation: 33
Columns after target creation: 34
Missing values: 0
```

Target distribution:

```text
not at risk: approximately 67%
at risk:     approximately 33%
```

The classes are moderately imbalanced, so accuracy alone is not enough to evaluate the models. For this reason, recall and F1-score for the `at_risk` class are especially important.

## Exploratory Data Analysis

A basic EDA showed clear differences between students in the at-risk and not-at-risk groups.

Students in the at-risk group had:

- lower previous grades `G1` and `G2`;
- lower final grade `G3` by definition;
- more previous class failures;
- slightly higher absences;
- slightly lower study time.

Average values by risk group:

| at_risk | studytime | failures | absences | G1 | G2 | G3 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.08 | 0.16 | 5.19 | 12.45 | 12.62 | 12.88 |
| 1 | 1.95 | 0.69 | 6.76 | 7.76 | 6.82 | 5.38 |

Internet access also showed a small difference:

```text
internet = no  → at-risk rate ≈ 39.4%
internet = yes → at-risk rate ≈ 31.6%
```

## Methodology

The project compares several approaches.

### 1. Majority-Class Baseline

The baseline model always predicts the majority class:

```text
at_risk = 0
```

This gives around 67% accuracy, but it completely fails to identify students in the at-risk group.

### 2. Logistic Regression with Previous Grades

This version removes `G3` and the target variable, but keeps `G1` and `G2`.

This answers the question:

> If we already know the first and second period grades, can we predict the risk of low final performance?

### 3. Early Prediction Without Previous Grades

This stricter version removes:

```text
G1, G2, G3, at_risk
```

This answers the question:

> Can we predict risk earlier, without using previous academic results?

For this setting, the following models were tested:

- Logistic Regression;
- Random Forest;
- Gradient Boosting.

### 4. Threshold Tuning

For Logistic Regression in the early-prediction setting, the decision threshold was adjusted from `0.5` to `0.4`.

The goal was to improve recall for the at-risk class. In an educational early-warning system, missing students who may need support can be more costly than flagging extra students for review.

## Preprocessing

The dataset contains both numerical and categorical features.

The preprocessing pipeline used:

- `StandardScaler` for numerical features;
- `OneHotEncoder` for categorical features;
- `ColumnTransformer` to apply different preprocessing steps to different column types;
- `Pipeline` to combine preprocessing and model training.

The dataset was split into training and test sets using stratified splitting:

```text
test_size = 0.2
random_state = 42
stratify = y
```

Stratification was used to preserve the class distribution in both the training and test sets.

## Results

| Model | Features | Threshold | Accuracy | At-risk Recall | At-risk F1 |
|---|---|---:|---:|---:|---:|
| Majority baseline | always predicts 0 | — | 0.67 | 0.00 | 0.00 |
| Logistic Regression | with `G1`, `G2` | 0.5 | 0.94 | 0.85 | 0.90 |
| Logistic Regression | without `G1`, `G2` | 0.5 | 0.70 | 0.31 | 0.40 |
| Random Forest | without `G1`, `G2` | 0.5 | 0.68 | 0.19 | 0.29 |
| Gradient Boosting | without `G1`, `G2` | 0.5 | 0.66 | 0.27 | 0.34 |
| Logistic Regression | without `G1`, `G2` | 0.4 | 0.72 | 0.62 | 0.59 |

## Key Findings

The model using previous grades `G1` and `G2` achieved the best overall performance:

```text
Accuracy:       0.94
At-risk recall: 0.85
At-risk F1:     0.90
```

This confirms that previous academic performance is highly predictive of final performance risk.

However, when `G1` and `G2` were removed, the task became significantly harder. The standard Logistic Regression model without previous grades achieved:

```text
Accuracy:       0.70
At-risk recall: 0.31
At-risk F1:     0.40
```

This means that without previous grades, the model had much less predictive signal and missed many at-risk students.

Threshold tuning improved the early-warning version. Lowering the Logistic Regression threshold from `0.5` to `0.4` increased at-risk recall:

```text
Recall before threshold tuning: 0.31
Recall after threshold tuning:  0.62
```

The F1-score for the at-risk class also improved:

```text
F1 before threshold tuning: 0.40
F1 after threshold tuning:  0.59
```

The trade-off was an increase in false positives, but this may be acceptable in an early-warning educational context.

## Confusion Matrix: Logistic Regression Without G1/G2, Threshold 0.4

```text
[[41 12]
 [10 16]]
```

Interpretation:

- 41 students were correctly classified as not at risk;
- 12 students were incorrectly flagged as at risk;
- 10 at-risk students were missed;
- 16 at-risk students were correctly identified.

Compared with the default threshold, the model found more at-risk students, which is useful for a support-oriented system.

## Final Interpretation

The strongest model was Logistic Regression with previous grades `G1` and `G2`, but this version depends heavily on existing academic performance data.

The stricter early-warning version without `G1` and `G2` performed worse, which shows that predicting student risk before previous grades are available is a much harder task.

Among the early-prediction models, Logistic Regression with a lowered threshold gave the most useful balance for identifying at-risk students. It improved recall while keeping the model simple and interpretable.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- KaggleHub
- Logistic Regression
- Random Forest
- Gradient Boosting
- ColumnTransformer
- Pipeline
- OneHotEncoder
- StandardScaler

## Project Structure

Recommended repository structure:

```text
student-performance-risk-predictor/
│
├── student_risk_analysis.py
├── README.md
├── requirements.txt
└── notebook/
    └── student_risk_analysis.ipynb
```

## How to Run

Install dependencies:

```bash
pip install pandas numpy scikit-learn kagglehub[pandas-datasets]
```

Run the script:

```bash
python student_risk_analysis.py
```

Alternatively, run the notebook in Google Colab.

## Possible Future Improvements

Possible next steps:

- save the best model with `joblib`;
- create a FastAPI inference endpoint;
- add a small demo API request for student risk prediction;
- test cross-validation instead of a single train-test split;
- tune model hyperparameters;
- compare results on the Portuguese language dataset as well;
- add model interpretation, such as feature importance or SHAP values;
- build a simple dashboard for risk probability visualization.

## Portfolio Note

This project demonstrates a complete small ML workflow:

- framing a real-world inspired problem;
- creating a binary target variable;
- checking class balance;
- building a majority-class baseline;
- preprocessing numerical and categorical data;
- training and comparing several models;
- evaluating results with classification metrics;
- improving recall through threshold tuning;
- interpreting model limitations honestly.

The main value of the project is not just the final accuracy, but the comparison between a strong model using previous grades and a stricter early-warning model without previous grades.
