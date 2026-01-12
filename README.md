# Exam Score Prediction Dashboard

**Name: ** Jyotiranjan Pattanaik
**BITS ID: ** 2024DC04018

## a. Problem Statement

The objective of this project is to predict whether a student will **pass or fail** an exam based on their study habits, attendance, and other related features. This is framed as a **binary classification problem** where:

- `Pass` (1) → exam score ≥ 50  
- `Fail` (0) → exam score < 50  

We aim to implement multiple machine learning models, compare their performance using key evaluation metrics, and visualize the results in a web-based dashboard.

---

## b. Dataset Description

The dataset contains information about students’ demographics, study patterns, and exam-related factors. Key columns include:

| Column Name        | Description |
|-------------------|-------------|
| student_id         | Unique identifier for each student |
| age                | Age of the student |
| gender             | Gender of the student |
| course             | Course enrolled |
| study_hours        | Number of hours spent studying |
| class_attendance   | Percentage of classes attended |
| internet_access    | Whether the student has internet access at home |
| sleep_hours        | Average sleep hours per day |
| sleep_quality      | Quality of sleep (subjective rating) |
| study_method       | Study method used by student |
| facility_rating    | Rating of school/facility resources |
| exam_difficulty    | Student-perceived exam difficulty |
| exam_score         | Actual score obtained in exam (target) |

The target variable `Result` is created as a **binary classification label**: Pass (1) or Fail (0).

---

## c. Models Used

We implemented **six machine learning models** on the same dataset. For each model, the following evaluation metrics were calculated:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

### Comparison Table of Evaluation Metrics

| ML Model Name        | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
|--------------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.0000   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 |
| Decision Tree       | 0.0000   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 |
| kNN                 | 0.0000   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 |
| Naive Bayes         | 0.0000   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 |
| Random Forest       | 0.0000   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 |
| XGBoost             | 0.0000   | 0.0000 | 0.0000    | 0.0000 | 0.0000 | 0.0000 |

> ⚠️ Note: Replace the `0.0000` placeholders with actual results obtained after running the Streamlit app.

---

### Observations on Model Performance

| ML Model Name        | Observation about Model Performance |
|--------------------|------------------------------------|
| Logistic Regression | Performs reasonably well for linearly separable features but may struggle with complex non-linear interactions. |
| Decision Tree       | Can capture non-linear relationships, but prone to overfitting without pruning. |
| kNN                 | Simple and effective for small datasets, but sensitive to feature scaling and outliers. |
| Naive Bayes         | Performs well with categorical features; assumes feature independence, which may not always hold. |
| Random Forest       | Ensemble model that reduces overfitting and improves generalization; typically provides strong performance. |
| XGBoost             | Gradient boosting ensemble; usually achieves the highest accuracy and AUC by handling complex feature interactions. |

> ⚠️ Replace/adjust observations based on **actual results from your trained models**.

---

### How to Run the Web App

1. Install dependencies:

```bash
pip install -r requirements.txt
