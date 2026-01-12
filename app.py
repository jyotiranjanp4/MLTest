import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import model functions
from models import logistic_regression, decision_tree, knn, gaussian_nb, multinomial_nb, random_forest, xgboost_model

st.set_page_config(page_title="ML Model Evaluation", layout="wide")
st.title("🤖 ML Model Evaluation Dashboard")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    target_column = st.selectbox("Select Target Column", df.columns)

    test_size = st.number_input("Test Size", 0.05, 0.95, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 1000, 42, 1)

    if target_column:
        y = df[target_column]
        class_counts = y.value_counts()

        if class_counts.min() < 2:
            st.error("❌ Stratified split not possible: Some classes have fewer than 2 samples.")
        else:
            X = pd.get_dummies(df.drop(columns=[target_column]))
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            selected_models = st.multiselect(
                "Select Models",
                ["Logistic Regression", "Decision Tree Classifier", "K-Nearest Neighbor Classifier",
                 "Naive Bayes - Gaussian", "Naive Bayes - Multinomial",
                 "Random Forest", "XGBoost"]
            )

            start_button = st.button("🚀 Start Training & Evaluation")

            if selected_models and start_button:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
                )

                for model_name in selected_models:
                    st.subheader(f"🔹 {model_name}")
                    if model_name == "Logistic Regression":
                        results = logistic_regression.train_evaluate(X_train, X_test, y_train, y_test, le)
                    elif model_name == "Decision Tree Classifier":
                        results = decision_tree.train_evaluate(X_train, X_test, y_train, y_test, le)
                    elif model_name == "K-Nearest Neighbor Classifier":
                        results = knn.train_evaluate(X_train, X_test, y_train, y_test, le)
                    elif model_name == "Naive Bayes - Gaussian":
                        results = gaussian_nb.train_evaluate(X_train, X_test, y_train, y_test, le)
                    elif model_name == "Naive Bayes - Multinomial":
                        results = multinomial_nb.train_evaluate(X_train, X_test, y_train, y_test, le)
                    elif model_name == "Random Forest":
                        results = random_forest.train_evaluate(X_train, X_test, y_train, y_test, le)
                    elif model_name == "XGBoost":
                        results = xgboost_model.train_evaluate(X_train, X_test, y_train, y_test, le)

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                    col2.metric("AUC", f"{results['auc']:.4f}" if results['auc'] else "N/A")
                    col3.metric("MCC", f"{results['mcc']:.4f}")

                    col4, col5, col6 = st.columns(3)
                    col4.metric("Precision", f"{results['precision']:.4f}")
                    col5.metric("Recall", f"{results['recall']:.4f}")
                    col6.metric("F1 Score", f"{results['f1']:.4f}")

                    st.write("Confusion Matrix")
                    st.dataframe(results['confusion_matrix'])

                    st.write("Classification Report")
                    st.dataframe(results['classification_report'])

                    st.divider()
