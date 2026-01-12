from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_auc_score
)
import numpy as np
import pandas as pd

def train_evaluate(X_train, X_test, y_train, y_test, label_encoder):
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # AUC Score
    y_prob = model.predict_proba(X_test_scaled)
    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr" if len(np.unique(y_train)) > 2 else "raise"
    )

    # Metrics
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "auc": auc,
        "confusion_matrix": pd.DataFrame(confusion_matrix(y_test, y_pred), 
                                         index=label_encoder.classes_, 
                                         columns=label_encoder.classes_),
        "classification_report": pd.DataFrame(classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
        )).transpose()
    }
    return results
