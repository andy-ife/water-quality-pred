import pandas as pd
import numpy as np
import catboost as cat
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    average_precision_score, confusion_matrix, log_loss, cohen_kappa_score
)
import os

# Load datasets
train_df = pd.read_csv('datasets/train.csv', index_col=0)
test_df = pd.read_csv('datasets/test.csv', index_col=0)

X_train = train_df.drop(columns=['EVENT']) # Features only
y_train = train_df['EVENT'] # Labels only
X_test = test_df.drop(columns=['EVENT'])
y_test = test_df['EVENT']

model = cat.CatBoostClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilities for positive class only

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("ROC-AUC: ", roc_auc_score(y_test, y_pred_proba))
print("PR-AUC (Average Precision): ", average_precision_score(y_test, y_pred_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Log Loss: ", log_loss(y_test, y_pred_proba))
print("Cohen's Kappa: ", cohen_kappa_score(y_test, y_pred))
