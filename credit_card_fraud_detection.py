# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load the dataset (replace 'creditcard.csv' with your dataset)
# Dataset should have features like transaction amount, time, etc., and a 'Class' column (1 for fraud, 0 for non-fraud)
data = pd.read_csv('creditcard.csv')

# Check the dataset
print(data.head())

# Separate features (X) and target (y)
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target (fraud or not)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features (important for many machine learning models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model (you can replace this with other models like Random Forest, XGBoost, etc.)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_pred_proba))

# Example of detecting fraud in a new transaction
new_transaction = np.array([[0, -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698, 0.363787, 0.090794, -0.551600, -0.617801, -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791, 0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, -0.189115, 0.133558, -0.021053, 149.62]])
new_transaction_scaled = scaler.transform(new_transaction)
prediction = model.predict(new_transaction_scaled)
prediction_proba = model.predict_proba(new_transaction_scaled)[:, 1]

print("\nNew Transaction Prediction:")
print("Fraudulent" if prediction[0] == 1 else "Not Fraudulent")
print(f"Probability of being fraudulent: {prediction_proba[0]:.4f}")
