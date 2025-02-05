
---

# Credit Card Fraud Detection

This project demonstrates how to build a machine learning model to detect fraudulent credit card transactions using Python. The model is trained on a dataset of anonymized credit card transactions and uses Logistic Regression for classification.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Setup and Installation](#setup-and-installation)
5. [Code Explanation](#code-explanation)
6. [Results](#results)
7. [Future Enhancements](#future-enhancements)
8. [License](#license)

---

## Project Overview
The goal of this project is to detect fraudulent credit card transactions using machine learning. The model is trained on a dataset containing features of credit card transactions and a binary label indicating whether the transaction is fraudulent (1) or not (0). The model is evaluated using metrics like confusion matrix, classification report, and ROC-AUC score.

---

## Dataset
The dataset used in this project is the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains anonymized credit card transactions with the following structure:
- Features: `V1`, `V2`, ..., `V28` (anonymized features obtained through PCA)
- `Time`: Time of the transaction
- `Amount`: Transaction amount
- `Class`: Binary label (1 for fraud, 0 for non-fraud)

---

## Requirements
To run this project, you need the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`

You can install the required libraries using the following command:
```bash
pip install pandas numpy scikit-learn
```

---

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project directory as `creditcard.csv`.

3. Run the Python script:
   ```bash
   python fraud_detection.py
   ```

---

## Code Explanation
The code is structured as follows:
1. **Data Loading**: The dataset is loaded using `pandas`.
2. **Data Preprocessing**:
   - Features and target are separated.
   - The dataset is split into training and testing sets.
   - Features are standardized using `StandardScaler`.
3. **Model Training**: A Logistic Regression model is trained on the training data.
4. **Model Evaluation**: The model is evaluated using a confusion matrix, classification report, and ROC-AUC score.
5. **Prediction**: The model predicts whether a new transaction is fraudulent or not.

---

## Results
The model's performance is evaluated using the following metrics:
- **Confusion Matrix**: Shows the number of true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Provides precision, recall, F1-score, and support for each class.
- **ROC-AUC Score**: Measures the model's ability to distinguish between fraudulent and non-fraudulent transactions.

Example output:
```
Confusion Matrix:
[[56862     5]
 [   23    52]]

Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00     56867
           1       0.91      0.69      0.79        75
    accuracy                           1.00     56942
   macro avg       0.96      0.85      0.89     56942
weighted avg       1.00      1.00      1.00     56942

ROC-AUC Score:
0.978
```

---

## Future Enhancements
- Use more advanced models like Random Forest, XGBoost, or Neural Networks.
- Address class imbalance using techniques like SMOTE or class weighting.
- Perform feature engineering to improve model performance.
- Deploy the model as a web application or API for real-time fraud detection.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

