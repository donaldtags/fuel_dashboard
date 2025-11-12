# Credit Card Fraud Detection using Supervised Learning


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings

warnings.filterwarnings('ignore')

# Step 1: Load and Explore the Dataset

print("Step 1: Loading Credit Card Fraud Dataset")

print("=" * 50)

# Download dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Or use this synthetic version for practice

np.random.seed(42)

# Create synthetic credit card transaction data (1000+ rows, 5+ columns)

n_samples = 2000

data = {

    'transaction_amount': np.random.exponential(100, n_samples),

    'account_age_days': np.random.randint(30, 3650, n_samples),

    'num_transactions_today': np.random.poisson(5, n_samples),

    'avg_transaction_amount': np.random.exponential(80, n_samples),

    'time_since_last_transaction': np.random.exponential(24, n_samples),

    'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'online', 'retail'], n_samples),

    'weekend_transaction': np.random.choice([0, 1], n_samples)

}

# Create fraud labels (0 = legitimate, 1 = fraud)

# Make fraud more likely for certain patterns

fraud_probability = (

        0.01 +  # Base fraud rate

        0.1 * (data['transaction_amount'] > 500) +  # High amounts more likely fraud

        0.05 * (data['account_age_days'] < 90) +  # New accounts more risky

        0.03 * (data['num_transactions_today'] > 10)  # Many transactions suspicious

)

data['is_fraud'] = np.random.binomial(1, fraud_probability, n_samples)

# Convert to DataFrame

df = pd.DataFrame(data)

# Convert categorical variables to numeric

df['merchant_category_encoded'] = pd.Categorical(df['merchant_category']).codes

print(f"Dataset shape: {df.shape}")

print(f"Columns: {list(df.columns)}")

print("\nFirst 5 rows:")

print(df.head())

# Step 2: Data Analysis and Visualization

print("\n\nStep 2: Data Analysis")

print("=" * 50)

# Basic statistics

print("Dataset Info:")

print(df.info())

print("\nFraud Distribution:")

fraud_counts = df['is_fraud'].value_counts()

print(f"Legitimate transactions: {fraud_counts[0]} ({fraud_counts[0] / len(df) * 100:.1f}%)")

print(f"Fraudulent transactions: {fraud_counts[1]} ({fraud_counts[1] / len(df) * 100:.1f}%)")

# Create visualizations

plt.figure(figsize=(15, 10))

# Plot 1: Fraud distribution

plt.subplot(2, 3, 1)

df['is_fraud'].value_counts().plot(kind='bar')

plt.title('Fraud vs Legitimate Transactions')

plt.xlabel('Transaction Type (0=Legitimate, 1=Fraud)')

plt.ylabel('Count')

# Plot 2: Transaction amount distribution

plt.subplot(2, 3, 2)

plt.hist(df[df['is_fraud'] == 0]['transaction_amount'], alpha=0.7, label='Legitimate', bins=50)

plt.hist(df[df['is_fraud'] == 1]['transaction_amount'], alpha=0.7, label='Fraud', bins=50)

plt.title('Transaction Amount Distribution')

plt.xlabel('Amount')

plt.ylabel('Frequency')

plt.legend()

# Plot 3: Account age vs fraud

plt.subplot(2, 3, 3)

plt.boxplot([df[df['is_fraud'] == 0]['account_age_days'],

             df[df['is_fraud'] == 1]['account_age_days']])

plt.title('Account Age by Transaction Type')

plt.xlabel('Transaction Type')

plt.ylabel('Account Age (days)')

plt.xticks([1, 2], ['Legitimate', 'Fraud'])

# Plot 4: Correlation heatmap

plt.subplot(2, 3, 4)

correlation_cols = ['transaction_amount', 'account_age_days', 'num_transactions_today',

                    'avg_transaction_amount', 'time_since_last_transaction', 'is_fraud']

correlation_matrix = df[correlation_cols].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)

plt.title('Feature Correlation Matrix')

# Plot 5: Transactions by merchant category

plt.subplot(2, 3, 5)

fraud_by_merchant = df.groupby('merchant_category')['is_fraud'].mean()

fraud_by_merchant.plot(kind='bar')

plt.title('Fraud Rate by Merchant Category')

plt.xlabel('Merchant Category')

plt.ylabel('Fraud Rate')

plt.xticks(rotation=45)

# Plot 6: Weekend vs weekday transactions

plt.subplot(2, 3, 6)

weekend_fraud = df.groupby('weekend_transaction')['is_fraud'].mean()

weekend_fraud.plot(kind='bar')

plt.title('Fraud Rate: Weekend vs Weekday')

plt.xlabel('Weekend Transaction (0=Weekday, 1=Weekend)')

plt.ylabel('Fraud Rate')

plt.tight_layout()

plt.show()

# Step 3: Data Preprocessing

print("\n\nStep 3: Data Preprocessing")

print("=" * 50)

# Select features for machine learning

feature_columns = ['transaction_amount', 'account_age_days', 'num_transactions_today',

                   'avg_transaction_amount', 'time_since_last_transaction',

                   'merchant_category_encoded', 'weekend_transaction']

X = df[feature_columns]

y = df['is_fraud']

print(f"Features shape: {X.shape}")

print(f"Target shape: {y.shape}")

# Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")

print(f"Test set size: {X_test.shape[0]}")

# Scale the features for better performance

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully")

# Step 4: Machine Learning Model Implementation

print("\n\nStep 4: Machine Learning Models")

print("=" * 50)

# Model 1: Logistic Regression

print("Training Logistic Regression...")

lr_model = LogisticRegression(random_state=42)

lr_model.fit(X_train_scaled, y_train)

# Make predictions

lr_predictions = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Model 2: Random Forest

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)  # Random Forest doesn't need scaling

# Make predictions

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Step 5: Model Evaluation

print("\n\nStep 5: Model Evaluation")

print("=" * 50)

# Compare models

print("Model Comparison:")

print(f"Logistic Regression: {lr_accuracy:.4f}")

print(f"Random Forest: {rf_accuracy:.4f}")

# Detailed evaluation for the better model

if rf_accuracy >= lr_accuracy:

    best_model = rf_model

    best_predictions = rf_predictions

    model_name = "Random Forest"

else:

    best_model = lr_model

    best_predictions = lr_predictions

    model_name = "Logistic Regression"

print(f"\nBest Model: {model_name}")

print("\nDetailed Classification Report:")

print(classification_report(y_test, best_predictions))

# Confusion Matrix

cm = confusion_matrix(y_test, best_predictions)

print("\nConfusion Matrix:")

print(cm)

# Visualize confusion matrix

plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',

            xticklabels=['Legitimate', 'Fraud'],

            yticklabels=['Legitimate', 'Fraud'])

plt.title(f'{model_name} - Confusion Matrix')

plt.ylabel('True Label')

plt.xlabel('Predicted Label')

plt.show()

# Feature importance (for Random Forest)

if model_name == "Random Forest":
    feature_importance = pd.DataFrame({

        'feature': feature_columns,

        'importance': rf_model.feature_importances_

    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")

    print(feature_importance)

    # Plot feature importance

    plt.figure(figsize=(10, 6))

    plt.barh(feature_importance['feature'], feature_importance['importance'])

    plt.title('Feature Importance in Fraud Detection')

    plt.xlabel('Importance Score')

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.show()

# Step 6: Making New Predictions

print("\n\nStep 6: Making New Predictions")

print("=" * 50)

# Example new transactions to predict

new_transactions = pd.DataFrame({

    'transaction_amount': [25.50, 1200.00, 45.75],

    'account_age_days': [1200, 45, 800],

    'num_transactions_today': [2, 15, 3],

    'avg_transaction_amount': [30.00, 200.00, 50.00],

    'time_since_last_transaction': [12.5, 2.0, 24.0],

    'merchant_category_encoded': [0, 3, 1],  # grocery, online, gas

    'weekend_transaction': [0, 1, 0]

})

print("New transactions to predict:")

print(new_transactions)

if model_name == "Random Forest":

    new_predictions = rf_model.predict(new_transactions)

    prediction_proba = rf_model.predict_proba(new_transactions)

else:

    new_transactions_scaled = scaler.transform(new_transactions)

    new_predictions = lr_model.predict(new_transactions_scaled)

    prediction_proba = lr_model.predict_proba(new_transactions_scaled)

print("\nPredictions:")

for i, (pred, proba) in enumerate(zip(new_predictions, prediction_proba)):
    status = "FRAUD" if pred == 1 else "LEGITIMATE"

    confidence = max(proba) * 100

    print(f"Transaction {i + 1}: {status} (Confidence: {confidence:.1f}%)")

# Step 7: Save the Model and Results

print("\n\nStep 7: Saving Results")

print("=" * 50)

# Save the trained model

import joblib

joblib.dump(best_model, 'fraud_detection_model.pkl')

if model_name == "Logistic Regression":
    joblib.dump(scaler, 'scaler.pkl')

print(f"Model saved as 'fraud_detection_model.pkl'")

if model_name == "Logistic Regression":
    print("Scaler saved as 'scaler.pkl'")

# Save the analysis results

results_summary = {

    'Model Used': model_name,

    'Accuracy': f"{max(lr_accuracy, rf_accuracy):.4f}",

    'Dataset Size': len(df),

    'Training Samples': len(X_train),

    'Test Samples': len(X_test),

    'Fraud Rate': f"{df['is_fraud'].mean():.2%}",

    'Features Used': feature_columns

}

# Save to file

with open('fraud_detection_results.txt', 'w') as f:
    f.write("Credit Card Fraud Detection - Results Summary\n")

    f.write("=" * 50 + "\n\n")

    for key, value in results_summary.items():
        f.write(f"{key}: {value}\n")

    f.write("\nClassification Report:\n")

    f.write(classification_report(y_test, best_predictions))

print("Results saved to 'fraud_detection_results.txt'")

print("\nProject completed successfully!")

# Instructions for sharing and running the code

print("\n" + "=" * 60)

print("HOW TO SAVE AND SHARE THIS CODE:")

print("=" * 60)

print("1. Copy this entire code into a file named 'fraud_detection.py'")

print("2. Install required packages: pip install pandas numpy matplotlib seaborn scikit-learn")

print("3. Run the code: python fraud_detection.py")

print("4. Share the .py file along with generated files:")

print("   - fraud_detection_model.pkl (trained model)")

print("   - fraud_detection_results.txt (results summary)")

print("   - scaler.pkl (if using Logistic Regression)")

print("\nTo use the saved model later:")

print("model = joblib.load('fraud_detection_model.pkl')")

print("predictions = model.predict(new_data)")