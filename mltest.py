import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score)
import joblib
import os

# File paths
train_path = "/Users/praveenbhat/Documents/mlpro/pro/crop_yield.csv"
test_path = "/Users/praveenbhat/Documents/mlpro/pro/test.csv"  # Replace if needed

# Load and clean training data
df = pd.read_csv(train_path)
df.dropna(inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded_train = encoder.fit_transform(df[categorical_cols])
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols))

# Merge and convert target
df.drop(columns=categorical_cols, inplace=True)
df = pd.concat([df.reset_index(drop=True), encoded_train_df], axis=1)
yield_median = df['Yield'].median()
df['Yield_Class'] = (df['Yield'] >= yield_median).astype(int)
df.drop(columns=['Yield'], inplace=True)
df.to_csv("cleaned_crop_yield_classification.csv", index=False)

# Train-validation split
X = df.drop(columns=['Yield_Class'])
y = df['Yield_Class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Train, evaluate, save
for name, model in models.items():
    print(f"\n==== {name} ====")
    
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_crop_model.pkl")
    
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    conf_matrix = confusion_matrix(y_val, y_pred)

    print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    
    print("Confusion Matrix:")
    print(conf_matrix)
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # Cross-validation fold plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=list(range(1, 11)), y=list(cv_scores * 100), marker='o', linestyle='--')
    plt.xlabel("Fold")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{name} - Cross-Validation Accuracy per Fold")
    plt.grid(True)
    plt.show()

# === EVALUATE ON SEPARATE TEST SET ===
if os.path.exists(test_path):
    test_df = pd.read_csv(test_path)
    print("\nLoaded separate test dataset.")
    test_df.dropna(inplace=True)

    # Encode like training set
    encoded_test = encoder.transform(test_df[categorical_cols])
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))
    test_df.drop(columns=categorical_cols, inplace=True)
    test_df = pd.concat([test_df.reset_index(drop=True), encoded_test_df], axis=1)

    # Extract target if available
    if 'Yield' in test_df.columns:
        y_actual = (test_df['Yield'] >= yield_median).astype(int)
        test_df.drop(columns=['Yield'], inplace=True)
    else:
        y_actual = None

    # Predict using all models
    for name in models:
        print(f"\nEvaluating on Separate Test Set - {name}")
        model_file = f"{name.replace(' ', '_').lower()}_crop_model.pkl"
        loaded_model = joblib.load(model_file)
        y_test_pred = loaded_model.predict(test_df)

        if y_actual is not None:
            acc = accuracy_score(y_actual, y_test_pred)
            print(f"Separate Test Accuracy: {acc * 100:.2f}%")
            print("Classification Report:\n", classification_report(y_actual, y_test_pred))
        else:
            print("Predictions (Yield Class):", y_test_pred)
else:
    print("\n[!] Separate test dataset not found at:", test_path)
