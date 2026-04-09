import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,  confusion_matrix, precision_score, recall_score, f1_score)
import joblib

# Load the dataset
file_path = "/Users/praveenbhat/Documents/mlpro/pro/crop_yield.csv"
df = pd.read_csv(file_path)

# Check columns
print("Columns in the dataset:")
print(df.columns)

# Handle missing values
df.dropna(inplace=True)

# Identify and encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_data = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
df.drop(columns=categorical_cols, inplace=True)
df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

# Convert target Yield into classification labels
yield_median = df['Yield'].median()
df['Yield_Class'] = (df['Yield'] >= yield_median).astype(int)

# Drop original Yield column
df.drop(columns=['Yield'], inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_crop_yield_classification.csv", index=False)

# Features and target
X = df.drop(columns=['Yield_Class'])
y = df['Yield_Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machine": SVC(kernel='linear', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# K-Fold setup
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Dictionary to store model performances
model_performance = {}

# Loop over models
for name, model in models.items():
    print(f"\n==== {name} ====")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_crop_model.pkl")
    
    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Store the performance of the model
    model_performance[name] = {
        "accuracy": accuracy,
        "cv_accuracy": cv_scores.mean(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    print("Confusion Matrix:")
    print(conf_matrix)
    TN, FP, FN, TP = conf_matrix.ravel()
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Positives (TP): {TP}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # Cross-validation plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=list(range(1, 11)), y=list(cv_scores * 100), marker='o', linestyle='--', color='green' if name == "Gradient Boosting" else 'blue')
    plt.xlabel("Fold")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{name} - Cross-Validation Accuracy per Fold")
    plt.grid(True)
    plt.show()

# Sort models by test accuracy and get top 3 models
top_3_models = sorted(model_performance.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]

print("\nTop 3 Models Based on Accuracy:")
for rank, (name, performance) in enumerate(top_3_models, 1):
    print(f"{rank}. {name} - Accuracy: {performance['accuracy'] * 100:.2f}%")

# Test dummy input (optional example)
test_data = np.zeros((1, X.shape[1]))  # Replace with real values for actual prediction
test_df = pd.DataFrame(test_data, columns=X.columns)

# Load and predict with the top model
top_model_name = top_3_models[0][0]
print(f"\nTop Model: {top_model_name}")

loaded_top_model = joblib.load(f"{top_model_name.replace(' ', '_').lower()}_crop_model.pkl")
model_pred = loaded_top_model.predict(test_df)
print(f"Predicted Yield Class (0 = Low, 1 = High): {model_pred[0]}")
