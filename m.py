import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVC  # Only import SVM classifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score)
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the dataset
file_path = "/Users/praveenbhat/Documents/mlpro/pro/crop_yield.csv"
df = pd.read_csv(file_path)

# Handle missing values
df.dropna(inplace=True)

# Identify and encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Initialize OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Fit encoder and transform categorical columns
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
df.to_csv("cleaned_crop_yield_classification_svm.csv", index=False)

# Features and target
X = df.drop(columns=['Yield_Class'])
y = df['Yield_Class']

# Store feature names in a variable
feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)

# Create kfold for cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Training and evaluation
print("\n==== Support Vector Machine (SVM) ====")

# Cross-validation
cv_scores = cross_val_score(svm_model, X, y, cv=kfold, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

# Train model on full training data
svm_model.fit(X_train, y_train)

# Evaluate on test data
y_pred = svm_model.predict(X_test)

# Calculate metrics
print(f"Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1 Score: {f1_score(y_test, y_pred) * 100:.2f}%")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()
print("Confusion Matrix:")
print(conf_matrix)
print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization - Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM - Confusion Matrix")
plt.show()

# Visualization - Cross-Validation Accuracy per Fold
plt.figure(figsize=(10, 6))
sns.lineplot(x=list(range(1, 11)), y=list(cv_scores * 100), marker='o', linestyle='--', color='purple')
plt.xlabel("Fold")
plt.ylabel("Accuracy (%)")
plt.title("SVM - Cross-Validation Accuracy per Fold")
plt.grid(True)
plt.show()

# Prediction functions
def preprocess_input_data(features, encoder, feature_names):
    """Preprocess input data for prediction"""
    categorical_data = features[list(encoder.feature_names_in_)]
    numerical_data = features.drop(columns=list(encoder.feature_names_in_))

    encoded_data = encoder.transform(categorical_data)

    all_feature_names = list(numerical_data.columns) + list(encoder.get_feature_names_out(encoder.feature_names_in_))
    processed_df = pd.DataFrame(np.hstack([numerical_data.values, encoded_data]), columns=all_feature_names)

    # Reorder columns to match training
    processed_df = processed_df.reindex(columns=feature_names, fill_value=0)

    print(f"Processed input data shape: {processed_df.shape}")
    return processed_df

def prediction(input_data):
    """
    Make a prediction using the SVM model.
    
    Parameters:
    input_data (dict): Dictionary containing input features
    
    Returns:
    str: "High" or "Low" yield prediction
    """
    # Create DataFrame from input data
    features = pd.DataFrame({
        'Crop_Year': [input_data['Crop_Year']],
        'Annual_Rainfall': [input_data['Annual_Rainfall']],
        'Pesticide': [input_data['Pesticide']],
        'Fertilizer': [input_data['Fertilizer']],
        'Production': [input_data['Production']],
        'Area': [input_data['Area']],
        'Crop': [input_data['Crop']],
        'Season': [input_data['Season']],
        'State': [input_data['State']]
    })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformed_features = preprocess_input_data(features, encoder, feature_names)
        predicted_class = svm_model.predict(transformed_features)

    return "High" if predicted_class[0] == 1 else "Low"

def make_batch_predictions(input_array):
    """
    Make predictions for multiple inputs.
    
    Parameters:
    input_array (list): List of dictionaries, each containing input features
    
    Returns:
    list: List of predictions ("High" or "Low")
    """
    results = []
    for input_data in input_array:
        try:
            result = prediction(input_data)
            results.append({
                'input': input_data,
                'prediction': result
            })
        except Exception as e:
            results.append({
                'input': input_data,
                'error': str(e)
            })
    return results

# Example usage with array input
if __name__ == "__main__":
    # Example array of inputs
    input_array = [
        {
            'Crop_Year': 2018,
            'Annual_Rainfall': 1200.5,
            'Pesticide': 100.0,
            'Fertilizer': 200.0,
            'Production': 1500.0,
            'Area': 300.0,
            'Crop': 'Rice',
            'Season': 'Kharif',
            'State': 'Karnataka'
        },
        {
            'Crop_Year': 2019,
            'Annual_Rainfall': 800.0,
            'Pesticide': 80.0,
            'Fertilizer': 150.0,
            'Production': 1200.0,
            'Area': 250.0,
            'Crop': 'Wheat',
            'Season': 'Rabi',
            'State': 'Punjab'
        },
        {
            'Crop_Year': 2020,
            'Annual_Rainfall': 950.0,
            'Pesticide': 90.0,
            'Fertilizer': 180.0,
            'Production': 1300.0,
            'Area': 280.0,
            'Crop': 'Maize',
            'Season': 'Zaid',
            'State': 'Maharashtra'
        }
    ]
    
    # Make batch predictions
    print("\nBatch predictions using SVM model:")
    results = make_batch_predictions(input_array)
    
    # Print results
    for i, result in enumerate(results):
        if 'prediction' in result:
            input_data = result['input']
            print(f"\nInput {i+1}:")
            print(f"  Crop: {input_data['Crop']}")
            print(f"  State: {input_data['State']}")
            print(f"  Season: {input_data['Season']}")
            print(f"  Year: {input_data['Crop_Year']}")
            print(f"  Prediction: {result['prediction']} Yield Expected")
        else:
            print(f"\nInput {i+1}: Error - {result['error']}")\
            