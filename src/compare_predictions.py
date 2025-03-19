import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# File paths
test_data_path = "data/test_data.csv"
svm_model_path = "models/svm_model.pkl"
knn_model_path = "models/knn_model.pkl"
rf_model_path = "models/random_forest.pkl"
scaler_path = "models/scaler.pkl"  # Shared scaler for SVM & KNN

def load_test_data(file_path):
    df = pd.read_csv(file_path)
    X_test = df.drop(columns=["Result"])
    y_test = df["Result"]
    return X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Accuracy: {accuracy:.4f}")

    print(f"\nClassification Report for {model_name}:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Phishy (-1)", "Suspicious (0)", "Legitimate (1)"],
                yticklabels=["Phishy (-1)", "Suspicious (0)", "Legitimate (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

def compare_models():

    X_test, y_test = load_test_data(test_data_path)

    svm_model = joblib.load(svm_model_path)
    knn_model = joblib.load(knn_model_path)
    rf_model = joblib.load(rf_model_path)

    # Load shared scaler (for SVM and KNN)
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)  # Apply same scaling as training

    print("\nComparing Models on Final Test Data...")

    evaluate_model(svm_model, X_test_scaled, y_test, "SVM")
    evaluate_model(knn_model, X_test_scaled, y_test, "KNN")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")  # RF does NOT need scaling

if __name__ == "__main__":
    compare_models()
