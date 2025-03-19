from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os



scaler_path = "models/scaler.pkl"  # Shared scaler path

def train_model_knn(data, model_path="models/knn_model.pkl"):

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns="Result"), data["Result"], test_size=0.2, random_state=0, stratify=data["Result"]
    )

    # Load the shared scaler (created in train_svm.py)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("\nLoaded shared scaler for KNN.")
    else:
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}. Train SVM first!")

    # Apply the same feature scaling
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean")

    print("\nTraining K-Nearest Neighbors model...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nKNN Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"\nKNN model saved to {model_path}")
