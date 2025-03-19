from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

scaler_path = "models/scaler.pkl"  # Shared scaler path

def train_model_svm(data, model_path="models/svm_model.pkl"):

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns="Result"), data["Result"], test_size=0.2, random_state=0, stratify=data["Result"]
    )

    # Apply feature scaling (Shared for both SVM & KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    print(f"\nShared Scaler saved to {scaler_path}")

    model = SVC(kernel='linear', class_weight='balanced', random_state=0)

    print("\nTraining SVM model...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nSVM Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"\nSVM model saved to {model_path}")
