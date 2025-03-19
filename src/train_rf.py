from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model_rf(data, model_path="models/random_forest.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns="Result"), data["Result"], test_size=0.2, random_state=0, stratify=data["Result"]
    )

    model = RandomForestClassifier(n_estimators=150, random_state=0)
    print("\n---------------------------------\n")
    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

