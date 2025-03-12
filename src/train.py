import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(file_path='data/preprocessed_data.csv'):
    data = pd.read_csv(file_path)
    data.drop_duplicates()
    data = data.sample(frac=1).reset_index(drop=True)
    return data

def train_model(data, model_path="models/random_forest.pkl", test_data_path="data/preprocessed_phishing_test.csv"):
    # Split data (80% training, 20% testing)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)
    test_data.to_csv(test_data_path)
    print("Train and Test data saved")
    X_train, y_train = train_data.drop(columns="label"), train_data["label"]
    X_test, y_test = test_data.drop(columns="label"), test_data["label"]

    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🔹 Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def check_feature_correlation(file_path="data/preprocessed_data.csv"):
    df = pd.read_csv(file_path)
    correlation_matrix = df.corr()
    print("\nFeature Correlation with Label:\n", correlation_matrix["label"].sort_values(ascending=False))



if __name__ == '__main__':
    data = load_data()
    train_model(data)
    check_feature_correlation()
