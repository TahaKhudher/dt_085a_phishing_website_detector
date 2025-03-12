import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_test_data(file_path="data/preprocessed_phishing_test.csv"):
    data = pd.read_csv(file_path)

    # Separate features and labels
    X_test = data.drop(columns=['label'])  # Features
    y_test = data['label']  # Actual labels

    return X_test, y_test

def predict_multiple(X_test, y_test, model_path="models/random_forest.pkl"):

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Prediction Accuracy: {accuracy:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)

    results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print("\n Sample Predictions:\n", results.head(10))

def check_train_test_overlap(train_file="data/preprocessed_data.csv", test_file="data/preprocessed_phishing_test.csv"):
    train_data = pd.read_csv(train_file).drop(columns=['label'])
    test_data = pd.read_csv(test_file).drop(columns=['label'])

    duplicates = train_data.merge(test_data, how='inner')
    print(f"Number of duplicates between train and test: {len(duplicates)}")



if __name__ == '__main__':
    X_test, y_test = load_test_data()
    predict_multiple(X_test, y_test)
    check_train_test_overlap()
