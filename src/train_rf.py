import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path='data/PhishingData.arff'):
    # Load ARFF file
    data, meta = arff.loadarff(file_path)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)

    # Convert categorical attributes from bytes to integers
    for col in df.columns:
        df[col] = df[col].astype(int)

    df.drop_duplicates(inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data

    return df

def train_model(data, model_path="models/random_forest.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns="Result"), data["Result"], test_size=0.2, random_state=0, stratify=data["Result"]
    )

    # Using Random Forest Classifier for multi-class classification
    model = RandomForestClassifier(n_estimators=150, random_state=0)

    print("⏳ Training Random Forest model...")
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

def feature_corr(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()
    print("\nFeature Correlation with Label:\n", correlation_matrix["Result"].sort_values(ascending=False))

if __name__ == '__main__':
    data = load_data()
    feature_corr(data)
    train_model(data)
