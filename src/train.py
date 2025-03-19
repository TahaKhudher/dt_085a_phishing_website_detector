from train_knn import train_model_knn
from train_rf import train_model_rf
from train_svm import train_model_svm

import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

def load_data(file_path="data/PhishingData.arff", output_train="data/preprocessed_data.csv", output_test="data/test_data.csv", reserve_size=0.1):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    # Convert categorical attributes from bytes to integers
    for col in df.columns:
        df[col] = df[col].astype(int)

    df.drop_duplicates(inplace=True)
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    # Reserve part of the dataset for final testing (not included in training)
    train_data, reserved_test_data = train_test_split(df, test_size=reserve_size, random_state=0, stratify=df["Result"])
    
    train_data.to_csv(output_train, index=False)
    reserved_test_data.to_csv(output_test, index=False)

    print(f"data saved to {output_train}")
    print(f"Reserved test data saved to {output_test} (For final testing)")

    return train_data  

def train_models():
    data = load_data()
    train_model_svm(data)
    train_model_knn(data)
    train_model_rf(data)

if __name__ == "__main__":
    train_models()