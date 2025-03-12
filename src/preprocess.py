import pandas as pd

def load_data(file_path = "data/phishing.csv", output_path = "data/preprocessed_data.csv"):
    data = pd.read_csv(file_path)
    print(data.head())
    print(data.info())
    data.drop(columns=["FILENAME", "URL", "Domain", "TLD", "Title"], inplace=True)
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    load_data()
