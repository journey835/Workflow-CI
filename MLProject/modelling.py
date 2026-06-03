import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def train_base_model(data_dir):
    print(f"Membaca data dari path: {data_dir}")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_ready.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_ready.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model berhasil dilatih dengan sukses!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="digital_marketing_campaign_dataset_preprocessing")
    args = parser.parse_args()
    
    train_base_model(data_dir=args.data_dir)