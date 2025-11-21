import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path
import joblib

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_model():
    df = pd.read_csv(PROCESSED_DIR / "invoice_features.csv")

    # 選出要用來辨識異常的特徵
    feature_cols = [
        "invoice_total_amount",
        "total_paid_amount",
        "balance_amount",
        "delay_days",
        "vendor_avg_invoice_amount",
        "vendor_std_invoice_amount",
        "vendor_avg_delay_days",
        "vendor_overdue_ratio",
        "amount_zscore",
    ]

    X = df[feature_cols].fillna(0.0)

    # IsolationForest 訓練
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,    # 假設 5% 是異常
        random_state=42
    )

    model.fit(X)

    # 儲存模型
    joblib.dump(model, MODEL_DIR / "anomaly_model.pkl")
    print("Anomaly model saved to models/anomaly_model.pkl")


if __name__ == "__main__":
    train_model()
