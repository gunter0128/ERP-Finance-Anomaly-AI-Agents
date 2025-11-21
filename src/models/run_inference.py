import pandas as pd
import joblib
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

def run_inference():
    df = pd.read_csv(PROCESSED_DIR / "invoice_features.csv")
    model = joblib.load(MODEL_DIR / "anomaly_model.pkl")

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

    # predict：
    # -1 = anomaly
    # 1  = normal
    df["anomaly_flag"] = model.predict(X)
    df["anomaly_score"] = model.decision_function(X)

    out_path = PROCESSED_DIR / "invoice_anomaly_result.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved anomaly result to {out_path}")


if __name__ == "__main__":
    run_inference()
