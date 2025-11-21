import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def build_features():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 讀原始資料
    vendors = pd.read_csv(RAW_DIR / "vendors.csv")
    invoices = pd.read_csv(RAW_DIR / "invoices.csv")
    payments = pd.read_csv(RAW_DIR / "payments.csv")

    # 2. 日期型別轉成 datetime
    invoices["invoice_date"] = pd.to_datetime(invoices["invoice_date"])
    invoices["due_date"] = pd.to_datetime(invoices["due_date"])
    payments["payment_date"] = pd.to_datetime(payments["payment_date"])

    # 3. 付款彙總：每張發票的總付款金額、最後付款日、付款次數
    pay_agg = (
        payments.groupby("invoice_id")
        .agg(
            total_paid_amount=("amount", "sum"),
            last_payment_date=("payment_date", "max"),
            num_payments=("payment_id", "count"),
        )
        .reset_index()
    )

    df = invoices.merge(pay_agg, on="invoice_id", how="left")

    # 4. 基本金額特徵
    df["invoice_total_amount"] = df["amount"] + df["tax_amount"]
    df["total_paid_amount"] = df["total_paid_amount"].fillna(0.0)
    df["num_payments"] = df["num_payments"].fillna(0).astype(int)

    # 5. 分析基準日（給還沒付完的發票算「已逾期幾天」）
    analysis_date = max(df["due_date"].max(), payments["payment_date"].max())

    df["effective_pay_date"] = df["last_payment_date"]
    df["effective_pay_date"] = df["effective_pay_date"].fillna(analysis_date)

    # 6. 付款延遲（天數）與逾期旗標
    df["delay_days"] = (df["effective_pay_date"] - df["due_date"]).dt.days
    df["is_overdue"] = df["delay_days"] > 0

    # 7. 餘額 / 超額付款 / 部分付款
    df["balance_amount"] = df["invoice_total_amount"] - df["total_paid_amount"]
    df["is_overpaid"] = df["balance_amount"] < -1e-6
    df["is_fully_paid"] = df["balance_amount"].abs() <= 1e-6
    df["is_partial_paid"] = (~df["is_fully_paid"]) & (df["total_paid_amount"] > 0)

    # 8. 供應商層級統計（之後拿來算 Z-score）
    vendor_stats = (
        df.groupby("vendor_id")
        .agg(
            vendor_avg_invoice_amount=("invoice_total_amount", "mean"),
            vendor_std_invoice_amount=("invoice_total_amount", "std"),
            vendor_avg_delay_days=("delay_days", "mean"),
            vendor_overdue_ratio=("is_overdue", "mean"),
        )
        .reset_index()
    )

    df = df.merge(vendor_stats, on="vendor_id", how="left")

    # 9. 金額 Z-score（看某張發票在該 vendor 的金額有多異常）
    df["vendor_std_invoice_amount"] = df["vendor_std_invoice_amount"].fillna(0.0)
    df["amount_zscore"] = 0.0
    mask = df["vendor_std_invoice_amount"] > 0
    df.loc[mask, "amount_zscore"] = (
        (df.loc[mask, "invoice_total_amount"] - df.loc[mask, "vendor_avg_invoice_amount"])
        / df.loc[mask, "vendor_std_invoice_amount"]
    )

    # 10. 選出要輸出的欄位
    cols = [
        "invoice_id",
        "vendor_id",
        "invoice_date",
        "due_date",
        "status",
        "currency",
        "invoice_total_amount",
        "total_paid_amount",
        "balance_amount",
        "num_payments",
        "last_payment_date",
        "effective_pay_date",
        "delay_days",
        "is_overdue",
        "is_fully_paid",
        "is_partial_paid",
        "is_overpaid",
        "vendor_avg_invoice_amount",
        "vendor_std_invoice_amount",
        "vendor_avg_delay_days",
        "vendor_overdue_ratio",
        "amount_zscore",
    ]

    df_out = df[cols].sort_values("invoice_id")

    out_path = PROCESSED_DIR / "invoice_features.csv"
    df_out.to_csv(out_path, index=False)
    print(f"saved features to {out_path}")


if __name__ == "__main__":
    build_features()
