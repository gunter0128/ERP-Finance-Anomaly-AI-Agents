import pandas as pd
from pathlib import Path
import os
from openai import OpenAI

client = OpenAI()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def _chat_zh(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def _split_summary_full(text: str) -> tuple[str, str]:
    if "【重點版】" in text and "【完整版】" in text:
        parts = text.split("【完整版】", 1)
        summary_part = parts[0].replace("【重點版】", "").strip()
        full_part = parts[1].strip()
        return summary_part, full_part
    else:
        lines = text.splitlines()
        summary_part = "\n".join(lines[:10])
        return summary_part.strip(), text.strip()


def build_vendor_risk_table(df: pd.DataFrame, vendors: pd.DataFrame, top_n: int = 20) -> str:
    """整理每個供應商的風險指標（逾期比率、平均延遲天數、異常發票數量等）"""
    vendor_agg = (
        df.groupby("vendor_id")
        .agg(
            num_invoices=("invoice_id", "count"),
            num_anomaly=("anomaly_flag", lambda x: (x == -1).sum()),
            avg_delay_days=("delay_days", "mean"),
            overdue_ratio=("is_overdue", "mean"),
            avg_amount=("invoice_total_amount", "mean"),
        )
        .reset_index()
    )

    vendor_agg = vendor_agg.merge(vendors, on="vendor_id", how="left")

    vendor_agg["risk_score"] = (
        vendor_agg["num_anomaly"] * 2
        + vendor_agg["overdue_ratio"] * 5
        + vendor_agg["avg_delay_days"].clip(lower=0) * 0.1
    )

    vendor_agg = vendor_agg.sort_values("risk_score", ascending=False).head(top_n)

    cols = [
        "vendor_id",
        "vendor_name",
        "category",
        "region",
        "num_invoices",
        "num_anomaly",
        "avg_delay_days",
        "overdue_ratio",
        "avg_amount",
        "risk_score",
    ]

    lines = []
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)

    for row in vendor_agg.itertuples(index=False):
        values = [str(getattr(row, c)) for c in cols]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def assess_vendor_risk(df_anomaly_result: pd.DataFrame, vendors: pd.DataFrame, top_n: int = 15) -> tuple[str, str]:
    """
    傳入 invoice_anomaly_result & vendors，
    回傳 (summary_text, full_text)
    """
    vendor_table_md = build_vendor_risk_table(df_anomaly_result, vendors, top_n=top_n)

    system_prompt = """
你是一位熟悉採購與財務風險的供應商管理專家，負責協助企業評估供應商信用與合作壓力。
請用繁體中文回答。
    """.strip()

    user_prompt = f"""
以下是系統彙整出的「供應商風險指標」表格（每一列是一個供應商）：

{vendor_table_md}

欄位說明：
- num_invoices：該供應商的發票數量
- num_anomaly：被異常偵測模型標記為「可疑」的發票數量
- avg_delay_days：平均付款延遲天數
- overdue_ratio：逾期發票比例
- avg_amount：平均單張發票金額
- risk_score：系統依據上述指標計算出的風險分數（越高代表風險越高）

請你輸出兩個段落：

【重點版】
- 以條列式列出風險最高的 3~5 個供應商與主要風險原因，給主管快速看懂。

【完整版】
- 詳細說明各高風險供應商的數字與情境
- 可能帶來的實務風險
- 建議財務 / 採購部門後續可以採取的行動
- 再做一段整體供應商風險的總結

請嚴格使用上面的標題標示分段：「【重點版】」「【完整版】」。
    """.strip()

    text = _chat_zh(system_prompt, user_prompt)
    return _split_summary_full(text)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    df_test = pd.read_csv(ROOT / "data" / "processed" / "invoice_anomaly_result.csv")
    vendors_test = pd.read_csv(ROOT / "data" / "raw" / "vendors.csv")
    s, f = assess_vendor_risk(df_test, vendors_test)
    print("=== 重點版 ===")
    print(s)
    print("\n=== 完整版 ===")
    print(f)
