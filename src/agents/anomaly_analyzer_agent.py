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
    """
    期待輸出格式：

    【重點版】
    ...

    【完整版】
    ...

    如果切不開，就全部當完整版，重點版用前幾行。
    """
    if "【重點版】" in text and "【完整版】" in text:
        parts = text.split("【完整版】", 1)
        summary_part = parts[0].replace("【重點版】", "").strip()
        full_part = parts[1].strip()
        return summary_part, full_part
    else:
        lines = text.splitlines()
        summary_part = "\n".join(lines[:10])
        return summary_part.strip(), text.strip()


def build_invoice_summary_table(df_anom: pd.DataFrame, top_n: int = 30) -> str:
    """把 top N 異常發票整理成給 LLM 看的 markdown 表格。"""
    cols = [
        "invoice_id",
        "vendor_id",
        "currency",
        "invoice_total_amount",
        "total_paid_amount",
        "balance_amount",
        "delay_days",
        "is_overdue",
        "is_partial_paid",
        "is_overpaid",
        "amount_zscore",
        "anomaly_score",
    ]

    df = df_anom.sort_values("anomaly_score").head(top_n)[cols]

    lines = []
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)

    for row in df.itertuples(index=False):
        values = [str(getattr(row, c)) for c in cols]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def analyze_anomalies(df_anomaly_result: pd.DataFrame, top_n: int = 30) -> tuple[str, str]:
    """
    傳入 invoice_anomaly_result 的 DataFrame，
    回傳 (summary_text, full_text)
    """
    df_anom = df_anomaly_result[df_anomaly_result["anomaly_flag"] == -1].copy()
    if df_anom.empty:
        return "目前沒有被模型標記為異常的發票。", "目前沒有被模型標記為異常的發票。"

    table_text = build_invoice_summary_table(df_anom, top_n=top_n)

    system_prompt = """
你是一位企業財務部門的資深風險控管經理，負責協助解讀「應付帳款 / 發票」異常狀況。
請用繁體中文回答，語氣專業但讓非財會背景的人也看得懂。
    """.strip()

    user_prompt = f"""
以下是一批由異常偵測模型標記為「可疑」的發票紀錄（每一列是一張發票）：

{table_text}

欄位說明：
- invoice_total_amount：發票金額（含稅）
- total_paid_amount：實際已付款總額
- balance_amount：發票金額 - 已付金額（負值代表超額付款）
- delay_days：實際付款日相對於應付日的延遲天數（負值代表提早付款）
- is_overdue：是否逾期
- is_partial_paid：是否只付部分款項
- is_overpaid：是否超額付款
- amount_zscore：此發票金額相對於該供應商歷史發票的標準化分數（>2 代表明顯偏大）
- anomaly_score：模型算出的異常分數，愈小代表愈異常

請你輸出兩個段落：

【重點版】
- 列出 3~6 點「最重要的異常類型與風險」，給主管快速看懂今天的異常狀況。

【完整版】
- 完整說明各種異常類型
- 逐類型說明可能原因與風險
- 列出 5~10 個最需要優先處理的風險情境與建議做法

請嚴格使用上面的標題標示分段：「【重點版】」「【完整版】」。
    """.strip()

    text = _chat_zh(system_prompt, user_prompt)
    return _split_summary_full(text)


if __name__ == "__main__":
    # 手動測試用
    ROOT = Path(__file__).resolve().parents[2]
    df_test = pd.read_csv(ROOT / "data" / "processed" / "invoice_anomaly_result.csv")
    s, f = analyze_anomalies(df_test)
    print("=== 重點版 ===")
    print(s)
    print("\n=== 完整版 ===")
    print(f)
