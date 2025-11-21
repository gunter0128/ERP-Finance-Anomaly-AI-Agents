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


def build_summary_text(df: pd.DataFrame) -> str:
    total_invoices = len(df)
    total_amount = df["invoice_total_amount"].sum()
    total_paid = df["total_paid_amount"].sum()
    total_balance = df["balance_amount"].sum()

    num_anomaly = (df["anomaly_flag"] == -1).sum()
    anomaly_amount = df.loc[df["anomaly_flag"] == -1, "invoice_total_amount"].sum()

    avg_delay = df["delay_days"].mean()
    overdue_ratio = df["is_overdue"].mean()

    summary_lines = [
        f"總發票數量: {total_invoices}",
        f"總發票金額(含稅): {round(total_amount, 2)}",
        f"已付款總金額: {round(total_paid, 2)}",
        f"整體應付餘額(可能為負代表超付): {round(total_balance, 2)}",
        "",
        f"被模型標記為異常的發票數量: {num_anomaly}",
        f"異常發票金額合計: {round(anomaly_amount, 2)}",
        "",
        f"平均付款延遲天數(delay_days 平均): {round(avg_delay, 2)}",
        f"逾期發票比例(is_overdue 平均): {round(overdue_ratio, 3)}",
    ]
    return "\n".join(summary_lines)


def generate_cfo_report(
    df_anomaly_result: pd.DataFrame,
    anomaly_summary: str,
    anomaly_full: str,
    vendor_summary: str,
    vendor_full: str,
) -> tuple[str, str]:
    """
    回傳 (summary_text, full_text)
    """

    summary_text = build_summary_text(df_anomaly_result)

    system_prompt = """
你是一位上市公司 CFO 的特助，負責把財務系統與風險分析結果整理成「管理報告」給高階主管。
請用繁體中文撰寫，語氣專業、條理清楚，但不要太技術化。
    """.strip()

    user_prompt = f"""
以下是系統產出的財務數據摘要與風險分析結果，請你幫我整理成一份給 CFO/財務長的簡報稿（文字版）。

[一] 系統數據摘要（數字）：
{summary_text}

[二] 異常發票分析 - 重點版：
{anomaly_summary}

[二-補充] 異常發票分析 - 完整版：
{anomaly_full}

[三] 供應商風險分析 - 重點版：
{vendor_summary}

[三-補充] 供應商風險分析 - 完整版：
{vendor_full}

請你輸出兩個段落：

【重點版】
- 用 2~4 個段落，讓 CFO 在 1 分鐘內掌握今天的財務風險總覽
- 包含：異常發票概況、供應商風險、付款行為的趨勢

【完整版】
- 依照以下結構撰寫完整報告：
  1. 今日財務風險總覽（1~2 段話）
  2. 異常發票與付款行為重點（條列式，3~6 點）
  3. 供應商風險與合作建議（條列式，3~6 點）
  4. 建議 CFO 可以關注或要求追蹤的行動項目（3~5 點）

注意：
- 報告對象是管理階層，不是工程師，因此不要提到 IsolationForest 或模型細節。
- 可以提到「系統偵測」、「異常趨勢」、「高風險供應商」、「付款習慣」等用語。
- 文字請用繁體中文。

請嚴格使用上面的標題標示分段：「【重點版】」「【完整版】」。
    """.strip()

    text = _chat_zh(system_prompt, user_prompt)
    return _split_summary_full(text)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    df_test = pd.read_csv(ROOT / "data" / "processed" / "invoice_anomaly_result.csv")

    # 示範隨便塞假資料給 anomaly/vendor summary
    demo_anom_sum = "（這裡通常會是 anomaly_analyzer_agent 給的重點版）"
    demo_anom_full = "（這裡通常會是 anomaly_analyzer_agent 給的完整版）"
    demo_vendor_sum = "（這裡通常會是 vendor_risk_agent 給的重點版）"
    demo_vendor_full = "（這裡通常會是 vendor_risk_agent 給的完整版）"

    s, f = generate_cfo_report(
        df_test, demo_anom_sum, demo_anom_full, demo_vendor_sum, demo_vendor_full
    )
    print("=== 重點版 ===")
    print(s)
    print("\n=== 完整版 ===")
    print(f)
