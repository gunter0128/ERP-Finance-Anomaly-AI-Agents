import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 讓 Python 找得到 src 這個專案目錄
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.append(str(SRC_DIR))

from agents.anomaly_analyzer_agent import analyze_anomalies
from agents.vendor_risk_agent import assess_vendor_risk
from agents.cfo_report_agent import generate_cfo_report



DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"

# -----------------------------------
# 工具函式
# -----------------------------------

@st.cache_data
def load_invoice_anomaly_result():
    path = DATA_PROCESSED_DIR / "invoice_anomaly_result.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    date_cols = ["invoice_date", "due_date", "last_payment_date", "effective_pay_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


@st.cache_data
def load_vendors():
    path = DATA_RAW_DIR / "vendors.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# -----------------------------------
# Dashboard 頁
# -----------------------------------

def page_dashboard():
    st.header("ERP 財務異常偵測 Dashboard")

    st.markdown("""
        **這個頁面提供整體 ERP / AP（應付帳款）風險的快速總覽：**
        - 顯示系統偵測到的發票異常比例
        - 顯示付款延遲、逾期、餘額等財務風險指標
        - 顯示供應商異常狀況統計
        - 作為 CFO / 會計主管的每日「快速風險檢視畫面」
        """)

    df = load_invoice_anomaly_result()
    if df is None:
        st.error("找不到 invoice_anomaly_result.csv，請先完成資料前處理與推論。")
        return

    st.subheader("整體指標")

    total_invoices = len(df)
    total_amount = df["invoice_total_amount"].sum()
    total_paid = df["total_paid_amount"].sum()
    total_balance = df["balance_amount"].sum()
    num_anomaly = (df["anomaly_flag"] == -1).sum()
    anomaly_ratio = num_anomaly / total_invoices
    avg_delay = df["delay_days"].mean()
    overdue_ratio = df["is_overdue"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("發票總數", f"{total_invoices}")
    c2.metric("總金額(含稅)", f"{total_amount:,.0f}")
    c3.metric("已付款總額", f"{total_paid:,.0f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("異常發票數", f"{num_anomaly}")
    c5.metric("異常比例", f"{anomaly_ratio:.2%}")
    c6.metric("平均延遲天數", f"{avg_delay:.2f}")

    st.metric("整體應付餘額", f"{total_balance:,.0f}")
    st.metric("逾期比例", f"{overdue_ratio:.2%}")

    st.divider()

    st.subheader("供應商異常統計")

    vendor_summary = (
        df.groupby("vendor_id")
        .agg(
            num_invoices=("invoice_id", "count"),
            num_anomaly=("anomaly_flag", lambda x: (x == -1).sum()),
            avg_delay_days=("delay_days", "mean"),
            overdue_ratio=("is_overdue", "mean"),
        )
        .reset_index()
    )
    vendor_summary["anomaly_ratio"] = (
        vendor_summary["num_anomaly"] / vendor_summary["num_invoices"]
    )

    st.dataframe(
        vendor_summary.sort_values("num_anomaly", ascending=False),
        use_container_width=True,
        height=400,
    )


# -----------------------------------
# 發票查詢頁
# -----------------------------------

def page_invoice_explorer():
    st.header("發票查詢與異常判斷")

    df = load_invoice_anomaly_result()
    if df is None:
        st.error("找不到 invoice_anomaly_result.csv")
        return

    tab1, tab2 = st.tabs(["發票列表", "單筆查詢"])

    with tab1:
        st.subheader("所有發票")

        mode = st.radio("篩選：", ["全部", "僅異常"], horizontal=True)
        df_show = df.copy()
        if mode == "僅異常":
            df_show = df_show[df_show["anomaly_flag"] == -1]

        st.dataframe(
            df_show.sort_values("anomaly_score").head(200),
            use_container_width=True,
            height=480,
        )

    with tab2:
        st.subheader("查詢單一發票")

        invoice_id = st.text_input("輸入發票編號（如：INV0001）")
        if invoice_id:
            row = df[df["invoice_id"] == invoice_id]
            if row.empty:
                st.warning("查無此發票")
            else:
                r = row.iloc[0]
                st.json(r.to_dict())

                # 顯示模型判斷
                if r["anomaly_flag"] == -1:
                    st.error("⚠ 這張發票被模型標記為異常")
                else:
                    st.success("正常")

                st.subheader("📌 這張發票的異常判斷依據")
                st.markdown(f"""
                **系統判斷這張發票是否異常會依據以下條件：**

                - **是否逾期（is_overdue）**：`{bool(r["is_overdue"])}`  
                - **是否部分付款（is_partial_paid）**：`{bool(r["is_partial_paid"])}`  
                - **是否超額付款（is_overpaid）**：`{bool(r["is_overpaid"])}`  
                - **金額是否明顯偏離供應商過去水平（amount_zscore）**：`{round(r["amount_zscore"], 3)}`  
                - **模型異常旗標（anomaly_flag）**：`{r["anomaly_flag"]}`（-1 表示異常）  
                - **模型異常分數（anomaly_score）**：`{round(r["anomaly_score"], 5)}`（愈小愈可疑）

                **模型邏輯簡化說明：**
                - 逾期＋延遲天數越高 → 風險上升  
                - 部分付款、超額付款會觸發異常特徵  
                - 金額 Z-score > 2（相對供應商歷史異常偏大） → 增加異常可能  
                - 以上特徵會透過 IsolationForest 權重化，算出 anomaly_score  
                - anomaly_score < 0 模型將標記為異常（-1）
                """)


# -----------------------------------
# 供應商風險頁
# -----------------------------------

def page_vendor_risk():
    st.header("供應商風險分析（LLM 驅動）")
    st.markdown("""
        **這個頁面用於分析供應商風險指標：**
        - 根據異常發票數、逾期比例、平均延遲天數等指標排序
        - LLM 會輸出「重點版」＋「完整版」供應商風險說明
        - 可用來支援採購部門、稽核、財務部門做供應商管理
        """)

    df = load_invoice_anomaly_result()
    vendors = load_vendors()
    if df is None or vendors is None:
        st.error("缺少資料，請確認 data/raw 與 data/processed 完整。")
        return

    st.subheader("一鍵產生：供應商風險分析（重點版 + 完整版）")

    if st.button("生成供應商風險報告（呼叫 OpenAI）"):
        with st.spinner("分析中，請稍候..."):
            summary, full = assess_vendor_risk(df, vendors)

        st.session_state["vendor_summary"] = summary
        st.session_state["vendor_full"] = full

    # 顯示結果
    if "vendor_summary" in st.session_state:
        st.subheader("【重點版】")
        st.text(st.session_state["vendor_summary"])

        st.subheader("【完整版】")
        with st.expander("點我展開完整版"):
            st.text(st.session_state["vendor_full"])


# -----------------------------------
# CFO 報告頁
# -----------------------------------

def page_cfo_report():
    st.header("CFO 財務風險報告（LLM 自動生成）")
    st.markdown("""
        **這個頁面會將全部風險整合成 CFO 等級的報告：**
        - 異常發票重點  
        - 供應商風險  
        - 整體財務風險與可操作建議  
        - 有「重點版」與「完整版」可切換  
        """)

    df = load_invoice_anomaly_result()
    vendors = load_vendors()
    if df is None:
        st.error("缺少 invoice_anomaly_result.csv")
        return

    st.subheader("一鍵產生完整 CFO 報告")

    if st.button("生成 CFO 報告"):
        # Step 1：異常發票
        with st.spinner("Step 1：異常發票分析..."):
            anom_sum, anom_full = analyze_anomalies(df)

        # Step 2：供應商風險
        with st.spinner("Step 2：供應商風險分析..."):
            vendor_sum, vendor_full = assess_vendor_risk(df, vendors)

        # Step 3：CFO 報告
        with st.spinner("Step 3：彙整 CFO 報告..."):
            cfo_sum, cfo_full = generate_cfo_report(
                df, anom_sum, anom_full, vendor_sum, vendor_full
            )

        st.session_state["cfo_sum"] = cfo_sum
        st.session_state["cfo_full"] = cfo_full

    if "cfo_sum" in st.session_state:
        st.subheader("【重點版】")
        st.text(st.session_state["cfo_sum"])

        st.subheader("【完整版】")
        with st.expander("點我展開完整版"):
            st.text(st.session_state["cfo_full"])


# -----------------------------------
# 主入口
# -----------------------------------

def main():
    st.set_page_config(
        page_title="ERP Finance AI",
        layout="wide"
    )

    st.sidebar.title("ERP AI 財務助理")
    page = st.sidebar.radio(
        "選擇頁面",
        ["📊 Dashboard", "🧾 發票查詢", "🏭 供應商風險", "📑 CFO 報告"],
    )

    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "🧾 發票查詢":
        page_invoice_explorer()
    elif page == "🏭 供應商風險":
        page_vendor_risk()
    elif page == "📑 CFO 報告":
        page_cfo_report()


if __name__ == "__main__":
    main()
