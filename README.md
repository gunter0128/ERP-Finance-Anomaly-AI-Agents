
# ERP Finance Anomaly AI Agents

### 財務異常偵測 × 供應商風險 × CFO 管理報告 × AI 多代理助理

> 使用「ERP 發票 / 付款資料（AP）＋異常偵測模型（ML）＋LLM 多代理（Agents）」
> 自動化產生 **異常發票解讀、供應商風險分析、CFO 財務風險摘要**，支援財務 / 採購部門的決策流程。

---

# 1. 為什麼做這個主題？

### 這是符合 **Business AI Engineer / Data Engineer / AI Solution Engineer** 職位的實際體現

並同時結合我的多代理設計、企業資料分析與 RAG/LLM 實戰能力。

ERP（Enterprise Resource Planning）是每間企業的核心系統，尤其 AP / AR（應付 / 應收）更攸關：

* **是否有異常付款？**
* **是否有高風險供應商？**
* **哪些發票金額不合理？**
* **哪些延遲付款已經造成財務壓力？**

本專案展示：

* 能處理 **ERP 的發票/付款資料**
* 自行製作 **財務異常偵測模型（Isolation Forest）**
* 建立 **延遲、超額、部分付款等財務特徵工程**
* **三個 AI Agents 分工合作：異常解讀 / 供應商風險 / CFO 報告**
* 使用 **Streamlit 建立財務 Dashboard**（可直接 Demo）

專案目的：
**讓主管看到我能把資料 → 模型 → 多代理 → UI 整合成一套真正能用的 ERP AI 系統。**

---

# **2. ERP Finance Anomaly AI：快速總覽（Problem / Input / Output）**

## 2.1 要解決的問題（Problem）

財務與採購部門常遇到：

* 每天有大量發票要核對
* 不確定哪些是「真正高風險」
* 逾期 / 超額付款不容易被即時發現
* 供應商風險不透明
* 報告整理需要人工花大量時間

本專案透過 **ML 異常偵測 + 多代理 LLM 分析**，自動找出財務風險並產生管理用報告。

---

## 2.2 系統輸入（Input）

### 2.2.1 ERP 原始資料（參考Oracle Cloud Financials：AP_INVOICE_PAYMENTS_ALL 表結構 & n8n 官方 workflow：Automated Invoice Payment Tracking 後自行建立）

三份 ERP 常見主檔：

```
vendors.csv
invoices.csv
payments.csv
```

每個欄位包含：

**Vendor 主檔：**
`vendor_id, vendor_name, region, category`

**Invoice 主檔：**
`invoice_id, vendor_id, invoice_date, due_date, currency, amount …`

**Payment 主檔：**
`payment_id, invoice_id, amount, payment_date …`

---

### 2.2.2 特徵工程後的欄位（由 build_features.py 自動生成）

* `delay_days`
* `is_overdue`
* `is_partial_paid`
* `is_overpaid`
* `amount_zscore`
* `balance_amount`
* `invoice_total_amount`
* `total_paid_amount`

---

## 2.3 系統輸出（Output）

* 異常發票重點版（供主管快速看）
* 異常發票完整版（深度解讀異常原因）
* 供應商風險分析（排名＋理由）
* CFO 財務風險總覽（重點版＋完整版）
* Streamlit Dashboard 互動呈現

---

# 3. 系統架構概觀

整體分成四層：

---

## 3.1 資料前處理層（Data Prep）

來源資料：
`data/raw/*.csv`（ERP 主檔）

工作：

* 合併發票＋付款資料
* 計算財務特徵（延遲、餘額、Z-score 等）
* 輸出 `invoice_features.csv`

---

## 3.2 異常模型層（ML Anomaly Detection）

使用模型：

```
Isolation Forest
```

輸出：

* `anomaly_flag`（-1 = 可疑）
* `anomaly_score`（越小越異常）

用途：

讓模型自動標記可能的財務風險。

---

## 3.3 AI Agents 層（Finance Multi-Agents）

本專案使用 **三個 Agents**：

---

### **Agent 1 — 異常分析 Agent**

* 讀取 top N 異常發票
* 自動分類異常原因
  （逾期、部分付款、超額、金額異常等）
* 輸出：

  * 【重點版】主管快速讀懂
  * 【完整版】深入異常類型分解

---

### **Agent 2 — 供應商風險 Agent**

* 根據供應商付款行為產生風險評分
* 分析原因（逾期、異常量、金額異常…）
* 輸出：

  * 【重點版】最高風險供應商摘要
  * 【完整版】完整供應商風險報告

---

### **Agent 3 — CFO 報告 Agent**

輸入前兩個 Agent 的結果，彙整成 CFO 等級報告：

1. 異常發票總結
2. 供應商風險
3. 財務健康度
4. CFO 可採取的行動建議

同樣分成 **重點版＋完整版**

---

## 3.4 Streamlit DEMO 層（UI）

功能：

✔ Dashboard：整體財務指標（異常比例、逾期、供應商狀況）
✔ 發票查詢：單筆發票異常原因
✔ 供應商風險：一鍵生成 LLM 分析
✔ CFO 報告：一次呼叫所有 Agents

---

# 4. 使用資料集：ERP Finance Dataset（參考Oracle Cloud Financials：AP_INVOICE_PAYMENTS_ALL 表結構 & n8n 官方 workflow：Automated Invoice Payment Tracking 後自行建立）

本專案使用自行建立的 ERP 三主檔：

```
vendors.csv
invoices.csv
payments.csv
```

包含資訊：

* 供應商基本資料
* 發票金額 / 帳期 / 到期日
* 付款紀錄（多筆付款）
* 特徵工程後的 delay / overdue / z-score

放在：

```
data/raw/
```

特徵工程輸出：

```
data/processed/invoice_features.csv
data/processed/invoice_anomaly_result.csv
```

---

# 5. 專案目錄結構

```
erp-finance-anomaly-ai/
├── data/
│   ├── raw/                      # ERP 原始資料
│   └── processed/                # 特徵工程 + 模型結果
│
├── src/
│   ├── data_prep/
│   │   └── build_features.py     # 特徵工程
│   │
│   ├── models/
│   │   ├── train_anomaly_model.py # 訓練 IsolationForest 異常偵測模型
│   │   └── run_inference.py       # 使用訓練好的模型對發票做異常預測 (產生 anomaly_flag / score)
│   │
│   ├── agents/
│   │   ├── anomaly_analyzer_agent.py # Agent 1：針對「異常發票」做解讀與分群，輸出重點版 + 完整版說明
│   │   ├── vendor_risk_agent.py      # Agent 2：彙總每個供應商的異常/逾期指標，產生供應商風險報告
│   │   └── cfo_report_agent.py       # Agent 3：整合數據 + 前兩個 Agent 結果，產生 CFO 財務風險管理報告
│   │
│   └── app/
│       └── streamlit_app.py      # Streamlit 前端
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 6. 如何重現專案

## 6.1 建立環境

```bash
git clone https://github.com/<your-account>/erp-finance-anomaly-ai.git
cd erp-finance-anomaly-ai

python -m venv .venv
.\.venv\Scripts\activate     # Windows

pip install -r requirements.txt
$env:OPENAI_API_KEY="你的API_KEY"
```

---

## 6.2 特徵工程

```bash
python src/data_prep/build_features.py
```

---

## 6.3 訓練與推論 ML 模型

```bash
python src/models/train_anomaly_model.py
python src/models/run_inference.py
```

---

## 6.4 執行 Streamlit Demo

```bash
streamlit run app/streamlit_app.py
```

包含：

* Dashboard + 財務風險圖表
* 發票異常檢視
* 供應商風險 LLM 報告
* CFO 管理報告（重點版＋完整版）

---

# 7. 模型與多代理可解釋性設計

## 7.1 異常偵測模型（ML）

Isolation Forest 會捕捉：

* 延遲付款異常
* 超額付款
* 金額偏離供應商歷史分布
* 多筆分期付款異常
* 異常的大額發票

輸出：

* `anomaly_flag`
* `anomaly_score`

---

## 7.2 異常分析 Agent（LLM）

輸出兩版內容：

### 【重點版】

* 今日最重要的異常類型
* 最可疑的發票
* 對財務流程的風險

### 【完整版】

* 逐類型深入說明
* 異常可能成因
* 需要追蹤的建議

---

## 7.3 供應商風險 Agent（LLM）

評估：

* 異常比例
* 逾期行為
* 平均付款延遲天數
* 金額異常程度

輸出：

* 供應商風險排名
* 高風險原因
* 財務與採購部門後續建議

---

## 7.4 CFO 報告 Agent

整合所有資料，輸出：

* **CFO 級別的重點（1分鐘可讀）**
* **深度完整版（用於會議/呈報）**

內容包含：

1. 異常發票概況
2. 供應商風險
3. 財務與付款行為健康度
4. 行動建議

---

# 8. 企業價值（Business Impact）

### **① 財務異常自動化檢測**

減少手動查核異常發票的時間
避免遺漏逾期付款、奇怪的金額、超付錯付。

---

### **② 供應商風險透明化**

可即時發現：

* 付款行為異常的供應商
* 過度集中或高風險供應商
* 需要控管信用的對象

---

### **③ CFO 即時決策支援**

一鍵產生：

* 今日財務風險總覽
* 重要異常
* 供應商異常
* 建議行動

---

### **④ 減少人工審查時間**

財務部門處理大量發票時，自動排序異常優先度。

---

### **⑤ 易於整合到企業內部系統（ERP / AP / SCM）**

可擴展至：

* AP 審核流程
* AR 逾期預測
* 採購付款策略
* 財務報表自動生成
