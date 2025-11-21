from anomaly_analyzer_agent import run_anomaly_analyzer
from vendor_risk_agent import run_vendor_risk_agent
from cfo_report_agent import run_cfo_report_agent


def main():
    print("Step 1: 分析異常發票...")
    run_anomaly_analyzer()

    print("\nStep 2: 分析供應商風險...")
    run_vendor_risk_agent()

    print("\nStep 3: 產出 CFO 管理報告...")
    run_cfo_report_agent()

    print("\n全部完成！請到 data/processed/ 底下查看：")
    print("- anomaly_analysis_report.txt")
    print("- vendor_risk_report.txt")
    print("- cfo_finance_risk_report.txt")


if __name__ == "__main__":
    main()
