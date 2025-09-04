import pandas as pd
import numpy as np

# Load file parquet
df = pd.read_parquet("data_pipeline.log")

# Lấy danh sách features (bỏ ticker, timestamp, date, close)
feature_cols = [c for c in df.columns if c not in ["ticker", "timestamp", "date", "close"]]

report = []
for col in feature_cols:
    series = df[col]
    nan_ratio = series.isna().mean()
    inf_ratio = np.isinf(series).mean()
    desc = series.describe(percentiles=[0.01,0.05,0.5,0.95,0.99])
    
    # Cảnh báo bất thường
    warnings = []
    if nan_ratio > 0.5:
        warnings.append("⚠️ Nhiều NaN (>50%)")
    if inf_ratio > 0:
        warnings.append("⚠️ Có Inf")
    if (desc["max"] > 1e3) or (desc["min"] < -1e3):
        warnings.append("⚠️ Giá trị cực trị bất thường")
    if "roe" in col.lower() or "roa" in col.lower() or "return" in col.lower():
        if desc["max"] > 5 or desc["min"] < -5:
            warnings.append("⚠️ ROE/Return bất thường (quá 500%)")
    if "pe" in col.lower():
        if desc["max"] > 100 or desc["min"] < -100:
            warnings.append("⚠️ P/E bất thường")
    if "current_ratio" in col.lower():
        if desc["min"] < 0 or desc["max"] > 100:
            warnings.append("⚠️ Current Ratio bất hợp lý")
    
    report.append({
        "feature": col,
        "nan_ratio": round(nan_ratio,3),
        "inf_ratio": round(inf_ratio,3),
        "min": desc["min"],
        "p1": desc["1%"],
        "p5": desc["5%"],
        "median": desc["50%"],
        "p95": desc["95%"],
        "p99": desc["99%"],
        "max": desc["max"],
        "warnings": "; ".join(warnings) if warnings else "✅ OK"
    })

report_df = pd.DataFrame(report)

# Xuất báo cáo
report_df.to_csv("feature_quality_report.csv", index=False)
report_df