# File: 1_feature_engineering_pro_final_v7.py
# MỤC ĐÍCH: Phiên bản cuối cùng, sửa lỗi KeyError bằng cách kiểm tra
# sự tồn tại của tất cả các cột FA trước khi sử dụng.

import pandas as pd
from pandas import json_normalize
import numpy as np
from FiinQuantX import FiinSession
import warnings
from tqdm import tqdm
import time

warnings.filterwarnings("ignore")

# ----------------------------
# 0. Cấu hình
# ----------------------------
USERNAME = "DSTC_35@fiinquant.vn"
PASSWORD = "Fiinquant0606"
FROM_DATE = "2018-01-01"
TO_DATE = "2025-08-28"
OUTPUT_FILE = "stock_features_pro_final.parquet"
START_YEAR_FA = 2017

# Tham số Target
SUCCESS_RETURN_THRESHOLD = 0.5
YH_YEARS = 2

# ----------------------------
# 1. Lấy dữ liệu
# ----------------------------
print("--- Bước 1: Đăng nhập và lấy dữ liệu cho toàn bộ thị trường ---")
client = FiinSession(username=USERNAME, password=PASSWORD).login()
if not client: raise RuntimeError("Đăng nhập thất bại.")

hose = client.TickerList(ticker="VNINDEX"); hnx = client.TickerList(ticker="HNXINDEX"); upcom = client.TickerList(ticker="UPCOMINDEX")
tickers = list(hose) + list(hnx) + list(upcom)
df_tickers = [t for t in tickers if 'FU' not in t and 'EIB' not in t and t != 'VNINDEX']
print(f"Sử dụng {len(df_tickers)} tickers để xử lý.")

price_df = client.Fetch_Trading_Data(
    realtime=False, tickers=df_tickers, fields=["close", "volume", "high", "low", "open"],
    adjusted=True, by='1d', from_date=FROM_DATE, to_date=TO_DATE
).get_data()
price_df['date'] = pd.to_datetime(price_df['timestamp'])
price_df = price_df.sort_values(['ticker', 'date']).reset_index(drop=True)
price_df['turnover'] = price_df['close'] * price_df['volume']
price_df['avg_turnover'] = price_df.groupby('ticker')['turnover'].transform(lambda x: x.rolling(21).mean())
print(f"Tải dữ liệu giá và tính turnover thành công. Kích thước: {price_df.shape}")

print("Đang lấy dữ liệu chỉ số FA hàng quý...")
latest_year = pd.Timestamp.now().year; all_fa_ratios = []
for year in tqdm(range(START_YEAR_FA, latest_year + 1), desc="Fetching FA Ratios"):
    for i in range(0, len(df_tickers), 200):
        subset_tickers = df_tickers[i:i+200]
        try:
            fa_quarterly = client.FundamentalAnalysis().get_ratios(
                tickers=subset_tickers, TimeFilter="Quarterly", LatestYear=year, NumberOfPeriod=4, Consolidated=True
            )
            if fa_quarterly and isinstance(fa_quarterly, list): all_fa_ratios.extend(fa_quarterly)
            time.sleep(1)
        except Exception as e:
            print(f"\nCẢNH BÁO: Không thể lấy dữ liệu FA cho lô bắt đầu bằng {subset_tickers[0]} năm {year}. Lỗi: {e}. Bỏ qua lô này.")
            continue

if not all_fa_ratios: raise RuntimeError("Không có dữ liệu FA Ratios nào được trả về.")
fa_raw_df = pd.DataFrame(all_fa_ratios)
fa_ratios_normalized = json_normalize(fa_raw_df['ratios'])
fa_ratios_normalized.columns = [col.replace('.', '_').lower() for col in fa_ratios_normalized.columns]
fa_ratios_df = pd.concat([fa_raw_df.drop(columns=['ratios', 'organizationId']), fa_ratios_normalized], axis=1)

rename_map = {
    'valuationratios_pricetoearning': 'pe', 'valuationratios_pricetobook': 'pb', 'valuationratios_pricetosales': 'ps',
    'profitabilityratio_roe': 'roe', 'profitabilityratio_roa': 'roa', 'valuationratios_basiceps': 'eps',
    'solvencyratio_debttoequityratio': 'debttoequity', 'currentratio': 'currentratio',
    'profitabilityratio_grossmargin': 'grossmargin'
}
cols_to_rename = {k: v for k, v in rename_map.items() if k in fa_ratios_df.columns}
fa_ratios_df.rename(columns=cols_to_rename, inplace=True)

fa_ratios_df['date'] = pd.to_datetime(fa_ratios_df['year'].astype(str) + 'Q' + fa_ratios_df['quarter'].astype(str))
fa_ratios_df = fa_ratios_df.sort_values(['ticker', 'date']).drop_duplicates(subset=['ticker', 'date']).reset_index(drop=True)
print(f"Tải chỉ số FA lịch sử thành công. Kích thước: {fa_ratios_df.shape}")

# ----------------------------
# 2. Gộp và Xây dựng Features
# ----------------------------
print("\n--- Bước 2: Gộp dữ liệu và xây dựng Features ---")
df = pd.merge_asof(price_df.sort_values('date'), fa_ratios_df.sort_values('date'), on='date', by='ticker', direction='backward')
required_cols = ['pe', 'pb', 'roe', 'eps', 'avg_turnover']
df.dropna(subset=[col for col in required_cols if col in df.columns], inplace=True)
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
g_ticker = df.groupby('ticker')

# --- SỬA LỖI: Kiểm tra sự tồn tại của từng cột trước khi sử dụng ---
if 'pe' in df.columns: df['value_pe'] = df['pe']
if 'pb' in df.columns: df['value_pb'] = df['pb']
if 'ps' in df.columns: df['value_ps'] = df['ps']
if 'roe' in df.columns: df['quality_roe'] = df['roe']
if 'roa' in df.columns: df['quality_roa'] = df['roa']
if 'debttoequity' in df.columns: df['quality_debt_equity'] = df['debttoequity']
if 'currentratio' in df.columns: df['quality_current_ratio'] = df['currentratio']
if 'grossmargin' in df.columns: df['quality_gross_margin'] = df['grossmargin']
if 'eps' in df.columns: df['growth_eps_yoy'] = g_ticker['eps'].transform(lambda x: x.pct_change(4))
if 'roe' in df.columns: df['quality_roe_stability_3y'] = g_ticker['roe'].transform(lambda x: x.rolling(12, min_periods=8).std())
df['tech_mom_120d'] = g_ticker['close'].transform(lambda x: x.pct_change(120))
df['tech_vol_60d'] = g_ticker['close'].transform(lambda x: x.pct_change().rolling(60).std())

# ----------------------------
# 3. Lọc, Định nghĩa Target và Lưu
# ----------------------------
print("\n--- Bước 3: Lọc, Định nghĩa Target và Lưu ---")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
feature_cols = [c for c in df.columns if c.startswith(('value_', 'quality_', 'growth_', 'tech_'))]
df.dropna(subset=feature_cols, inplace=True)

df_filtered = df[(df['avg_turnover'] > 1_000_000_000) & (df['close'] > 5000)].copy()
print(f"Kích thước sau khi lọc thanh khoản: {df_filtered.shape}")
if df_filtered.empty: raise RuntimeError("Không có mã nào qua bộ lọc thanh khoản.")

lookahead_days = int(YH_YEARS * 252)
future_prices = df_filtered.groupby('ticker')['close'].shift(-1)
df_filtered['future_max_return'] = (future_prices.rolling(lookahead_days, min_periods=120).max() / df_filtered['close']) - 1
    
value_rank_cols = [col for col in ['value_pe', 'value_pb', 'value_ps'] if col in df_filtered.columns]
if value_rank_cols: df_filtered['value_rank'] = df_filtered[value_rank_cols].rank(pct=True).mean(axis=1)

quality_rank_cols = [col for col in ['quality_roe', 'quality_gross_margin'] if col in df_filtered.columns]
if quality_rank_cols: df_filtered['quality_rank'] = df_filtered[quality_rank_cols].rank(pct=True, ascending=False).mean(axis=1)

if 'value_rank' in df_filtered.columns and 'quality_rank' in df_filtered.columns:
    df_filtered['is_good_candidate'] = (df_filtered['value_rank'] <= 0.3) & (df_filtered['quality_rank'] <= 0.3)
    df_filtered['is_successful'] = df_filtered['future_max_return'] >= SUCCESS_RETURN_THRESHOLD
    df_filtered['target'] = (df_filtered['is_good_candidate'] & df_filtered['is_successful']).astype(int)
    
    df_final = df_filtered.dropna(subset=['target'])
    
    if df_final.empty: raise RuntimeError("Không có điểm dữ liệu nào thỏa mãn điều kiện target.")
    print("Phân phối target:")
    print(df_final['target'].value_counts(normalize=True))
            
    final_feature_cols = [c for c in df_final.columns if c.startswith(('value_', 'quality_', 'growth_', 'tech_'))]
    final_cols_to_save = ['date', 'ticker', 'close', 'volume', 'target'] + final_feature_cols
    df_final[final_cols_to_save].to_parquet(OUTPUT_FILE, index=False)

    print(f"\n--- HOÀN TẤT ---")
    print(f"Đã lưu file đã xử lý vào: {OUTPUT_FILE}")
    print(f"Đã xây dựng thành công {len(final_feature_cols)} features.")
    print(f"Kích thước cuối cùng: {df_final.shape}")
else:
    raise RuntimeError("Không đủ cột 'value_rank' hoặc 'quality_rank' để tạo target.")