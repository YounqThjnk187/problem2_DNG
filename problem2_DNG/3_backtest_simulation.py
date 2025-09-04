# File: 3_backtest_simulation_pro_final_v3.py
# MỤC ĐÍCH: Sửa lỗi FileNotFoundError khi đọc file dự đoán OOF.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from FiinQuantX import FiinSession
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ----------------------------
# 1. Cấu hình Backtest
# ----------------------------
PRICE_DATA_FILE = "stock_features_pro_final.parquet"
MODEL_DIR = "trained_models_pro_final" # Đảm bảo tên thư mục này đúng

# SỬA LỖI: Tên file không bao gồm tên thư mục
OOF_PREDICTIONS_FILE = "oof_predictions.csv" 

INITIAL_CAPITAL = 1_000_000_000
ALLOCATION_PER_STOCK = 100_000_000
MAX_POSITIONS = 10
TRANSACTION_COST = 0.0025

try:
    BUY_CONFIDENCE_THRESHOLD = joblib.load(os.path.join(MODEL_DIR, "best_threshold.joblib"))
    print(f"Sử dụng ngưỡng mua tối ưu đã được tính toán: {BUY_CONFIDENCE_THRESHOLD:.4f}")
except (FileNotFoundError, AttributeError):
    BUY_CONFIDENCE_THRESHOLD = 0.5399 # Sử dụng giá trị từ lần chạy trước
    print(f"Không tìm thấy threshold, sử dụng giá trị mặc định: {BUY_CONFIDENCE_THRESHOLD}")

MAX_HOLDING_PERIOD_YEARS = 2
MIN_PROFIT_FOR_TRAIL = 0.20
TRAIL_STOP_PERCENT = 0.15

# ----------------------------
# 2. Chuẩn bị dữ liệu
# ----------------------------
print("--- Bước 1: Chuẩn bị dữ liệu cho Backtest ---")
price_df = pd.read_parquet(PRICE_DATA_FILE)[['date', 'ticker', 'close']]
price_df['date'] = pd.to_datetime(price_df['date'])

# SỬA LỖI: Tạo đường dẫn đầy đủ đến file predictions
predictions_path = os.path.join(MODEL_DIR, OOF_PREDICTIONS_FILE)
try:
    predictions_df = pd.read_csv(predictions_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Không tìm thấy file dự đoán tại đường dẫn: {predictions_path}. Hãy đảm bảo File 2 đã chạy thành công.")

predictions_df['date'] = pd.to_datetime(predictions_df['date'])
price_pivot = price_df.pivot(index='date', columns='ticker', values='close')
signal_pivot = predictions_df.pivot(index='date', columns='ticker', values='prediction_proba')
common_dates = price_pivot.index.intersection(signal_pivot.index)
price_pivot = price_pivot.loc[common_dates]; signal_pivot = signal_pivot.loc[common_dates]
weekly_dates = price_pivot.asfreq('W-FRI').index; weekly_dates = weekly_dates.intersection(price_pivot.index)
price_pivot_weekly = price_pivot.loc[weekly_dates]; signal_pivot_weekly = signal_pivot.loc[weekly_dates]
print(f"Dữ liệu sẵn sàng để backtest HÀNG TUẦN từ {weekly_dates.min().date()} đến {weekly_dates.max().date()}")

vnindex_ma_df = None
try:
    client = FiinSession(username="DSTC_35@fiinquant.vn", password="Fiinquant0606").login()
    vnindex_raw = client.Fetch_Trading_Data(
        realtime=False, tickers=['VNINDEX'], fields=['close'], by='1d',
        from_date=(weekly_dates.min() - pd.Timedelta(days=300)).strftime('%Y-%m-%d'),
        to_date=weekly_dates.max().strftime('%Y-%m-%d')).get_data()
    vnindex_raw['date'] = pd.to_datetime(vnindex_raw['timestamp'])
    vnindex_df = vnindex_raw.set_index('date')['close']
    vnindex_ma_df = vnindex_df.to_frame(name='close')
    vnindex_ma_df['ma200'] = vnindex_ma_df['close'].rolling(200, min_periods=100).mean()
    print("Tải dữ liệu VN-Index và tính MA200 thành công.")
except Exception as e: print(f"Cảnh báo: Không thể tải dữ liệu VN-Index. Lỗi: {e}")

# ----------------------------
# 3. Chạy vòng lặp Backtest
# ----------------------------
print("\n--- Bước 2: Bắt đầu mô phỏng giao dịch hàng tuần ---")
# (Toàn bộ logic backtest giữ nguyên)
cash = INITIAL_CAPITAL; portfolio = {}; portfolio_history = []; trades_log = []; already_recommended = set()
for date in tqdm(price_pivot_weekly.index, desc="Weekly Backtesting"):
    current_prices = price_pivot_weekly.loc[date].dropna()
    portfolio_value = cash; tickers_to_sell = []
    for ticker, position in portfolio.items():
        if ticker in current_prices:
            current_price = current_prices[ticker]
            position['highest_price'] = max(position.get('highest_price', 0), current_price)
            portfolio_value += current_price * position['shares']
            holding_days = (date - position['entry_date']).days
            current_return = (current_price / position['entry_price']) - 1
            trailing_stop_price = position['highest_price'] * (1 - TRAIL_STOP_PERCENT)
            reason = ""
            if current_return >= MIN_PROFIT_FOR_TRAIL and current_price < trailing_stop_price: reason = "Trailing Stop"
            elif holding_days >= MAX_HOLDING_PERIOD_YEARS * 365: reason = "Max Hold Period"
            if reason: tickers_to_sell.append((ticker, reason, current_price, current_return))
    for ticker, reason, sell_price, pnl in tickers_to_sell:
        position = portfolio.pop(ticker); sell_value = sell_price * position['shares']
        cash += sell_value * (1 - TRANSACTION_COST)
        trades_log.append({'exit_date': date, 'ticker': ticker, 'pnl_percent': pnl, 'reason': reason})
    market_is_bullish = True
    if vnindex_ma_df is not None and date in vnindex_ma_df.index:
        if vnindex_ma_df.loc[date, 'close'] < vnindex_ma_df.loc[date, 'ma200']: market_is_bullish = False
    if market_is_bullish:
        weekly_signals = signal_pivot_weekly.loc[date].dropna()
        buy_candidates = weekly_signals[weekly_signals > BUY_CONFIDENCE_THRESHOLD]
        new_candidates = {t: p for t, p in buy_candidates.items() if t not in already_recommended and t not in portfolio}
        potential_buys = sorted(new_candidates.items(), key=lambda item: item[1], reverse=True)
        for ticker, proba in potential_buys:
            if len(portfolio) >= MAX_POSITIONS: break
            if cash >= ALLOCATION_PER_STOCK and ticker in current_prices:
                current_price = current_prices[ticker]
                shares_to_buy = int(ALLOCATION_PER_STOCK / current_price)
                if shares_to_buy > 0:
                    buy_value = shares_to_buy * current_price; cash -= buy_value * (1 + TRANSACTION_COST)
                    portfolio[ticker] = {'entry_price': current_price, 'shares': shares_to_buy, 'entry_date': date, 'highest_price': current_price}
                    already_recommended.add(ticker)
    final_portfolio_value = cash
    for ticker, position in portfolio.items():
        price = current_prices.get(ticker, position['entry_price'])
        final_portfolio_value += position['shares'] * price
    portfolio_history.append({'date': date, 'value': final_portfolio_value})

# ----------------------------
# 4. Phân tích kết quả
# ----------------------------
print("\n--- Bước 3: Phân tích kết quả Backtest ---")
# (Toàn bộ logic phân tích và vẽ biểu đồ giữ nguyên)
history_df = pd.DataFrame(portfolio_history).set_index('date')
trades_df = pd.DataFrame(trades_log)
final_value = history_df['value'].iloc[-1]; total_return = (final_value / INITIAL_CAPITAL) - 1
daily_returns = history_df['value'].pct_change().dropna()
sharpe_ratio = (daily_returns.mean() * 52) / (daily_returns.std() * np.sqrt(52)) if not daily_returns.empty else 0
rolling_max = history_df['value'].cummax(); daily_drawdown = (history_df['value'] / rolling_max) - 1
max_drawdown = daily_drawdown.min(); win_rate = (trades_df['pnl_percent'] > 0).mean() if not trades_df.empty else 0
avg_win = trades_df[trades_df['pnl_percent'] > 0]['pnl_percent'].mean() if not trades_df.empty else 0
avg_loss = trades_df[trades_df['pnl_percent'] <= 0]['pnl_percent'].mean() if not trades_df.empty else 0
profit_factor = trades_df[trades_df['pnl_percent'] > 0]['pnl_percent'].sum() / abs(trades_df[trades_df['pnl_percent'] <= 0]['pnl_percent'].sum()) if avg_loss != 0 else np.inf

print("\n--- KẾT QUẢ MÔ PHỎNG CUỐI CÙNG ---")
print(f"Giai đoạn: {history_df.index.min().date()} -> {history_df.index.max().date()}")
print(f"Vốn ban đầu: {INITIAL_CAPITAL:,.0f} VNĐ"); print(f"Vốn cuối kỳ: {final_value:,.0f} VNĐ")
print(f"Tổng lợi nhuận: {total_return:.2%}"); print(f"Mức sụt giảm tối đa (Max Drawdown): {max_drawdown:.2%}")
print(f"Tỷ lệ Sharpe (Hàng tuần): {sharpe_ratio:.2f}"); print("-" * 20); print("--- Thống kê Giao dịch ---")
print(f"Tổng số giao dịch: {len(trades_df)}"); print(f"Tỷ lệ thắng (Win Rate): {win_rate:.2%}")
print(f"Lãi trung bình/giao dịch thắng: {avg_win:.2%}"); print(f"Lỗ trung bình/giao dịch thua: {avg_loss:.2%}")
print(f"Tỷ lệ Lãi/Lỗ (Profit Factor): {profit_factor:.2f}")

# Vẽ biểu đồ
plt.style.use('seaborn-v0_8-darkgrid'); fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
ax1.plot(history_df.index, history_df['value'], label='Chiến lược', color='royalblue')
if vnindex_df is not None and not vnindex_df.empty:
    results = history_df.join(vnindex_df.rename('vnindex')).dropna()
    if not results.empty:
        results['vnindex_normalized'] = results['vnindex'] * (INITIAL_CAPITAL / results['vnindex'].iloc[0])
        ax1.plot(results.index, results['vnindex_normalized'], label='VN-Index (chuẩn hóa)', color='gray', linestyle='--')
ax1.set_title('Hiệu suất Chiến lược (Tối ưu hóa) vs. VN-Index', fontsize=16)
ax1.set_ylabel('Giá trị Danh mục (VNĐ)', fontsize=12); ax1.legend(fontsize=12); ax1.grid(True)
ax2.fill_between(daily_drawdown.index, daily_drawdown, 0, color='indianred', alpha=0.3)
ax2.set_title('Sụt giảm (Drawdown) của Chiến lược', fontsize=12); ax2.set_ylabel('Sụt giảm (%)', fontsize=12)
ax2.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format)); ax2.grid(True)
plt.tight_layout(); plt.show()
