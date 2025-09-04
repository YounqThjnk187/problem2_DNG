# File: 2_model_training_final.py
# MỤC ĐÍCH: Huấn luyện mô hình với bộ features chuyên nghiệp từ file đã tạo.

import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# ----------------------------
# 1. Cấu hình
# ----------------------------
INPUT_FILE = "stock_features_pro_final.parquet" # SỬ DỤNG FILE BẠN ĐÃ TẠO
MODEL_OUTPUT_DIR = "trained_models_pro_final"
N_SPLITS = 5
RANDOM_STATE = 42
GAP_DAYS = 21 # Khoảng trống 1 tháng

if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)

# ----------------------------
# 2. Tải và Chuẩn bị dữ liệu
# ----------------------------
print("--- Bước 1: Tải và chuẩn bị dữ liệu cho mô hình ---")
df = pd.read_parquet(INPUT_FILE)
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

# Lấy danh sách features tự động từ các tiền tố
feature_cols = [c for c in df.columns if c.startswith(('value_', 'quality_', 'growth_', 'tech_', 'flow_', 'industry_'))]
target_col = 'target'; base_cols = ['date', 'ticker']

# Áp dụng logic tách rời (GAP)
features_df = df[base_cols + feature_cols]
target_df = df[base_cols + [target_col]].copy()
target_df['date'] = target_df['date'] - pd.to_timedelta(GAP_DAYS, unit='d')
final_df = pd.merge(features_df, target_df, on=['date', 'ticker'], how='inner')
print(f"Tải và xử lý dữ liệu thành công. Kích thước cuối cùng: {final_df.shape}")

# Sắp xếp lại lần cuối để TimeSeriesSplit hoạt động đúng
final_df = final_df.sort_values('date').reset_index(drop=True)
X = final_df[feature_cols]; y = final_df[target_col]

print(f"Sử dụng {len(feature_cols)} features để huấn luyện.")

# ----------------------------
# 3. Huấn luyện Walk-Forward để Đánh giá (OOF)
# ----------------------------
print("\n--- Bước 2: Bắt đầu Walk-Forward Cross-Validation để đánh giá ---")
tss = TimeSeriesSplit(n_splits=N_SPLITS)
oof_preds, oof_indices, auc_scores = [], [], []

for fold, (train_idx, test_idx) in enumerate(tss.split(X, y)):
    print(f"\n--- FOLD {fold+1}/{N_SPLITS} ---")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_dates = final_df['date'].iloc[train_idx]; test_dates = final_df['date'].iloc[test_idx]
    print(f"Train: {train_dates.min().date()} -> {train_dates.max().date()}")
    print(f"Test:  {test_dates.min().date()} -> {test_dates.max().date()}")
    
    scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
    
    model = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
        validation_fraction=0.1, n_iter_no_change=30,
        random_state=RANDOM_STATE, class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)

    preds_proba = model.predict_proba(X_test_scaled)[:,1]
    oof_preds.extend(preds_proba)
    oof_indices.extend(test_idx)

    if len(np.unique(y_test)) > 1:
        fold_auc = roc_auc_score(y_test, preds_proba)
        fold_prec = precision_score(y_test, (preds_proba>0.5).astype(int))
        fold_recall = recall_score(y_test, (preds_proba>0.5).astype(int))
        fold_f1 = f1_score(y_test, (preds_proba>0.5).astype(int))
        auc_scores.append(fold_auc)
        print(f"AUC: {fold_auc:.4f} | Precision: {fold_prec:.4f} | Recall: {fold_recall:.4f} | F1: {fold_f1:.4f}")
    else:
        print("Chỉ có 1 class trong test, bỏ qua đánh giá fold này.")

# ----------------------------
# 4. Lưu kết quả OOF và Tối ưu hóa Ngưỡng
# ----------------------------
oof_df = final_df.iloc[oof_indices].copy()
oof_df['prediction_proba'] = oof_preds
oof_df[['date','ticker','target','prediction_proba']].to_csv(
    os.path.join(MODEL_OUTPUT_DIR,"oof_predictions.csv"), index=False)
print("\nĐã lưu kết quả dự đoán Out-of-Fold (OOF).")

if auc_scores:
    print(f"Trung bình AUC các fold hợp lệ: {np.mean(auc_scores):.4f}")

# Tối ưu hóa ngưỡng dựa trên OOF
precision, recall, thresholds = precision_recall_curve(oof_df['target'], oof_df['prediction_proba'])
thresholds = np.append(thresholds, 1) 
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Ngưỡng xác suất tối ưu theo F1-score trên OOF: {best_threshold:.4f}")

joblib.dump(best_threshold, os.path.join(MODEL_OUTPUT_DIR,"best_threshold.joblib"))
print("Đã lưu ngưỡng tối ưu.")

# ----------------------------
# 5. Huấn luyện model cuối cùng trên TOÀN BỘ dữ liệu
# ----------------------------
print("\n--- Bước 3: Huấn luyện model cuối cùng trên toàn bộ dữ liệu ---")
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)

final_model = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
    validation_fraction=0.1, n_iter_no_change=30,
    random_state=RANDOM_STATE, class_weight='balanced'
)
final_model.fit(X_scaled, y)

joblib.dump(final_model, os.path.join(MODEL_OUTPUT_DIR,"final_model.joblib"))
joblib.dump(final_scaler, os.path.join(MODEL_OUTPUT_DIR,"final_scaler.joblib"))
joblib.dump(feature_cols, os.path.join(MODEL_OUTPUT_DIR,"feature_cols.joblib"))
print("Đã lưu model, scaler và danh sách features cuối cùng.")