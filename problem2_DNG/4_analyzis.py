# File: analyze_feature_importance.py
# MỤC ĐÍCH: Nạp dữ liệu feature importance đã được lưu và tạo ra các biểu đồ
#           phân tích chuyên sâu, tái tạo lại Figure 7 và Figure 10 của
#           nghiên cứu Caparrini et al.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

# --- CẤU HÌNH ---
MODEL_DIR = "trained_models_pro_final"
IMPORTANCE_FILE = "feature_importance_over_time.csv"
OUTPUT_DIR = "results_analysis" # Thư mục riêng cho các biểu đồ phân tích

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- BƯỚC 1: NẠP DỮ LIỆU ---
print("--- Bước 1: Nạp dữ liệu Feature Importance ---")
file_path = os.path.join(MODEL_DIR, IMPORTANCE_FILE)
try:
    importance_df = pd.read_csv(file_path)
    importance_df['fold_end_date'] = pd.to_datetime(importance_df['fold_end_date'])
    print("Nạp dữ liệu thành công.")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file '{file_path}'. Vui lòng chạy script huấn luyện trước.")
    exit()

# --- BƯỚC 2: TÁI TẠO FIGURE 7 - TẦM QUAN TRỌNG TRUNG BÌNH CỦA CÁC YẾU TỐ ---
print("\n--- Bước 2: Tạo biểu đồ Tầm quan trọng Trung bình (Giống Figure 7) ---")

# Tính toán giá trị trung bình và độ lệch chuẩn cho mỗi feature
agg_importance = importance_df.groupby('feature').agg(
    mean_importance=('importance_mean', 'mean'),
    std_importance=('importance_mean', 'std')
).reset_index()

# Sắp xếp các features theo tầm quan trọng trung bình giảm dần
agg_importance = agg_importance.sort_values('mean_importance', ascending=False)

# Vẽ biểu đồ
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(16, 9))

# Sử dụng màu xanh giống nghiên cứu
bar_color = 'royalblue'
ax1.bar(
    x=agg_importance['feature'],
    height=agg_importance['mean_importance'],
    yerr=agg_importance['std_importance'], # Thêm thanh lỗi (error bars)
    capsize=5, # Kích thước của "mũ" ở đầu thanh lỗi
    color=bar_color,
    edgecolor='black',
    alpha=0.75
)

ax1.set_title('Xếp hạng Tầm quan trọng của các Yếu tố (Trung bình qua các Thời kỳ)', fontsize=18, pad=20)
ax1.set_ylabel('Permutation Importance (Mean)', fontsize=14)
ax1.set_xlabel('Feature', fontsize=14)

# Xoay nhãn trục X để dễ đọc
plt.xticks(rotation=45, ha='right')

# Tăng khoảng trống để nhãn không bị cắt
plt.margins(x=0.01)
plt.tight_layout()

# Lưu biểu đồ
fig1_path = os.path.join(OUTPUT_DIR, "fig7_mean_feature_importance.png")
fig1.savefig(fig1_path, dpi=300)
print(f"Đã lưu biểu đồ giống Figure 7 vào: {fig1_path}")
plt.show()


# --- BƯỚC 3: TÁI TẠO FIGURE 10 - SỰ THAY ĐỔI CỦA TỪNG YẾU TỐ THEO THỜI GIAN ---
print("\n--- Bước 3: Tạo biểu đồ Lưới về sự thay đổi của từng Yếu tố (Giống Figure 10) ---")

# Lấy danh sách các features duy nhất
features = importance_df['feature'].unique()
num_features = len(features)

# Thiết lập lưới biểu đồ (subplot grid)
# Chúng ta sẽ tạo một lưới 4 cột, số hàng sẽ được tính tự động
num_cols = 4
num_rows = math.ceil(num_features / num_cols)

fig2, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4), sharey=False)
# Làm phẳng mảng axes để dễ dàng lặp qua
axes = axes.flatten()

# Lặp qua từng feature và vẽ biểu đồ đường tương ứng
for i, feature in enumerate(features):
    ax = axes[i]
    
    # Lọc dữ liệu cho feature hiện tại
    feature_data = importance_df[importance_df['feature'] == feature]
    
    # Vẽ biểu đồ đường
    ax.plot(feature_data['fold_end_date'], feature_data['importance_mean'], marker='o', markersize=3, linestyle='-')
    
    # Lấy màu của đường vừa vẽ để tô bóng cho vùng độ lệch chuẩn
    line_color = ax.get_lines()[-1].get_color()
    
    # Thêm vùng thể hiện độ lệch chuẩn (standard deviation)
    ax.fill_between(
        feature_data['fold_end_date'],
        feature_data['importance_mean'] - feature_data['importance_std'],
        feature_data['importance_mean'] + feature_data['importance_std'],
        alpha=0.2,
        color=line_color
    )
    
    ax.set_title(feature, fontsize=10)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Ẩn các subplot không sử dụng (nếu có)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig2.suptitle('Sự thay đổi về Tầm quan trọng của Từng Yếu tố theo Thời gian', fontsize=24, y=1.02)
fig2.tight_layout(pad=3.0)

# Lưu biểu đồ
fig2_path = os.path.join(OUTPUT_DIR, "fig10_feature_importance_evolution.png")
fig2.savefig(fig2_path, dpi=300)
print(f"Đã lưu biểu đồ giống Figure 10 vào: {fig2_path}")
plt.show()

print("\n--- HOÀN TẤT ---")