# 🚀 problem2_DNG DSTC 2025: Xây Dựng Chiến Lược Đầu Tư Định Lượng Lai Dựa Trên Học Máy

## 🎯 Giới Thiệu

Đây là bài dự thi **DSTC 2025**, nơi chúng tôi trình bày một hệ thống **sàng lọc & giao dịch cổ phiếu định lượng hoàn chỉnh** cho thị trường chứng khoán Việt Nam.

Dự án được lấy cảm hứng từ nghiên cứu *"Machine learning-based stock picking using value investing and quality features"* của **Priel & Rokach (2024)**, nhưng đã được **“Việt hóa” và tối ưu** để phù hợp với dữ liệu cũng như giới hạn của **API FiinQuant**.

✨ Điểm đặc biệt: Thay vì cố gắng dự đoán giá ngắn hạn (vốn nhiều nhiễu và rủi ro), hệ thống tập trung trả lời một câu hỏi cốt lõi:

👉 **“Liệu cổ phiếu này có đang rẻ + chất lượng, và có khả năng bứt phá >50% trong 2 năm tới không?”**

## 🧩 Chiến Lược Hybrid – 3 Trụ Cột

1. **Giá trị (Value):** Săn tìm cổ phiếu đang bị thị trường “bỏ quên”.
2. **Chất lượng (Quality):** Ưu tiên doanh nghiệp nền tảng tài chính vững chắc.
3. **Kỹ thuật (Technical):** Nắm bắt động lượng giá & biến động để chọn thời điểm.

Kết hợp cả ba yếu tố này, hệ thống vừa có **tư duy dài hạn** của nhà đầu tư giá trị, vừa có **sự nhạy bén ngắn hạn** của phân tích kỹ thuật, lại được **học máy** hỗ trợ tối ưu.

## 📂 Cấu Trúc Dự Án

Dự án gồm **3 file Python chính**, chạy theo thứ tự:

1. `1_get_feature.py` → Thu thập & tạo bộ **features + target**.
2. `2_training_model.py` → Huấn luyện mô hình `HistGradientBoostingClassifier` với **Time Series Cross-Validation**.
3. `3_backtest_simulation.py` → Backtest chiến lược + trực quan hóa hiệu suất.
4. `4_analyzis.py` → Phân tích Feature Importance.

## ⚙️ Hướng Dẫn Cài Đặt

### 1️⃣ Yêu Cầu Hệ Thống

* Python **3.10+**
* Thư viện trong `requirements.txt`

### 2️⃣ Cài Đặt Thư Viện

```bash
pip install pandas numpy scikit-learn tqdm joblib matplotlib fiinquantx
```

### 3️⃣ Cấu Hình Tài Khoản

Trong từng file `.py`, điền thông tin đăng nhập API FiinQuant:

```python
USERNAME = "DSTC_35@fiinquant.vn"
PASSWORD = "Fiinquant0606"
```

### 4️⃣ Quy Trình Chạy

🔥 Thực hiện **tuần tự 3 bước**:

#### **Bước 1: Tạo Dữ Liệu**

```bash
python 1_get_feature.py
```

* Đầu vào: dữ liệu từ API FiinQuant.
* Đầu ra: `stock_features_pro_final.parquet`.
  ⚠️ Quá trình này **có thể >1h**, hãy chuẩn bị kết nối mạng ổn định.

#### **Bước 2: Huấn Luyện Mô Hình**

```bash
python 2_training_model.py
```

* Đầu ra: thư mục `trained_models_pro_final` chứa model + scaler + features.
* Đồng thời tạo file `oof_predictions_pro_final.csv` cho backtest.

#### **Bước 3: Backtest Chiến Lược**

```bash
python 3_backtest_simulation.py
```

* Trực quan hóa **Equity Curve & Drawdown** so với VN-Index.
* In báo cáo hiệu suất chi tiết.

#### **Bước 4: Phân tích Feature Importance**

```bash
python 4_analyzis.py
```

* Xếp hạng trung bình các yếu tố (giống Figure 7).
* Sự thay đổi của từng yếu tố theo thời gian (giống Figure 10).
---

## 🎉 Kết

Với dự án này, chúng tôi không chỉ xây dựng một mô hình học máy, mà còn tạo ra một **hệ thống giao dịch định lượng có tính ứng dụng thực tế** cho thị trường Việt Nam.

🔮 Biết đâu trong tương lai, đây sẽ là một “chìa khóa vàng” cho nhà đầu tư thông minh!
