# Ứng Dụng Tối Ưu Hóa Danh Mục Đầu Tư Bằng Học Máy (LSTM-GRU)

Ứng dụng Streamlit này được xây dựng dựa trên nghiên cứu và mã nguồn từ Jupyter Notebook tối ưu hóa danh mục đầu tư cổ phiếu Việt Nam bằng mạng neural lai LSTM-GRU và hàm mất mát Sharpe cải tiến (Sharpe Loss với Entropy Regularization).

## Các Tính Năng Chính
1. **Tải Dữ Liệu Tự Động**: Sử dụng thư viện `vnstock` để tải dữ liệu lịch sử giá cổ phiếu của các ngành tại Việt Nam (Thép, Ngân hàng, Bất động sản, Chứng khoán, Công nghệ, v.v.).
2. **Lọc Cổ Phiếu Tốt Nhất**: Tính toán tỷ số Sharpe lịch sử của từng mã và tự động chọn ra Top N cổ phiếu (mặc định là 10) có hiệu suất tốt nhất để tối ưu.
3. **Mô Hình Học Máy LSTM-GRU**: 
   - Tự động tạo đặc trưng (Features) như Daily Returns, Multi-day Returns, MA Ratios, Volatility và chỉ báo kỹ thuật RSI.
   - Huấn luyện mô hình mạng neural LSTM-GRU để dự đoán trọng số tối ưu trực tiếp bằng cách tối đa hóa tỷ số Sharpe (Sharpe Loss) và phân bổ đều tài sản (Entropy Regularization).
   - Huấn luyện đồng thời trên nhiều seed khác nhau để tìm ra mô hình tốt nhất trên tập kiểm thử (Test set).
4. **Trực Quan Hóa Trực Quan**: 
   - Biểu đồ phân bổ trọng số danh mục đầu tư (Pie chart và Bar chart tương tác).
   - Biểu đồ so sánh hiệu suất sinh lời tích lũy (Cumulative Returns) của danh mục tối ưu so với danh mục phân bổ đều (Equal-Weighted 1/N) và các mã cổ phiếu riêng lẻ.
   - Bảng so sánh các chỉ số hiệu suất: Tỷ suất sinh lời năm (Annualized Return), Độ lệch chuẩn năm (Annualized Volatility) và Tỷ số Sharpe.

## Hướng Dẫn Cài Đặt và Chạy Local

### 1. Chuẩn bị môi trường
Yêu cầu Python từ `3.9` đến `3.11`. Nên tạo môi trường ảo (virtual environment):

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (Windows)
.venv\Scripts\activate

# Kích hoạt môi trường ảo (macOS/Linux)
source .venv/bin/activate
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng
```bash
streamlit run app.py
```

---

## Hướng Dẫn Deploy Lên Streamlit Share (Streamlit Community Cloud)

Để deploy ứng dụng của bạn lên cloud miễn phí của Streamlit, hãy thực hiện các bước sau:

1. **Đưa mã nguồn lên GitHub**:
   - Tạo một repository mới trên GitHub (ví dụ: `vietnam-portfolio-optimization`).
   - Push toàn bộ các tệp tin trong thư mục này lên GitHub (bao gồm `app.py`, `industry_tickers.py`, `requirements.txt`, và `README.md`).
2. **Đăng nhập vào Streamlit Community Cloud**:
   - Truy cập trang web [share.streamlit.io](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub của bạn.
3. **Deploy ứng dụng**:
   - Nhấn vào nút **New app**.
   - Chọn Repository, Branch (thường là `main` hoặc `master`), và Main file path (nhập `app.py`).
   - Nhấn **Deploy!** Streamlit sẽ tự động cài đặt các thư viện từ `requirements.txt` và khởi chạy ứng dụng của bạn trong vài phút.

---

## Cấu Trúc Thư Mục Dự Án
- `app.py`: Tệp mã nguồn chính chứa giao diện Streamlit và logic xử lý học máy.
- `industry_tickers.py`: Chứa danh sách các mã cổ phiếu phân nhóm theo ngành tại thị trường Việt Nam.
- `requirements.txt`: Danh sách các thư viện cần cài đặt để chạy ứng dụng.
- `README.md`: Hướng dẫn này.
