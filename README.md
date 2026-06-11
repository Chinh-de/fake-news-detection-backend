---
title: Fake News Detection Backend
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "latest"
app_file: Dockerfile
pinned: false
---

# Fake News Detection Backend (FastAPI)

Bộ mã nguồn máy chủ (Backend) thuộc hệ thống kiểm chứng tin giả **Fake News Detection**, được viết bằng **FastAPI** và kết nối cơ sở dữ liệu **PostgreSQL**.

Hệ thống kết hợp các mô hình:
1. **PhoBERT (SLM)**: Phân tích sâu ngữ pháp, từ ngữ và phong cách viết để phát hiện tin giả.
2. **XGBoost Classifier**: Phân tích bổ trợ sử dụng kết hợp đặc trưng TF-IDF và vector nhúng CLS từ mô hình PhoBERT base.
3. **RAG + LLM (Qwen)**: Tự động tra cứu tìm kiếm chéo tin tức liên quan trên Internet & Wikipedia để làm bằng chứng xác thực nội dung, đưa ra lập luận giải thích logic.

---

## 🛠️ Công nghệ sử dụng
- **FastAPI**: Khung phát triển ứng dụng Web API bất đồng bộ hiệu năng cao.
- **SQLAlchemy (Async)** & **asyncpg**: Giao tiếp cơ sở dữ liệu PostgreSQL bất đồng bộ.
- **Transformers (PyTorch)**: Chạy mô hình học sâu PhoBERT.
- **XGBoost** & **scikit-learn**: Chạy mô hình phân loại phụ XGBoost.
- **Underthesea**: Tách từ, tiền xử lý văn bản tiếng Việt.
- **Hugging Face Hub**: Tải và quản lý phiên bản mô hình trực tiếp từ đám mây.

---

## 📋 Yêu cầu hệ thống
- **Python**: Phiên bản `3.10` trở lên.
- **PostgreSQL**: Phiên bản `15` trở lên (Khuyến nghị chạy qua Docker Compose).
- **Docker & Docker Compose** (Tùy chọn).

---

## 🚀 Hướng dẫn cài đặt & Chạy cục bộ (Local)

### Bước 1: Clone kho lưu trữ và truy cập thư mục Backend
```bash
cd Backend
```

### Bước 2: Thiết lập môi trường ảo Python
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (Windows PowerShell)
.venv\Scripts\activate

# Kích hoạt môi trường ảo (Linux / macOS)
source .venv/bin/activate
```

### Bước 3: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 4: Thiết lập cấu hình tệp môi trường `.env`
Sao chép tệp mẫu cấu hình và chỉnh sửa các tham số kết nối database cũng như API keys:
```bash
cp .env.example .env
```
*Lưu ý: Điền cấu hình cơ sở dữ liệu và API key dịch vụ LLM nếu cần chạy luồng RAG nâng cao.*

### Bước 5: Khởi động Cơ sở dữ liệu (PostgreSQL)
Cách dễ nhất là sử dụng Docker Compose có sẵn trong thư mục để khởi động nhanh Postgres trên cổng `5433`:
```bash
docker-compose up -d
```

### Bước 6: Khởi chạy Máy chủ Backend
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- API sẽ chạy tại: `http://localhost:8000`
- Tài liệu kiểm thử API tự động (Swagger UI): `http://localhost:8000/docs`

---

## 🧪 Chạy Kiểm thử (Tests)
Bạn có thể chạy các tệp test tích hợp có sẵn để kiểm tra hoạt động của cơ sở dữ liệu, luồng download model và kết quả dự đoán của PhoBERT + XGBoost:
```bash
# Thiết lập encoding UTF-8 cho terminal (Windows)
$env:PYTHONIOENCODING='utf-8'

# Chạy test kết nối và dự đoán hệ thống
pytest
```

---

## ☁️ Triển khai lên Hugging Face Space
Mã nguồn này được thiết kế để tích hợp triển khai tự động lên Hugging Face Spaces (sử dụng Docker SDK):
1. Thiết lập `HF_TOKEN` trong Secrets của GitHub Actions để đẩy code tự động.
2. Cấu hình các biến môi trường (`DATABASE_URL`, `LLM_API_KEY`,...) trong cài đặt của Space (Hugging Face Console).
3. Ứng dụng sẽ tự động tải các mô hình PhoBERT và XGBoost từ các repository `chinhde/fake-news-detection-slm` và `chinhde/xgboost_model` về phân vùng `/tmp` trên Space khi khởi chạy.
